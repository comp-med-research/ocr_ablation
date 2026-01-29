import json
import os
import re
import xml.etree.ElementTree as ET

PAGE_NS = {"pc": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"}

TYPE_RE = re.compile(r"type:([^;]+);")

# Map PAGE-XML region types to your Label Studio labels
TYPE_MAP = {
    "text": "paragraph",
    "paragraph": "paragraph",
    "header": "header",
    "footer": "footer",
    "table": "table_cell",
    "tablecell": "table_cell",
    "form": "form_field",
    "field": "form_field",
    "handwritten": "handwritten",
}

def parse_points(points_str: str):
    # "x1,y1 x2,y2 ..." -> [(x,y), ...]
    pts = []
    for token in points_str.strip().split():
        x, y = token.split(",")
        pts.append((float(x), float(y)))
    return pts

def bbox_from_points(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)

def to_pct(val, total):
    return (100.0 * val / total) if total else 0.0

def extract_region_type(custom_attr: str | None) -> str:
    # custom="structure {type:text;}" -> "text"
    if not custom_attr:
        return "paragraph"
    m = TYPE_RE.search(custom_attr)
    if not m:
        return "paragraph"
    raw = m.group(1).strip().lower()
    raw = raw.replace(" ", "")
    return TYPE_MAP.get(raw, "paragraph")

def extract_region_text(region_el) -> str:
    # More robust: prefer line-level, fall back to region-level. Use last TextEquiv if multiple.
    lines = []
    for line in region_el.findall(".//pc:TextLine", PAGE_NS):
        equivs = line.findall(".//pc:TextEquiv/pc:Unicode", PAGE_NS)
        if equivs:
            txt = (equivs[-1].text or "").strip()
            if txt:
                lines.append(txt)
    if lines:
        return "\n".join(lines).strip()

    equivs = region_el.findall("./pc:TextEquiv/pc:Unicode", PAGE_NS)
    if equivs:
        return (equivs[-1].text or "").strip()

    return ""

def pagexml_to_ls_results(pagexml_path: str):
    root = ET.parse(pagexml_path).getroot()

    page = root.find(".//pc:Page", PAGE_NS)
    if page is None:
        raise ValueError(f"No <Page> found in {pagexml_path}")

    img_w = float(page.get("imageWidth") or 0)
    img_h = float(page.get("imageHeight") or 0)

    results = []
    i = 0

    for region in root.findall(".//pc:TextRegion", PAGE_NS):
        coords = region.find("./pc:Coords", PAGE_NS)
        if coords is None or not coords.get("points"):
            continue

        pts = parse_points(coords.get("points"))
        x1, y1, x2, y2 = bbox_from_points(pts)

        rect_id = f"r{i}"
        text_id = f"t{i}"
        i += 1

        label = extract_region_type(region.get("custom"))
        region_text = extract_region_text(region)

        # Rectangle geometry in %
        x_pct = to_pct(x1, img_w)
        y_pct = to_pct(y1, img_h)
        w_pct = to_pct(x2 - x1, img_w)
        h_pct = to_pct(y2 - y1, img_h)

        # Rectangle
        results.append({
            "id": rect_id,
            "from_name": "region_type",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {
                "x": x_pct,
                "y": y_pct,
                "width": w_pct,
                "height": h_pct,
                "rectanglelabels": [label],
            }
        })

        # Per-region transcription
        # IMPORTANT: include x/y/width/height so LS reliably binds it to the region UI
        results.append({
            "id": text_id,
            "from_name": "transcription",
            "to_name": "image",
            "type": "textarea",
            "value": {
                "x": x_pct,
                "y": y_pct,
                "width": w_pct,
                "height": h_pct,
                "text": [region_text],
            },
            "parentID": rect_id
        })

    return results

def build_tasks(pagexml_dir: str, image_map_path: str, out_path: str):
    """
    image_map.json maps PAGE-XML filename -> Label Studio upload path
    Example:
      {"GP_notes_P5-13.pdf_page_5.xml": "/data/upload/9a3f2d-output_page-05.png"}
    """
    with open(image_map_path, "r", encoding="utf-8") as f:
        image_map = json.load(f)

    tasks = []
    for fname in sorted(os.listdir(pagexml_dir)):
        if not fname.lower().endswith(".xml"):
            continue
        if fname.lower() == "mets.xml":
            continue

        if fname not in image_map:
            raise KeyError(f"Missing image mapping for {fname} in {image_map_path}")

        px_path = os.path.join(pagexml_dir, fname)
        results = pagexml_to_ls_results(px_path)

        tasks.append({
            "data": {"image": image_map[fname]},
            # Keep as predictions (works for rectangles; textarea binding is fixed by geometry above)
            "predictions": [{
                "model_version": "escriptorium-pagexml",
                "result": results
            }],
            "meta": {"source_pagexml": fname}
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(tasks)} tasks to {out_path}")

if __name__ == "__main__":
    build_tasks(
        pagexml_dir="pagexml",
        image_map_path="image_map.json",
        out_path="labelstudio_tasks_with_preds.json"
    )
