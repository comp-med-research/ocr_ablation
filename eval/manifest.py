"""Build a task/region manifest from a Label Studio export JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _textarea_to_str(value: dict[str, Any]) -> str:
    t = value.get("text")
    if t is None:
        return ""
    if isinstance(t, list):
        return " ".join(str(x) for x in t if x is not None).strip()
    return str(t).strip()


def _rectangle_order(result: list[dict[str, Any]]) -> list[str]:
    """Region ids in order of first rectangle appearance (creation / listing order)."""
    out: list[str] = []
    seen: set[str] = set()
    for r in result:
        if r.get("type") != "rectangle":
            continue
        rid = r.get("id")
        if not rid or rid in seen:
            continue
        seen.add(rid)
        out.append(str(rid))
    return out


def _region_dict_from_results(
    result: list[dict[str, Any]],
    *,
    bbox_from_name: str = "bbox",
    text_from_name: str = "ground_truth_text",
) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for r in result:
        rid = r.get("id")
        if not rid:
            continue
        rid = str(rid)
        slot = by_id.setdefault(
            rid,
            {
                "region_id": rid,
                "bbox_pct": None,
                "transcription_gt": "",
                "choices": {},
            },
        )
        typ = r.get("type")
        fn = r.get("from_name") or ""
        val = r.get("value") or {}
        if typ == "rectangle" and fn == bbox_from_name:
            slot["bbox_pct"] = {
                k: val.get(k)
                for k in ("x", "y", "width", "height", "rotation")
                if k in val
            }
        elif typ == "textarea" and fn == text_from_name:
            slot["transcription_gt"] = _textarea_to_str(val)
        elif typ == "choices":
            slot["choices"][fn] = list(val.get("choices") or [])
    return by_id


def build_manifest_from_ls_export(
    export_path: Path | str,
    *,
    bbox_from_name: str = "bbox",
    text_from_name: str = "ground_truth_text",
    region_order: str = "rectangle_first_seen",
) -> dict[str, Any]:
    """
    Parse Label Studio export (list of tasks or ``{\"tasks\": [...]}``).

    ``region_order``:
      - ``rectangle_first_seen``: order of first ``rectangle`` per id in annotation results
      - ``geometric``: sort by center (y then x) in percent space
    """
    path = Path(export_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "tasks" in raw:
        tasks = raw["tasks"]
    elif isinstance(raw, list):
        tasks = raw
    else:
        raise ValueError("Export must be a list of tasks or {\"tasks\": [...]}")

    out_tasks: list[dict[str, Any]] = []
    for task in tasks:
        tid = task.get("id")
        data = task.get("data") or {}
        merged_result: list[dict[str, Any]] = []
        for ann in task.get("annotations") or []:
            merged_result.extend(ann.get("result") or [])

        by_id = _region_dict_from_results(
            merged_result,
            bbox_from_name=bbox_from_name,
            text_from_name=text_from_name,
        )
        if region_order == "geometric":
            ids = list(by_id.keys())

            def sort_key(rid: str) -> tuple[float, float]:
                b = by_id[rid].get("bbox_pct") or {}
                try:
                    x = float(b.get("x", 0))
                    y = float(b.get("y", 0))
                    w = float(b.get("width", 0))
                    h = float(b.get("height", 0))
                    return (y + h / 2.0, x + w / 2.0)
                except (TypeError, ValueError):
                    return (0.0, 0.0)

            ids.sort(key=sort_key)
        else:
            ids = _rectangle_order(merged_result)
            ids = [i for i in ids if i in by_id]

        regions = []
        for rid in ids:
            row = by_id[rid]
            if row.get("bbox_pct") is None:
                continue
            regions.append(
                {
                    "region_id": row["region_id"],
                    "bbox_pct": row["bbox_pct"],
                    "transcription_gt": row["transcription_gt"],
                    "choices": row["choices"],
                }
            )

        out_tasks.append(
            {
                "task_id": tid,
                "inner_id": task.get("inner_id"),
                "data": data,
                "regions": regions,
                "source_export": str(path.name),
            }
        )

    return {
        "schema": "ocr_ablation_gt_manifest_v1",
        "region_order": region_order,
        "tasks": out_tasks,
    }


def save_manifest(manifest: dict[str, Any], path: Path | str) -> None:
    Path(path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def load_manifest(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
