"""Normalized 2D geometry helpers (0-1, top-left origin) for layout alignment."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class BoxNorm:
    """Axis-aligned rectangle in normalized image coordinates (top-left origin)."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if self.x1 < self.x0 or self.y1 < self.y0:
            object.__setattr__(self, "x0", min(self.x0, self.x1))
            object.__setattr__(self, "x1", max(self.x0, self.x1))
            object.__setattr__(self, "y0", min(self.y0, self.y1))
            object.__setattr__(self, "y1", max(self.y0, self.y1))

    @property
    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    def center(self) -> tuple[float, float]:
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2


def intersection_area(a: BoxNorm, b: BoxNorm) -> float:
    ix0 = max(a.x0, b.x0)
    iy0 = max(a.y0, b.y0)
    ix1 = min(a.x1, b.x1)
    iy1 = min(a.y1, b.y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def iou(a: BoxNorm, b: BoxNorm) -> float:
    inter = intersection_area(a, b)
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def box_from_bbox_pct(bp: dict[str, Any]) -> BoxNorm | None:
    """Label Studio–style percentage rectangle: x, y, width, height (0–100), top-left origin."""
    try:
        x = float(bp["x"])
        y = float(bp["y"])
        w = float(bp["width"])
        h = float(bp["height"])
    except (KeyError, TypeError, ValueError):
        return None
    return BoxNorm(x / 100.0, y / 100.0, (x + w) / 100.0, (y + h) / 100.0)


def docling_bbox_to_box_norm(
    bbox: dict[str, Any],
    page_w: float,
    page_h: float,
) -> BoxNorm | None:
    """
    Docling ``prov.bbox`` with ``coord_origin: BOTTOMLEFT``.

    If any of l,t,r,b exceeds 1.5, treat as **pixel** coords (divide by page size).
    Otherwise treat as **normalized** 0–1 coords on the page.
    """
    try:
        l = float(bbox["l"])
        t = float(bbox["t"])
        r = float(bbox["r"])
        b = float(bbox["b"])
    except (KeyError, TypeError, ValueError):
        return None
    if page_w <= 0 or page_h <= 0:
        return None
    mx = max(abs(l), abs(r), abs(t), abs(b))
    if mx > 1.5:
        top_y = page_h - t
        bot_y = page_h - b
        y0 = min(top_y, bot_y) / page_h
        y1 = max(top_y, bot_y) / page_h
        x0 = min(l, r) / page_w
        x1 = max(l, r) / page_w
    else:
        x0 = min(l, r)
        x1 = max(l, r)
        y0_tl = 1.0 - max(t, b)
        y1_tl = 1.0 - min(t, b)
        y0 = min(y0_tl, y1_tl)
        y1 = max(y0_tl, y1_tl)
    return BoxNorm(x0, y0, x1, y1)


def reading_order_sort_key(box: BoxNorm) -> tuple[float, float]:
    """
    Sort key: top-to-bottom, then **left-to-right** on the same line.

    Uses the vertical **center** rounded to one decimal (normalized coords) as a row bucket so
    small y jitter between spans on one line does not dominate over ``x0``.
    Pure ``(y0, x0)`` ordering can place a left fragment after a right fragment when ``y0`` differs slightly.
    """
    y_mid = (box.y0 + box.y1) / 2.0
    row = round(y_mid, 1)
    return (row, box.x0)


def rotated_polygon_from_bbox_pct(bp: dict[str, Any]) -> list[tuple[float, float]] | None:
    """
    Convert Label Studio bbox percentages to a normalized polygon.

    Uses ``x,y,width,height,rotation`` where rotation is in degrees. Rotation is applied
    around the rectangle center (Label Studio rectangle behavior).
    """
    try:
        x = float(bp["x"]) / 100.0
        y = float(bp["y"]) / 100.0
        w = float(bp["width"]) / 100.0
        h = float(bp["height"]) / 100.0
    except (KeyError, TypeError, ValueError):
        return None
    rot = float(bp.get("rotation", 0.0) or 0.0)

    pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    if abs(rot) < 1e-9:
        return pts

    cx = x + (w / 2.0)
    cy = y + (h / 2.0)
    th = math.radians(rot)
    c = math.cos(th)
    s = math.sin(th)
    out: list[tuple[float, float]] = []
    for px, py in pts:
        dx = px - cx
        dy = py - cy
        rx = cx + (dx * c) - (dy * s)
        ry = cy + (dx * s) + (dy * c)
        out.append((rx, ry))
    return out


def polygon_area(poly: list[tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    a = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        a += (x1 * y2) - (x2 * y1)
    return abs(a) * 0.5


def point_in_polygon(px: float, py: float, poly: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon; boundary counts as inside."""
    if len(poly) < 3:
        return False
    inside = False
    eps = 1e-12
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
        if abs(cross) <= eps and min(x1, x2) - eps <= px <= max(x1, x2) + eps and min(y1, y2) - eps <= py <= max(y1, y2) + eps:
            return True
        intersects = ((y1 > py) != (y2 > py)) and (
            px < (x2 - x1) * (py - y1) / ((y2 - y1) + eps) + x1
        )
        if intersects:
            inside = not inside
    return inside


def clip_polygon(subject: list[tuple[float, float]], clipper: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    Sutherland-Hodgman polygon clipping for convex clipper.

    Returns the intersection polygon between ``subject`` and ``clipper``.
    """
    if len(subject) < 3 or len(clipper) < 3:
        return []

    def _signed_area(poly: list[tuple[float, float]]) -> float:
        s = 0.0
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            s += (x1 * y2) - (x2 * y1)
        return 0.5 * s

    ccw = _signed_area(clipper) >= 0.0

    def inside(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> bool:
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        return cross >= 0 if ccw else cross <= 0

    def intersection(
        s: tuple[float, float],
        e: tuple[float, float],
        a: tuple[float, float],
        b: tuple[float, float],
    ) -> tuple[float, float]:
        x1, y1 = s
        x2, y2 = e
        x3, y3 = a
        x4, y4 = b
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-12:
            return e
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    output = subject[:]
    for i in range(len(clipper)):
        a = clipper[i]
        b = clipper[(i + 1) % len(clipper)]
        inp = output
        output = []
        if not inp:
            break
        s = inp[-1]
        for e in inp:
            if inside(e, a, b):
                if not inside(s, a, b):
                    output.append(intersection(s, e, a, b))
                output.append(e)
            elif inside(s, a, b):
                output.append(intersection(s, e, a, b))
            s = e
    return output


def iou_polygon_vs_box(poly: list[tuple[float, float]], box: BoxNorm) -> float:
    """IoU where GT may be rotated polygon and prediction is axis-aligned box."""
    if len(poly) < 3:
        return 0.0
    bpoly = [(box.x0, box.y0), (box.x1, box.y0), (box.x1, box.y1), (box.x0, box.y1)]
    inter_poly = clip_polygon(poly, bpoly)
    inter = polygon_area(inter_poly)
    if inter <= 0:
        return 0.0
    a_poly = polygon_area(poly)
    a_box = box.area
    union = a_poly + a_box - inter
    return inter / union if union > 0 else 0.0
