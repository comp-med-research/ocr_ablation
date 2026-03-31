"""Normalized 2D boxes (0–1, top-left origin) and IoU for layout alignment."""

from __future__ import annotations

from dataclasses import dataclass
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
