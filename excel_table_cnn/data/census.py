"""Ground-truth box census: measure table shapes and anchor-lattice coverage.

Used to derive the anchor defaults in ``model/rcnn.py`` from the actual
annotation corpus instead of guessing. Rerun on a new corpus with::

    from excel_table_cnn.data.census import box_census, lattice_coverage
    from excel_table_cnn.data.markup import MarkupLoader
    stats = box_census(MarkupLoader().get_markup("tablesense"))
"""

import math
from typing import Dict, List, Sequence, Tuple

from ..training.dataset import parse_table_range


def collect_box_dims(markup_df) -> List[Tuple[float, float]]:
    """(width, height) in cells for every parseable annotated table range."""
    dims: List[Tuple[float, float]] = []
    for ranges in markup_df["table_range"]:
        for rng in ranges:
            rng = rng.strip()
            if not rng:
                continue
            try:
                x1, y1, x2, y2 = parse_table_range(rng)
            except ValueError:
                continue
            dims.append((x2 - x1, y2 - y1))
    return dims


def _percentile(sorted_values: Sequence[float], p: float) -> float:
    return sorted_values[round(p / 100 * (len(sorted_values) - 1))]


def box_census(markup_df) -> Dict[str, Dict[str, float]]:
    """Percentile summary of widths, heights, sqrt-areas and h/w ratios."""
    dims = collect_box_dims(markup_df)
    series = {
        "width": sorted(w for w, _ in dims),
        "height": sorted(h for _, h in dims),
        "sqrt_area": sorted(math.sqrt(w * h) for w, h in dims),
        "hw_ratio": sorted(h / w for w, h in dims),
    }
    return {
        name: {f"p{p}": _percentile(values, p) for p in (1, 5, 25, 50, 75, 95, 99)}
        for name, values in series.items()
    }


def _anchor_dims(size: float, ratio: float) -> Tuple[float, float]:
    """torchvision convention: h = size*sqrt(ratio), w = size/sqrt(ratio)."""
    return size / math.sqrt(ratio), size * math.sqrt(ratio)


def lattice_coverage(
    dims: Sequence[Tuple[float, float]],
    sizes: Sequence[float],
    ratios: Sequence[float],
    iou_thresholds: Sequence[float] = (0.5, 0.7),
) -> Dict[float, float]:
    """Fraction of GT boxes whose best (centered) anchor reaches each IoU."""

    def best_iou(width: float, height: float) -> float:
        best = 0.0
        for size in sizes:
            for ratio in ratios:
                aw, ah = _anchor_dims(size, ratio)
                inter = min(width, aw) * min(height, ah)
                union = width * height + aw * ah - inter
                best = max(best, inter / union)
        return best

    coverage = [best_iou(w, h) for w, h in dims]
    n = max(len(coverage), 1)
    return {t: sum(c >= t for c in coverage) / n for t in iou_thresholds}
