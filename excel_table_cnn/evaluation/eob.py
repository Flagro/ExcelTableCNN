"""Error-of-Boundary (EoB) metric.

EoB of a (prediction, ground truth) pair is the maximum absolute deviation of
the four boundaries, in cells. A detection is correct at threshold t if
EoB <= t; we report EoB-0 (exact) and EoB-2. Both boxes must use the same
convention (this package: half-open cell coordinates), which makes the
deviations identical to the closed-range ones.
"""

from typing import Dict, List, Sequence, Tuple


def eob(pred_box: Sequence[float], gt_box: Sequence[float]) -> float:
    """Max absolute boundary deviation between two boxes, in cells."""
    return max(abs(float(p) - float(g)) for p, g in zip(pred_box, gt_box))


def match_detections(
    pred_boxes: Sequence[Sequence[float]],
    pred_scores: Sequence[float],
    gt_boxes: Sequence[Sequence[float]],
    threshold: float,
) -> Tuple[int, int, int]:
    """Greedy score-ordered matching. Returns (tp, fp, fn).

    Each ground-truth table can be matched by at most one detection; extra
    detections of the same table count as false positives.
    """
    order = sorted(range(len(pred_boxes)), key=lambda i: -float(pred_scores[i]))
    matched_gt = set()
    tp = fp = 0
    for i in order:
        best_gt, best_eob = None, None
        for j, gt_box in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            candidate = eob(pred_boxes[i], gt_box)
            if best_eob is None or candidate < best_eob:
                best_gt, best_eob = j, candidate
        if best_gt is not None and best_eob <= threshold:
            matched_gt.add(best_gt)
            tp += 1
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def eob_precision_recall(
    per_sheet_predictions: List[Tuple[Sequence[Sequence[float]], Sequence[float]]],
    per_sheet_gt_boxes: List[Sequence[Sequence[float]]],
    thresholds: Sequence[float] = (0, 2),
) -> Dict[str, Dict[str, float]]:
    """Corpus-level precision/recall at each EoB threshold.

    ``per_sheet_predictions`` is a list of (boxes, scores) pairs, one entry
    per sheet, aligned with ``per_sheet_gt_boxes``.
    """
    report: Dict[str, Dict[str, float]] = {}
    for threshold in thresholds:
        tp = fp = fn = 0
        for (boxes, scores), gt_boxes in zip(per_sheet_predictions, per_sheet_gt_boxes):
            sheet_tp, sheet_fp, sheet_fn = match_detections(boxes, scores, gt_boxes, threshold)
            tp, fp, fn = tp + sheet_tp, fp + sheet_fp, fn + sheet_fn
        report[f"eob{int(threshold)}"] = {
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return report
