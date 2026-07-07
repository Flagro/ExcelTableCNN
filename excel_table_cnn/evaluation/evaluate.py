"""Evaluation harness: run the detector over a dataset and report EoB metrics."""

import argparse
import logging
from typing import Dict, List, Optional, Sequence

import torch

from .eob import eob, eob_precision_recall
from ..device import DeviceLike, resolve_device
from ..training.dataset import SpreadsheetDataset, box_to_range

DEFAULT_THRESHOLDS = (0, 2)


@torch.no_grad()
def evaluate_model(
    model,
    dataset: SpreadsheetDataset,
    device: DeviceLike = "auto",
    score_threshold: float = 0.5,
    eob_thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> Dict:
    """Returns EoB precision/recall plus per-sheet details for error analysis."""
    device = resolve_device(device)
    model.to(device)
    model.eval()

    per_sheet_predictions = []
    per_sheet_gt_boxes = []
    per_sheet_details = []

    for sample in dataset.samples:
        output = model([sample["tensor"].to(device)])[0]
        keep = output["scores"] >= score_threshold
        boxes = output["boxes"][keep].cpu().tolist()
        scores = output["scores"][keep].cpu().tolist()
        gt_boxes = sample["boxes"].tolist()

        per_sheet_predictions.append((boxes, scores))
        per_sheet_gt_boxes.append(gt_boxes)
        per_sheet_details.append(
            {
                "file_path": sample.get("file_path"),
                "sheet_name": sample.get("sheet_name"),
                "predictions": [
                    {"range": box_to_range(box), "score": score}
                    for box, score in zip(boxes, scores)
                ],
                "ground_truth": [box_to_range(box) for box in gt_boxes],
                # Best (lowest) EoB achieved for each ground-truth table:
                # the number to look at when hunting systematic boundary errors.
                "best_eob_per_gt": [
                    min((eob(box, gt) for box in boxes), default=None)
                    for gt in gt_boxes
                ],
            }
        )

    report = eob_precision_recall(
        per_sheet_predictions, per_sheet_gt_boxes, thresholds=eob_thresholds
    )
    report["n_sheets"] = len(dataset)
    report["n_gt"] = sum(len(boxes) for boxes in per_sheet_gt_boxes)
    report["per_sheet"] = per_sheet_details
    return report


def format_report(report: Dict) -> str:
    lines = [f"Evaluated {report['n_sheets']} sheets, {report['n_gt']} tables:"]
    for key, metrics in report.items():
        if not key.startswith("eob"):
            continue
        lines.append(
            f"  EoB-{key.removeprefix('eob')}: "
            f"precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} "
            f"(tp={metrics['tp']} fp={metrics['fp']} fn={metrics['fn']})"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint on the annotated test split."
    )
    parser.add_argument("--weights", required=True,
                        help="checkpoint produced by excel-table-cnn-train")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--dataset", default="VEnron2")
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-cols", type=int, default=None)
    parser.add_argument("--worst", type=int, default=0,
                        help="also print the N worst sheets for error analysis")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from ..data.features import DEFAULT_MAX_COLS, DEFAULT_MAX_ROWS
    from ..data.pipeline import get_train_test
    from ..training.train import load_checkpoint

    _, test_samples = get_train_test(
        data_folder_path=args.data_dir,
        dataset_name=args.dataset,
        train_size=0,
        testing_size=args.test_size,
        cache_dir=args.cache_dir,
        max_rows=args.max_rows or DEFAULT_MAX_ROWS,
        max_cols=args.max_cols or DEFAULT_MAX_COLS,
    )
    dataset = SpreadsheetDataset(test_samples)
    # Rebuild with the internal RoI score gate at (or below) the requested
    # threshold — otherwise the model default (0.5) silently filters
    # detections before --score-threshold ever sees them.
    model = load_checkpoint(
        args.weights, box_score_thresh=min(args.score_threshold, 0.05)
    )
    report = evaluate_model(
        model, dataset, device=args.device, score_threshold=args.score_threshold
    )
    print(format_report(report))

    if args.worst:
        worst = sorted(
            report["per_sheet"],
            key=lambda s: max((e for e in s["best_eob_per_gt"] if e is not None),
                              default=float("inf")),
            reverse=True,
        )
        for sheet in worst[: args.worst]:
            print(f"\n{sheet['file_path']} [{sheet['sheet_name']}]")
            print(f"  ground truth: {sheet['ground_truth']}")
            print(f"  predictions : "
                  f"{[(p['range'], round(p['score'], 2)) for p in sheet['predictions']]}")
            print(f"  best EoB    : {sheet['best_eob_per_gt']}")


if __name__ == "__main__":
    main()
