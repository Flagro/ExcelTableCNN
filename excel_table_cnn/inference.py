"""User-facing inference API: spreadsheet file in, table ranges out.

Reads .xlsx/.xlsm (openpyxl) and legacy .xls (xlrd) natively — no LibreOffice
needed; .xlsb must be converted first (see ``excel_table_cnn.data.converter``).
"""

import argparse
import logging
from typing import Dict, List, Optional

import torch

from .data.features import DEFAULT_MAX_COLS, DEFAULT_MAX_ROWS, NUM_FEATURES
from .data.workbook import WorkbookReader
from .device import DeviceLike, resolve_device
from .model.detector import TableDetectionModel, build_model
from .training.dataset import box_to_range
from .training.train import load_checkpoint

logger = logging.getLogger(__name__)


def load_model(weights_path: str, device: DeviceLike = "auto", **rcnn_kwargs) -> TableDetectionModel:
    """Load a model from a checkpoint produced by the trainer."""
    return load_checkpoint(weights_path, device=resolve_device(device), **rcnn_kwargs)


@torch.no_grad()
def detect_tables(
    file_path: str,
    sheet_name: Optional[str] = None,
    model: Optional[TableDetectionModel] = None,
    weights: Optional[str] = None,
    device: DeviceLike = "auto",
    score_threshold: float = 0.5,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_cols: int = DEFAULT_MAX_COLS,
) -> List[Dict]:
    """Detect tables in a spreadsheet file (.xlsx, .xlsm or .xls).

    Returns one dict per detection: ``{"sheet", "range", "score"}``, e.g.
    ``{"sheet": "Q3", "range": "B2:H45", "score": 0.97}``.
    ``sheet_name=None`` scans all sheets.
    """
    device = resolve_device(device)
    if model is None:
        if weights is not None:
            # Keep the model's internal score gate at or below the requested
            # threshold so score_threshold alone governs what's returned.
            model = load_model(
                weights, device=device,
                box_score_thresh=min(score_threshold, 0.05),
            )
        else:
            logger.warning(
                "No model or weights given — using randomly initialized weights; "
                "detections will be meaningless."
            )
            model = build_model(in_channels=NUM_FEATURES)
    model.to(device)
    model.eval()

    detections: List[Dict] = []
    with WorkbookReader(file_path) as reader:
        sheet_names = [sheet_name] if sheet_name is not None else reader.sheet_names
        for name in sheet_names:
            array = reader.sheet_array(name, max_rows=max_rows, max_cols=max_cols)
            tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous().to(device)
            output = model([tensor])[0]
            for box, score in zip(output["boxes"], output["scores"]):
                if float(score) < score_threshold:
                    continue
                detections.append(
                    {
                        "sheet": name,
                        "range": box_to_range(box.cpu().tolist()),
                        "score": float(score),
                    }
                )
    return detections


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Detect tables in a spreadsheet file (.xlsx, .xlsm or .xls)."
    )
    parser.add_argument("file", help="path to the spreadsheet")
    parser.add_argument("--weights", required=True,
                        help="checkpoint produced by excel-table-cnn-train")
    parser.add_argument("--sheet", default=None, help="sheet name (default: all sheets)")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    detections = detect_tables(
        args.file, sheet_name=args.sheet, weights=args.weights,
        device=args.device, score_threshold=args.score_threshold,
    )
    if not detections:
        print("No tables detected above the score threshold.")
        return
    for det in detections:
        print(f"{det['sheet']}!{det['range']}\tscore={det['score']:.3f}")


if __name__ == "__main__":
    main()
