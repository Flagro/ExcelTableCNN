"""ExcelTableCNN: spreadsheet table detection with CNNs.

An independent open-source reimplementation inspired by the TableSense paper
(Dong et al., AAAI 2019, arXiv:2106.13500).
"""

__version__ = "0.3.0"

from .data.census import box_census, lattice_coverage
from .data.features import FEATURE_NAMES, NUM_FEATURES, featurize_sheet
from .data.pipeline import build_samples, build_sheet_sample, get_train_test
from .data.workbook import UnsupportedFormatError, WorkbookReader, load_sheet_array
from .device import resolve_device
from .evaluation.eob import eob, eob_precision_recall
from .evaluation.evaluate import evaluate_model, format_report
from .inference import detect_tables, load_model
from .model.detector import TableDetectionModel, build_model
from .training.dataset import (
    SpreadsheetDataset,
    box_to_range,
    collate_fn,
    parse_table_range,
)
from .training.train import TrainConfig, load_checkpoint, save_checkpoint, train_model

__all__ = [
    "__version__",
    "box_census",
    "lattice_coverage",
    "FEATURE_NAMES",
    "NUM_FEATURES",
    "featurize_sheet",
    "build_samples",
    "build_sheet_sample",
    "get_train_test",
    "UnsupportedFormatError",
    "WorkbookReader",
    "load_sheet_array",
    "resolve_device",
    "eob",
    "eob_precision_recall",
    "evaluate_model",
    "format_report",
    "detect_tables",
    "load_model",
    "TableDetectionModel",
    "build_model",
    "SpreadsheetDataset",
    "box_to_range",
    "collate_fn",
    "parse_table_range",
    "TrainConfig",
    "load_checkpoint",
    "save_checkpoint",
    "train_model",
]
