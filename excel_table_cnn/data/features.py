"""Cell featurization: turn an openpyxl worksheet into an (H, W, C) float array.

Each cell is described by ``NUM_FEATURES`` binary features (``FEATURE_NAMES``
gives the channel order). The array always starts at A1 so that annotation
coordinates remain valid without offset bookkeeping; trailing all-default
rows/columns are trimmed (plus a small margin), and the used range is capped
to protect against sheets with huge stray formatting.
"""

import logging
from typing import List

import numpy as np
from openpyxl.worksheet.worksheet import Worksheet

logger = logging.getLogger(__name__)

# Bump when FEATURE_NAMES or their semantics change: invalidates cached tensors.
FEATURES_VERSION = "v1"

FEATURE_NAMES: List[str] = [
    "is_empty",
    "is_string",
    "is_merged",
    "is_bold",
    "is_italic",
    "left_border",
    "right_border",
    "top_border",
    "bottom_border",
    "is_filled",
    "horizontal_alignment",
    "left_horizontal_alignment",
    "right_horizontal_alignment",
    "center_horizontal_alignment",
    "wrapped_text",
    "indent",
    "formula",
]

NUM_FEATURES = len(FEATURE_NAMES)

_IS_EMPTY_IDX = FEATURE_NAMES.index("is_empty")

# Hard caps on the featurized used range. Real tables larger than this exist
# but are rare in the training corpus; stray formatting thousands of rows below
# the data is common and would otherwise explode memory.
DEFAULT_MAX_ROWS = 2048
DEFAULT_MAX_COLS = 512

# Rows/columns kept beyond the last non-default cell, so tables whose
# annotation extends slightly into empty cells aren't clipped.
TRIM_MARGIN = 2


def default_cell_vector() -> np.ndarray:
    """Feature vector of an untouched cell: empty, everything else off."""
    vec = np.zeros(NUM_FEATURES, dtype=np.float32)
    vec[_IS_EMPTY_IDX] = 1.0
    return vec


def cell_feature_vector(cell, is_merged: bool) -> List[float]:
    """Features of a single openpyxl cell, ordered as FEATURE_NAMES."""
    alignment = cell.alignment
    border = cell.border
    return [
        float(cell.value is None),
        float(cell.data_type in ("s", "str")),
        float(is_merged),
        float(bool(cell.font.b)),
        float(bool(cell.font.i)),
        # ``border.left`` is always a Side object in openpyxl; only its
        # ``style`` says whether a border is actually drawn.
        float(border.left.style is not None),
        float(border.right.style is not None),
        float(border.top.style is not None),
        float(border.bottom.style is not None),
        float(cell.fill.patternType is not None),
        float(alignment.horizontal is not None),
        float(alignment.horizontal == "left"),
        float(alignment.horizontal == "right"),
        float(alignment.horizontal == "center"),
        float(bool(alignment.wrapText)),
        float(bool(alignment.indent)),
        float(cell.data_type == "f"),
    ]


def featurize_sheet(
    ws: Worksheet,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_cols: int = DEFAULT_MAX_COLS,
) -> np.ndarray:
    """Featurize a worksheet into an (H, W, NUM_FEATURES) float32 array."""
    used_rows = ws.max_row or 1
    used_cols = ws.max_column or 1
    if used_rows > max_rows or used_cols > max_cols:
        logger.warning(
            "Sheet %r used range %dx%d exceeds cap %dx%d; clipping",
            ws.title, used_rows, used_cols, max_rows, max_cols,
        )
    n_rows = min(used_rows, max_rows)
    n_cols = min(used_cols, max_cols)

    arr = np.tile(default_cell_vector(), (n_rows, n_cols, 1))

    merged_coords = set()
    for merged_range in ws.merged_cells.ranges:
        for r in range(merged_range.min_row, merged_range.max_row + 1):
            for c in range(merged_range.min_col, merged_range.max_col + 1):
                merged_coords.add((r, c))

    for row in ws.iter_rows(min_row=1, max_row=n_rows, min_col=1, max_col=n_cols):
        for cell in row:
            arr[cell.row - 1, cell.column - 1] = cell_feature_vector(
                cell, (cell.row, cell.column) in merged_coords
            )

    return trim_trailing_default(arr)


def trim_trailing_default(arr: np.ndarray) -> np.ndarray:
    """Trim trailing rows/columns whose cells all equal the default vector."""
    non_default = (arr != default_cell_vector()).any(axis=2)
    if not non_default.any():
        return arr[:1, :1]
    last_row = int(np.nonzero(non_default.any(axis=1))[0][-1]) + 1
    last_col = int(np.nonzero(non_default.any(axis=0))[0][-1]) + 1
    return arr[: min(last_row + TRIM_MARGIN, arr.shape[0]),
               : min(last_col + TRIM_MARGIN, arr.shape[1])]


def load_sheet_features(
    file_path: str,
    sheet_name: str,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_cols: int = DEFAULT_MAX_COLS,
) -> np.ndarray:
    """Convenience: open a workbook and featurize one sheet."""
    import openpyxl

    wb = openpyxl.load_workbook(file_path)
    try:
        return featurize_sheet(wb[sheet_name], max_rows=max_rows, max_cols=max_cols)
    finally:
        wb.close()
