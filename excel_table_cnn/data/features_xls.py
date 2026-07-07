"""Cell featurization for legacy .xls workbooks via xlrd — no LibreOffice.

xlrd (with ``formatting_info=True``) exposes fonts, borders, fills, alignment,
number formats and merged ranges for the BIFF .xls format, so the channels
match the openpyxl featurizer. Known deviations, both documented and covered
by the parity test's exclusions:

- the ``formula`` channel is always 0 — xlrd returns cached formula results
  without flagging them (value-string statistics therefore describe the
  cached *result* for formula cells, not the formula text);
- date/time cells surface as floats, so their value-string statistics differ
  from openpyxl's datetime rendering (format channels still agree).
"""

import logging

import numpy as np

from .features import (
    DEFAULT_MAX_COLS,
    DEFAULT_MAX_ROWS,
    FEATURE_NAMES,
    default_cell_vector,
    format_features,
    text_stats,
    trim_trailing_default,
    value_text,
)

logger = logging.getLogger(__name__)

# xlrd cell types
_XL_CELL_EMPTY = 0
_XL_CELL_TEXT = 1
_XL_CELL_BLANK = 6

# xlrd XFAlignment.hor_align values
_HALIGN_GENERAL = 0
_HALIGN_LEFT = 1
_HALIGN_CENTER = 2
_HALIGN_RIGHT = 3

# BIFF color indexes treated as "default": automatic (32767), black (0, 8),
# white (1, 9), and the auto foreground/background pattern indexes (64, 65).
_DEFAULT_FONT_COLOR_INDEXES = frozenset({32767, 0, 8})
_DEFAULT_FILL_COLOR_INDEXES = frozenset({64, 65, 0, 1, 8, 9})


def _xls_cell_vector(cell_type, value, xf, font, number_format, merged_extent):
    horizontal = xf.alignment.hor_align
    text = value_text(value)
    filled = xf.background.fill_pattern != 0
    return [
        float(cell_type in (_XL_CELL_EMPTY, _XL_CELL_BLANK)),
        float(cell_type == _XL_CELL_TEXT),
        float(merged_extent is not None),
        float(bool(font.bold)),
        float(bool(font.italic)),
        float(xf.border.left_line_style != 0),
        float(xf.border.right_line_style != 0),
        float(xf.border.top_line_style != 0),
        float(xf.border.bottom_line_style != 0),
        float(filled),
        float(horizontal != _HALIGN_GENERAL),
        float(horizontal == _HALIGN_LEFT),
        float(horizontal == _HALIGN_RIGHT),
        float(horizontal == _HALIGN_CENTER),
        float(bool(xf.alignment.text_wrapped)),
        float(bool(xf.alignment.indent_level)),
        0.0,  # formula: not detectable via xlrd (cached results only)
        *text_stats("" if cell_type in (_XL_CELL_EMPTY, _XL_CELL_BLANK) else text),
        *format_features(number_format),
        float(bool(merged_extent and merged_extent[0])),
        float(bool(merged_extent and merged_extent[1])),
        float(
            filled
            and xf.background.pattern_colour_index not in _DEFAULT_FILL_COLOR_INDEXES
        ),
        float(font.colour_index not in _DEFAULT_FONT_COLOR_INDEXES),
    ]


def featurize_xls_sheet(
    book,
    sheet,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_cols: int = DEFAULT_MAX_COLS,
) -> np.ndarray:
    """Featurize an xlrd sheet into an (H, W, NUM_FEATURES) float32 array.

    ``book`` must have been opened with ``formatting_info=True``.
    """
    used_rows = max(sheet.nrows, 1)
    used_cols = max(sheet.ncols, 1)
    if used_rows > max_rows or used_cols > max_cols:
        logger.warning(
            "Sheet %r used range %dx%d exceeds cap %dx%d; clipping",
            sheet.name, used_rows, used_cols, max_rows, max_cols,
        )
    n_rows = min(used_rows, max_rows)
    n_cols = min(used_cols, max_cols)

    arr = np.tile(default_cell_vector(), (n_rows, n_cols, 1))

    merged_extents = {}
    for row_lo, row_hi, col_lo, col_hi in sheet.merged_cells:  # half-open
        extent = (col_hi - col_lo > 1, row_hi - row_lo > 1)
        for r in range(row_lo, row_hi):
            for c in range(col_lo, col_hi):
                merged_extents[(r, c)] = extent

    # Iterate only rows that physically exist (sheet.nrows can be 0 for an
    # empty sheet while the array keeps its 1x1 default floor).
    for r in range(min(n_rows, sheet.nrows)):
        row_len = sheet.row_len(r)
        for c in range(min(n_cols, row_len)):
            cell_type = sheet.cell_type(r, c)
            xf = book.xf_list[sheet.cell_xf_index(r, c)]
            font = book.font_list[xf.font_index]
            format_record = book.format_map.get(xf.format_key)
            number_format = format_record.format_str if format_record else None
            arr[r, c] = _xls_cell_vector(
                cell_type, sheet.cell_value(r, c), xf, font, number_format,
                merged_extents.get((r, c)),
            )

    assert arr.shape[2] == len(FEATURE_NAMES)
    return trim_trailing_default(arr)
