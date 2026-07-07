"""The xlrd featurizer must produce the same channels as the openpyxl one —
this is what makes LibreOffice optional for the .xls training corpus."""

import numpy as np
import pytest
from openpyxl.utils.cell import coordinate_to_tuple

from excel_table_cnn.data.features import NUM_FEATURES, default_cell_vector
from excel_table_cnn.data.workbook import load_sheet_array
from .conftest import TOY_SHEET, feature_index


def cell(arr, ref):
    row, col = coordinate_to_tuple(ref)
    return arr[row - 1, col - 1]


@pytest.fixture()
def xls_features(toy_xls_path):
    return load_sheet_array(toy_xls_path, TOY_SHEET)


def test_shape_and_dtype(xls_features):
    assert xls_features.ndim == 3
    assert xls_features.shape[2] == NUM_FEATURES
    assert xls_features.dtype == np.float32


def test_header_cell_features(xls_features):
    b2 = cell(xls_features, "B2")
    assert b2[feature_index("is_empty")] == 0
    assert b2[feature_index("is_string")] == 1
    assert b2[feature_index("is_bold")] == 1
    assert b2[feature_index("is_filled")] == 1
    for side in ("left_border", "right_border", "top_border", "bottom_border"):
        assert b2[feature_index(side)] == 1
    assert b2[feature_index("horizontal_alignment")] == 1
    assert b2[feature_index("center_horizontal_alignment")] == 1


def test_untouched_cell_is_default(xls_features):
    assert (cell(xls_features, "E4") == default_cell_vector()).all()


def test_numeric_cell_is_not_string(xls_features):
    c3 = cell(xls_features, "C3")
    assert c3[feature_index("is_empty")] == 0
    assert c3[feature_index("is_string")] == 0


def test_merged_cells(xls_features):
    assert cell(xls_features, "F2")[feature_index("is_merged")] == 1  # anchor
    assert cell(xls_features, "G3")[feature_index("is_merged")] == 1  # covered
    assert cell(xls_features, "B2")[feature_index("is_merged")] == 0


def test_assorted_features(xls_features):
    assert cell(xls_features, "F6")[feature_index("is_italic")] == 1
    assert cell(xls_features, "F8")[feature_index("wrapped_text")] == 1
    assert cell(xls_features, "G8")[feature_index("indent")] == 1
    assert cell(xls_features, "F10")[feature_index("left_horizontal_alignment")] == 1
    assert cell(xls_features, "G10")[feature_index("right_horizontal_alignment")] == 1


def test_formula_channel_is_always_zero(xls_features):
    """Documented limitation: xlrd exposes cached formula results only."""
    assert xls_features[:, :, feature_index("formula")].sum() == 0


def test_empty_xls_sheet_yields_default_array(tmp_path):
    """Regression: real corpus files contain fully empty sheets (nrows=0) —
    they must featurize to a default array, not crash on row_len()."""
    import xlwt

    wb = xlwt.Workbook()
    ws = wb.add_sheet("HasData")
    ws.write(0, 0, "x")
    wb.add_sheet("Empty")
    path = tmp_path / "empty_sheet.xls"
    wb.save(str(path))

    arr = load_sheet_array(str(path), "Empty")
    assert arr.shape == (1, 1, NUM_FEATURES)
    assert (arr[0, 0] == default_cell_vector()).all()


def test_parity_with_openpyxl_featurizer(toy_workbook_path, toy_xls_path):
    """The two backends must agree channel-for-channel on equivalent content
    (except ``formula``, absent from the .xls fixture by design)."""
    from excel_table_cnn.data.features import FEATURE_NAMES

    xlsx = load_sheet_array(toy_workbook_path, TOY_SHEET)
    xls = load_sheet_array(toy_xls_path, TOY_SHEET)

    probes = ["B2", "C2", "D2", "B3", "C3", "D6", "E4",
              "F2", "G2", "G3", "F6", "F8", "G8", "F10", "G10"]
    channels = [name for name in FEATURE_NAMES if name != "formula"]
    for ref in probes:
        for name in channels:
            idx = feature_index(name)
            assert cell(xlsx, ref)[idx] == cell(xls, ref)[idx], (
                f"{ref} channel {name}: openpyxl={cell(xlsx, ref)[idx]} "
                f"xlrd={cell(xls, ref)[idx]}"
            )
