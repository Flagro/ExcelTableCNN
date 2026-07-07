import numpy as np
import pytest

from excel_table_cnn.data.workbook import (
    UnsupportedFormatError,
    WorkbookReader,
    load_sheet_array,
)
from .conftest import TOY_SHEET


def test_reader_xlsx(toy_workbook_path):
    with WorkbookReader(toy_workbook_path) as reader:
        assert reader.sheet_names == [TOY_SHEET]
        arr = reader.sheet_array(TOY_SHEET)
    assert isinstance(arr, np.ndarray) and arr.ndim == 3


def test_reader_xls(toy_xls_path):
    with WorkbookReader(toy_xls_path) as reader:
        assert reader.sheet_names == [TOY_SHEET]
        arr = reader.sheet_array(TOY_SHEET)
    assert isinstance(arr, np.ndarray) and arr.ndim == 3


def test_load_sheet_array_one_shot(toy_workbook_path):
    arr = load_sheet_array(toy_workbook_path, TOY_SHEET)
    assert arr.ndim == 3


def test_xlsb_raises_with_conversion_hint(tmp_path):
    path = tmp_path / "book.xlsb"
    path.write_bytes(b"\x00")
    with pytest.raises(UnsupportedFormatError, match="libreoffice"):
        WorkbookReader(str(path))


def test_unknown_extension_raises(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(UnsupportedFormatError):
        WorkbookReader(str(path))
