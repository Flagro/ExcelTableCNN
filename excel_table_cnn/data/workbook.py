"""Format-dispatching workbook reader: .xlsx/.xlsm via openpyxl, .xls via xlrd.

This is what makes LibreOffice optional: legacy .xls files (the bulk of the
VEnron2 corpus) are featurized natively. Only .xlsb has no native reader —
``UnsupportedFormatError`` points at the LibreOffice conversion path.
"""

import logging
import os

import numpy as np

from .features import DEFAULT_MAX_COLS, DEFAULT_MAX_ROWS, featurize_sheet
from .features_xls import featurize_xls_sheet

logger = logging.getLogger(__name__)

OPENPYXL_EXTENSIONS = (".xlsx", ".xlsm")
XLRD_EXTENSIONS = (".xls",)
NATIVE_EXTENSIONS = OPENPYXL_EXTENSIONS + XLRD_EXTENSIONS


class UnsupportedFormatError(ValueError):
    pass


class WorkbookReader:
    """Uniform lazy reader over supported workbook formats.

    Usage::

        with WorkbookReader(path) as reader:
            for name in reader.sheet_names:
                array = reader.sheet_array(name)
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        ext = os.path.splitext(file_path)[1].lower()
        if ext in OPENPYXL_EXTENSIONS:
            self._backend = "openpyxl"
        elif ext in XLRD_EXTENSIONS:
            self._backend = "xlrd"
        else:
            raise UnsupportedFormatError(
                f"No native reader for {ext!r} files ({file_path!r}). "
                "Convert to .xlsx first — e.g. with LibreOffice: "
                "libreoffice --headless --convert-to xlsx <file> "
                "(see excel_table_cnn.data.converter)."
            )
        self._book = None

    def __enter__(self):
        if self._backend == "openpyxl":
            import openpyxl

            self._book = openpyxl.load_workbook(self.file_path)
        else:
            import xlrd

            self._book = xlrd.open_workbook(self.file_path, formatting_info=True)
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False

    def close(self):
        if self._book is not None:
            if self._backend == "openpyxl":
                self._book.close()
            else:
                self._book.release_resources()
            self._book = None

    @property
    def sheet_names(self):
        if self._backend == "openpyxl":
            return list(self._book.sheetnames)
        return list(self._book.sheet_names())

    def sheet_array(
        self,
        sheet_name: str,
        max_rows: int = DEFAULT_MAX_ROWS,
        max_cols: int = DEFAULT_MAX_COLS,
    ) -> np.ndarray:
        """Featurize one sheet into an (H, W, NUM_FEATURES) float32 array."""
        if self._backend == "openpyxl":
            return featurize_sheet(
                self._book[sheet_name], max_rows=max_rows, max_cols=max_cols
            )
        return featurize_xls_sheet(
            self._book,
            self._book.sheet_by_name(sheet_name),
            max_rows=max_rows,
            max_cols=max_cols,
        )


def load_sheet_array(
    file_path: str,
    sheet_name: str,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_cols: int = DEFAULT_MAX_COLS,
) -> np.ndarray:
    """One-shot convenience for a single sheet."""
    with WorkbookReader(file_path) as reader:
        return reader.sheet_array(sheet_name, max_rows=max_rows, max_cols=max_cols)
