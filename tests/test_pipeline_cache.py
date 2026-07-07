import pandas as pd
import pytest
import torch

import excel_table_cnn.data.pipeline as pipeline
from excel_table_cnn.data.features import FEATURE_NAMES
from .conftest import TOY_SHEET, TOY_TABLE_RANGE


def test_build_sheet_sample_contents(toy_workbook_path, tmp_path):
    sample = pipeline.build_sheet_sample(
        toy_workbook_path, TOY_SHEET, [TOY_TABLE_RANGE], cache_dir=str(tmp_path / "cache")
    )
    assert sample["tensor"].shape[0] == len(FEATURE_NAMES)
    assert sample["boxes"].tolist() == [[1.0, 1.0, 4.0, 6.0]]  # B2:D6 half-open
    assert sample["sheet_name"] == TOY_SHEET
    assert sample["feature_names"] == list(FEATURE_NAMES)


def test_second_call_hits_cache(toy_workbook_path, tmp_path, monkeypatch):
    cache_dir = str(tmp_path / "cache")
    first = pipeline.build_sheet_sample(
        toy_workbook_path, TOY_SHEET, [TOY_TABLE_RANGE], cache_dir=cache_dir
    )

    class Boom:
        def __init__(self, *args, **kwargs):
            raise AssertionError("workbook opened despite warm cache")

    monkeypatch.setattr(pipeline, "WorkbookReader", Boom)
    second = pipeline.build_sheet_sample(
        toy_workbook_path, TOY_SHEET, [TOY_TABLE_RANGE], cache_dir=cache_dir
    )
    assert torch.equal(first["tensor"], second["tensor"])
    assert torch.equal(first["boxes"], second["boxes"])


def test_build_sheet_sample_from_native_xls(toy_xls_path, tmp_path):
    """LibreOffice-free path: legacy .xls straight into a training sample."""
    sample = pipeline.build_sheet_sample(
        toy_xls_path, TOY_SHEET, [TOY_TABLE_RANGE], cache_dir=str(tmp_path / "cache")
    )
    assert sample["tensor"].shape[0] == len(FEATURE_NAMES)
    assert sample["boxes"].tolist() == [[1.0, 1.0, 4.0, 6.0]]


def test_cache_key_changes_with_file_content(toy_workbook_path, tmp_path):
    key_before = pipeline._cache_key(toy_workbook_path, TOY_SHEET)
    with open(toy_workbook_path, "ab") as f:
        f.write(b"tamper")
    key_after = pipeline._cache_key(toy_workbook_path, TOY_SHEET)
    assert key_before != key_after


def test_build_samples_from_files_df(toy_workbook_path, tmp_path):
    import os

    files_df = pd.DataFrame(
        [
            {
                "parent_path": os.path.dirname(toy_workbook_path),
                "file_name": os.path.basename(toy_workbook_path),
                "sheet_name": TOY_SHEET,
                "table_range": [TOY_TABLE_RANGE],
            }
        ]
    )
    samples = pipeline.build_samples(files_df, data_folder_path="/")
    assert len(samples) == 1
    assert samples[0]["boxes"].shape == (1, 4)


def test_build_samples_skips_broken_files(tmp_path, caplog):
    broken = tmp_path / "broken.xlsx"
    broken.write_bytes(b"this is not a workbook")
    files_df = pd.DataFrame(
        [
            {
                "parent_path": str(tmp_path),
                "file_name": "broken.xlsx",
                "sheet_name": "Sheet1",
                "table_range": ["A1:B2"],
            }
        ]
    )
    with caplog.at_level("ERROR"):
        samples = pipeline.build_samples(files_df, data_folder_path="/")
    assert samples == []
    assert any("Skipping" in record.message for record in caplog.records)
