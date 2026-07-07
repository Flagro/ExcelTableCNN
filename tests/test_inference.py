import re

from excel_table_cnn.inference import detect_tables
from excel_table_cnn.model.detector import build_model
from excel_table_cnn.training.train import seed_everything
from .conftest import TOY_SHEET

RANGE_PATTERN = re.compile(r"^[A-Z]+[0-9]+:[A-Z]+[0-9]+$")


def test_detect_tables_output_schema(toy_workbook_path):
    seed_everything(0)
    model = build_model(in_channels=17, box_score_thresh=0.0)
    detections = detect_tables(
        toy_workbook_path, model=model, score_threshold=0.0
    )
    assert isinstance(detections, list)
    assert len(detections) > 0  # untrained but with threshold 0 something comes out
    for det in detections:
        assert det["sheet"] == TOY_SHEET
        assert RANGE_PATTERN.match(det["range"]), det["range"]
        assert 0.0 <= det["score"] <= 1.0


def test_score_threshold_filters_everything(toy_workbook_path):
    seed_everything(0)
    model = build_model(in_channels=17, box_score_thresh=0.0)
    detections = detect_tables(toy_workbook_path, model=model, score_threshold=1.1)
    assert detections == []


def test_explicit_sheet_name(toy_workbook_path):
    seed_everything(0)
    model = build_model(in_channels=17, box_score_thresh=0.0)
    detections = detect_tables(
        toy_workbook_path, sheet_name=TOY_SHEET, model=model, score_threshold=0.0
    )
    assert all(det["sheet"] == TOY_SHEET for det in detections)


def test_untrained_warning_when_no_model_given(toy_workbook_path, caplog):
    with caplog.at_level("WARNING"):
        detect_tables(toy_workbook_path, score_threshold=0.99, device="cpu")
    assert any("randomly initialized" in record.message for record in caplog.records)


def test_detect_tables_on_native_xls(toy_xls_path):
    """Inference reads legacy .xls directly — no LibreOffice."""
    seed_everything(0)
    model = build_model(in_channels=17, box_score_thresh=0.0)
    detections = detect_tables(
        toy_xls_path, model=model, score_threshold=0.0, device="cpu"
    )
    assert isinstance(detections, list)
    for det in detections:
        assert det["sheet"] == TOY_SHEET
        assert RANGE_PATTERN.match(det["range"])
