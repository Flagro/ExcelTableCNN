import torch

from excel_table_cnn.evaluation.evaluate import evaluate_model, format_report
from excel_table_cnn.training.dataset import SpreadsheetDataset
from .conftest import make_synthetic_sample


class OracleModel(torch.nn.Module):
    """Fake detector that returns configured boxes — lets the harness be
    tested independently of a real model."""

    def __init__(self, boxes, scores):
        super().__init__()
        self._boxes = boxes
        self._scores = scores

    def forward(self, images):
        return [
            {
                "boxes": torch.tensor(self._boxes, dtype=torch.float32),
                "scores": torch.tensor(self._scores, dtype=torch.float32),
                "labels": torch.ones(len(self._boxes), dtype=torch.int64),
            }
        ]


def test_perfect_oracle_scores_perfectly():
    sample = make_synthetic_sample(box=(3, 5, 11, 21))
    dataset = SpreadsheetDataset([sample])
    model = OracleModel([[3, 5, 11, 21]], [0.99])

    report = evaluate_model(model, dataset, score_threshold=0.5)
    assert report["eob0"] == {"precision": 1.0, "recall": 1.0, "tp": 1, "fp": 0, "fn": 0}
    assert report["n_sheets"] == 1
    assert report["n_gt"] == 1
    detail = report["per_sheet"][0]
    assert detail["predictions"][0]["range"] == detail["ground_truth"][0]
    assert detail["best_eob_per_gt"] == [0]


def test_near_miss_counts_at_eob2_only():
    sample = make_synthetic_sample(box=(3, 5, 11, 21))
    dataset = SpreadsheetDataset([sample])
    model = OracleModel([[3, 5, 13, 21]], [0.9])  # right edge off by 2

    report = evaluate_model(model, dataset, score_threshold=0.5)
    assert report["eob0"]["recall"] == 0.0
    assert report["eob2"]["recall"] == 1.0
    assert report["per_sheet"][0]["best_eob_per_gt"] == [2]


def test_low_scores_are_filtered():
    sample = make_synthetic_sample(box=(3, 5, 11, 21))
    dataset = SpreadsheetDataset([sample])
    model = OracleModel([[3, 5, 11, 21]], [0.3])

    report = evaluate_model(model, dataset, score_threshold=0.5)
    assert report["eob0"]["fn"] == 1
    assert report["per_sheet"][0]["predictions"] == []


def test_format_report_is_readable():
    sample = make_synthetic_sample(box=(3, 5, 11, 21))
    dataset = SpreadsheetDataset([sample])
    model = OracleModel([[3, 5, 11, 21]], [0.99])
    text = format_report(evaluate_model(model, dataset))
    assert "EoB-0" in text and "EoB-2" in text
    assert "precision=1.000" in text
