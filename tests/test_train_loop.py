import os

import torch

from excel_table_cnn.model.detector import build_model
from excel_table_cnn.training.dataset import SpreadsheetDataset
from excel_table_cnn.training.train import (
    LOSS_KEYS,
    TrainConfig,
    load_checkpoint,
    save_checkpoint,
    train_model,
)
from .conftest import make_synthetic_sample


def test_train_loop_logs_all_components_and_checkpoints(tmp_path):
    dataset = SpreadsheetDataset([make_synthetic_sample(height=16, width=10, box=(2, 3, 8, 12))])
    model = build_model(in_channels=17)
    config = TrainConfig(
        epochs=2, lr=0.001, warmup_steps=5, log_every=0,
        checkpoint_dir=str(tmp_path), device="cpu", seed=0,
    )
    history = train_model(model, dataset, config)

    assert len(history) == 2  # 1 sample x 2 epochs
    for record in history:
        for key in LOSS_KEYS:
            assert key in record
            assert record[key] >= 0
        assert torch.isfinite(torch.tensor(record["loss_total"]))
    assert os.path.exists(tmp_path / "last.pt")


def test_checkpoint_round_trip(tmp_path):
    model = build_model(in_channels=17)
    path = str(tmp_path / "model.pt")
    save_checkpoint(model, path, note="test")

    restored = load_checkpoint(path)
    assert restored.in_channels == 17
    assert restored.num_classes == 2


def test_load_checkpoint_honors_score_thresh_override(tmp_path):
    """Regression: eval/detect must be able to lower the internal RoI score
    gate — the default (0.5) otherwise hides low-confidence detections from
    any external --score-threshold below it."""
    model = build_model(in_channels=17)
    path = str(tmp_path / "model.pt")
    save_checkpoint(model, path)

    restored = load_checkpoint(path, box_score_thresh=0.05)
    assert restored.model.roi_heads.score_thresh == 0.05
    original = model.state_dict()
    for key, value in restored.state_dict().items():
        assert torch.equal(value, original[key])


def test_checkpoint_stores_feature_contract(tmp_path):
    from excel_table_cnn.data.features import FEATURE_NAMES

    model = build_model(in_channels=17)
    path = str(tmp_path / "model.pt")
    save_checkpoint(model, path)
    payload = torch.load(path, weights_only=True)
    assert payload["feature_names"] == list(FEATURE_NAMES)
    assert payload["in_channels"] == 17
