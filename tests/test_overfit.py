"""M0 smoke test: the model must be able to overfit a single synthetic sheet.

This is the project's core regression gate. Both historic failure modes —
loss collapsing to 0 without learning, and construction bugs that crash
training — fail this test. If it is red, nothing downstream matters.
"""

import pytest
import torch

from excel_table_cnn.evaluation.eob import eob
from excel_table_cnn.data.features import NUM_FEATURES
from excel_table_cnn.model.detector import build_model
from excel_table_cnn.training.dataset import SpreadsheetDataset
from excel_table_cnn.training.train import TrainConfig, train_model
from .conftest import make_synthetic_sample

GT_BOX = (3, 5, 11, 21)


@pytest.mark.slow
def test_overfit_single_sheet():
    sample = make_synthetic_sample(height=30, width=15, box=GT_BOX)
    dataset = SpreadsheetDataset([sample])
    model = build_model(in_channels=NUM_FEATURES, box_score_thresh=0.05)

    config = TrainConfig(
        epochs=250,  # 1 sample per epoch = 250 steps
        lr=0.01,
        warmup_steps=50,
        seed=42,
        device="cpu",
        log_every=0,
    )
    history = train_model(model, dataset, config)

    first = sum(r["loss_total"] for r in history[:10]) / 10
    last = sum(r["loss_total"] for r in history[-10:]) / 10
    assert last < first * 0.5, f"loss did not halve: {first:.4f} -> {last:.4f}"

    # Early in training every component must be alive — a silent zero here
    # is the historic "loss goes to 0 but nothing works" bug.
    early = history[:5]
    for key in ("loss_objectness", "loss_classifier"):
        assert all(r[key] > 0 for r in early), f"{key} is zero from the start"

    model.eval()
    with torch.no_grad():
        output = model([sample["tensor"]])[0]
    assert len(output["boxes"]) > 0, "overfit model detects nothing on its own sheet"
    top_box = output["boxes"][int(output["scores"].argmax())].tolist()
    error = eob(top_box, GT_BOX)
    # With the PBR snapping head the gate demands cell-exact boundaries.
    assert error == 0, f"top detection {top_box} vs GT {GT_BOX}: EoB={error}"
