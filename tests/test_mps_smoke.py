"""Apple Silicon compatibility gate: every op the model uses (convs,
GroupNorm, RPN, NMS, RoIAlign, detection postprocess) must run on MPS.

Kept short — MPS is *supported*, not auto-selected (measured slower than the
M-series CPU cores for this small model; see excel_table_cnn.device).
"""

import pytest
import torch

from excel_table_cnn.model.detector import build_model
from excel_table_cnn.training.dataset import SpreadsheetDataset
from .conftest import make_synthetic_sample

requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS backend not available"
)


@requires_mps
@pytest.mark.slow
def test_train_and_eval_steps_run_on_mps():
    device = torch.device("mps")
    dataset = SpreadsheetDataset([make_synthetic_sample()])
    image, target = dataset[0]
    image = image.to(device)
    target = {k: v.to(device) for k, v in target.items()}

    model = build_model(in_channels=17, box_score_thresh=0.0).to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    for _ in range(3):
        loss_dict = model([image], [target])
        total = sum(loss_dict.values())
        assert torch.isfinite(total), loss_dict
        optimizer.zero_grad()
        total.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        output = model([image])[0]
    assert output["boxes"].device.type == "mps"
    assert torch.isfinite(output["boxes"]).all()
