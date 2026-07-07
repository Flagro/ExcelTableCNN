import pytest
import torch

from excel_table_cnn.model.detector import build_model
from excel_table_cnn.training.dataset import SpreadsheetDataset
from excel_table_cnn.training.train import LOSS_KEYS, seed_everything
from .conftest import make_synthetic_sample


@pytest.fixture()
def sample_batch():
    dataset = SpreadsheetDataset([make_synthetic_sample()])
    tensor, target = dataset[0]
    return [tensor], [target]


def test_model_constructs_with_arbitrary_channels():
    for channels in (17, 20):
        model = build_model(in_channels=channels)
        assert model.in_channels == channels
        assert model.backbone.body[0].in_channels == channels


def test_wrong_channel_input_raises(sample_batch):
    """Regression for the num_classes/in_channels argument mix-up: a model
    built for N channels must reject differently-shaped input loudly."""
    images, targets = sample_batch
    model = build_model(in_channels=5)
    model.train()
    with pytest.raises(RuntimeError):
        model(images, targets)


def test_training_forward_returns_all_loss_components(sample_batch):
    seed_everything(0)
    images, targets = sample_batch
    model = build_model(in_channels=17)
    model.train()
    loss_dict = model(images, targets)

    assert set(LOSS_KEYS) == set(loss_dict.keys())
    for key in LOSS_KEYS:
        value = loss_dict[key]
        assert torch.isfinite(value), f"{key} is not finite"
        assert value.item() >= 0
    # With random weights on a sheet containing a table, classification
    # losses cannot be zero — a zero here is the historic "loss 0" failure.
    assert loss_dict["loss_objectness"].item() > 0
    assert loss_dict["loss_classifier"].item() > 0


def test_eval_forward_returns_detection_dicts(sample_batch):
    seed_everything(0)
    images, _ = sample_batch
    model = build_model(in_channels=17, box_score_thresh=0.0)
    model.eval()
    with torch.no_grad():
        outputs = model(images)
    assert len(outputs) == 1
    output = outputs[0]
    assert set(output.keys()) >= {"boxes", "scores", "labels"}
    assert output["boxes"].shape[1] == 4
    assert output["scores"].shape[0] == output["boxes"].shape[0]
    if len(output["scores"]):
        assert float(output["scores"].max()) <= 1.0
        assert float(output["scores"].min()) >= 0.0


def test_degenerate_target_rejected_by_torchvision():
    """The dataset layer must filter degenerate boxes because torchvision
    refuses them — pin that contract."""
    sample = make_synthetic_sample()
    tensor = sample["tensor"]
    bad_target = {
        "boxes": torch.tensor([[3.0, 5.0, 3.0, 21.0]]),  # zero width
        "labels": torch.ones((1,), dtype=torch.int64),
    }
    model = build_model(in_channels=17)
    model.train()
    with pytest.raises(Exception):
        model([tensor], [bad_target])
