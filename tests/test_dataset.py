import torch

from excel_table_cnn.data.features import NUM_FEATURES
from excel_table_cnn.training.dataset import SpreadsheetDataset, collate_fn
from .conftest import make_synthetic_sample


def test_getitem_shapes_and_labels():
    sample = make_synthetic_sample(height=20, width=10, box=(2, 3, 8, 9))
    dataset = SpreadsheetDataset([sample])
    tensor, target = dataset[0]
    assert tensor.shape == (NUM_FEATURES, 20, 10)
    assert tensor.dtype == torch.float32
    assert target["boxes"].shape == (1, 4)
    assert target["labels"].tolist() == [1]
    assert dataset.num_cell_features == NUM_FEATURES


def test_out_of_bounds_boxes_clamped():
    sample = make_synthetic_sample(height=20, width=10, box=(2, 3, 8, 9))
    sample["boxes"] = torch.tensor([[5.0, 5.0, 50.0, 50.0]])
    dataset = SpreadsheetDataset([sample])
    _, target = dataset[0]
    assert target["boxes"].tolist() == [[5.0, 5.0, 10.0, 20.0]]


def test_sheet_with_only_degenerate_boxes_is_skipped():
    good = make_synthetic_sample(height=20, width=10, box=(2, 3, 8, 9))
    bad = make_synthetic_sample(height=20, width=10, box=(2, 3, 8, 9))
    bad["boxes"] = torch.tensor([[3.0, 3.0, 3.0, 10.0]])
    dataset = SpreadsheetDataset([good, bad])
    assert len(dataset) == 1
    assert len(dataset.skipped) == 1


def test_collate_keeps_variable_sizes():
    a = SpreadsheetDataset([make_synthetic_sample()])[0]
    b = SpreadsheetDataset([make_synthetic_sample(height=12, width=8, box=(1, 1, 5, 6))])[0]
    images, targets = collate_fn([a, b])
    assert len(images) == 2 and len(targets) == 2
    assert images[0].shape != images[1].shape
