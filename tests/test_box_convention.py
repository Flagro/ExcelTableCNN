import pytest
import torch

from excel_table_cnn.training.dataset import (
    box_to_range,
    parse_table_range,
    validate_boxes,
)


def test_parse_basic():
    assert parse_table_range("A1:C3") == [0.0, 0.0, 3.0, 3.0]


def test_parse_single_cell_has_positive_extent():
    # Regression: the old inclusive convention produced zero-area boxes here.
    box = parse_table_range("A1:A1")
    assert box == [0.0, 0.0, 1.0, 1.0]
    assert box[2] > box[0] and box[3] > box[1]


def test_parse_single_column_has_positive_width():
    box = parse_table_range("A1:A5")
    assert box == [0.0, 0.0, 1.0, 5.0]
    assert box[2] > box[0]


@pytest.mark.parametrize("rng", ["A1:C3", "B2:H45", "AA10:AB12", "A1:A1", "D7:D20"])
def test_round_trip(rng):
    assert box_to_range(parse_table_range(rng)) == rng


def test_box_to_range_rounds_float_predictions():
    assert box_to_range([0.3, 0.4, 2.8, 3.2]) == "A1:C3"


def test_parse_unbounded_range_raises():
    with pytest.raises(ValueError):
        parse_table_range("A:C")


def test_parse_garbage_raises():
    with pytest.raises(ValueError):
        parse_table_range("not a range")


def test_validate_boxes_clamps_out_of_bounds():
    boxes = torch.tensor([[5.0, 5.0, 50.0, 50.0]])
    result = validate_boxes(boxes, height=20, width=10)
    assert result.tolist() == [[5.0, 5.0, 10.0, 20.0]]


def test_validate_boxes_drops_degenerate():
    boxes = torch.tensor([[3.0, 3.0, 3.0, 10.0], [1.0, 1.0, 4.0, 4.0]])
    result = validate_boxes(boxes, height=20, width=10)
    assert result.tolist() == [[1.0, 1.0, 4.0, 4.0]]


def test_validate_boxes_empty_input():
    result = validate_boxes(torch.empty((0, 4)), height=5, width=5)
    assert result.shape == (0, 4)
