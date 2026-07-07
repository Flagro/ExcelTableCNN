import torch

from excel_table_cnn.data.features import NUM_FEATURES
from excel_table_cnn.model.backbone import FCNBackbone
from excel_table_cnn.model.grid_context import (
    NUM_DERIVED_CHANNELS,
    AxialStripBlock,
    DerivedChannels,
)


def test_derived_channels_values():
    # 2 rows x 3 cols; is_empty is channel 0.
    x = torch.zeros(1, 2, 2, 3)
    x[0, 0] = torch.tensor([[1.0, 0.0, 1.0],   # row 0: 1 of 3 filled
                            [0.0, 0.0, 0.0]])  # row 1: all filled
    out = DerivedChannels(is_empty_index=0)(x)
    assert out.shape == (1, 2 + NUM_DERIVED_CHANNELS, 2, 3)

    row_density = out[0, 2]
    assert torch.allclose(row_density[0], torch.full((3,), 1 / 3))
    assert torch.allclose(row_density[1], torch.ones(3))

    col_density = out[0, 3]
    assert torch.allclose(col_density[:, 0], torch.tensor([0.5, 0.5]))
    assert torch.allclose(col_density[:, 1], torch.ones(2))

    row_coord, col_coord = out[0, 4], out[0, 5]
    assert row_coord[0, 0] == 0.0 and row_coord[1, 0] == 1.0
    assert col_coord[0, 0] == 0.0 and col_coord[0, 2] == 1.0


def test_axial_strip_block_preserves_shape():
    block = AxialStripBlock(channels=16)
    x = torch.randn(2, 16, 20, 12)
    out = block(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_backbone_with_and_without_grid_context():
    x = torch.randn(1, NUM_FEATURES, 24, 16)
    for use_grid_context in (True, False):
        backbone = FCNBackbone(NUM_FEATURES, use_grid_context=use_grid_context)
        out = backbone(x)
        assert out.shape == (1, backbone.out_channels, 24, 16)  # stride 1
        expected_in = NUM_FEATURES + (NUM_DERIVED_CHANNELS if use_grid_context else 0)
        assert backbone.body[0].in_channels == expected_in
        assert backbone.in_channels == NUM_FEATURES  # data channels, always