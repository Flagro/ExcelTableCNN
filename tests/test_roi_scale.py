"""Regression: RoI pooling scale must not depend on the first sheet seen.

torchvision's MultiScaleRoIAlign lazily infers its spatial scale from the
first forward pass (padded feature size vs unpadded image size, rounded to a
power of two) and caches it forever. With the stride-1 backbone the true
scale is always 1, but a short first sheet (height <= 22, padded to 32 rows)
used to lock scale 2 and silently corrupt every subsequent detection —
surfacing as "a reloaded checkpoint performs far worse than the in-memory
model" whenever the two objects saw different first sheets.
"""

import torch

from excel_table_cnn.data.features import NUM_FEATURES
from excel_table_cnn.model.detector import build_model
from .conftest import make_synthetic_sample

TALL = torch.zeros((NUM_FEATURES, 100, 60))  # pads to 128 rows: ratio < sqrt(2)
SHORT = torch.zeros((NUM_FEATURES, 12, 60))  # pads to 32 rows: ratio > sqrt(2)


def test_roi_scale_pinned_at_construction():
    model = build_model(in_channels=NUM_FEATURES)
    pooler = model.model.roi_heads.box_roi_pool
    assert pooler.scales == [1.0]
    assert pooler.map_levels is not None


def test_first_sheet_size_does_not_change_predictions():
    model_a = build_model(in_channels=NUM_FEATURES)
    model_b = build_model(in_channels=NUM_FEATURES)
    model_b.load_state_dict(model_a.state_dict())
    model_a.eval()
    model_b.eval()

    probe = make_synthetic_sample(height=30, width=15, box=(3, 5, 11, 21))["tensor"]
    with torch.no_grad():
        model_a([TALL])
        model_b([SHORT])
        assert model_a.model.roi_heads.box_roi_pool.scales == [1.0]
        assert model_b.model.roi_heads.box_roi_pool.scales == [1.0]

        out_a = model_a([probe])[0]
        out_b = model_b([probe])[0]

    assert torch.equal(out_a["boxes"], out_b["boxes"])
    assert torch.equal(out_a["scores"], out_b["scores"])
