import torch

from excel_table_cnn.model.pbr import PBRHead, jitter_boxes


def test_jitter_stays_valid_and_targets_within_k():
    torch.manual_seed(0)
    boxes = torch.tensor([[3.0, 5.0, 11.0, 21.0], [0.0, 0.0, 2.0, 2.0]])
    for _ in range(50):
        jittered, offsets = jitter_boxes(boxes, k=7, height=30, width=15)
        assert (jittered[:, 2] > jittered[:, 0]).all()
        assert (jittered[:, 3] > jittered[:, 1]).all()
        assert (jittered[:, 0] >= 0).all() and (jittered[:, 1] >= 0).all()
        assert offsets.abs().max() <= 7
        assert torch.equal(jittered + offsets, (jittered + offsets))  # finite


def test_jitter_empty_boxes():
    boxes = torch.empty((0, 4))
    jittered, offsets = jitter_boxes(boxes, k=7, height=10, width=10)
    assert jittered.numel() == 0 and offsets.numel() == 0


def test_forward_shapes():
    head = PBRHead(in_channels=16, k=7)
    features = torch.randn(1, 16, 40, 30)
    boxes = torch.tensor([[3.0, 5.0, 11.0, 21.0], [10.0, 2.0, 20.0, 30.0]])
    logits = head(features, boxes)
    assert logits.shape == (2, 4, 15)  # N x edges x (2k+1)


def test_loss_scalar_and_refine_valid():
    torch.manual_seed(0)
    head = PBRHead(in_channels=16, k=7)
    features = torch.randn(1, 16, 40, 30)
    boxes = torch.tensor([[3.0, 5.0, 11.0, 21.0]])
    loss = head.loss(features, boxes, height=40, width=30)
    assert loss.ndim == 0 and torch.isfinite(loss) and loss > 0

    refined = head.refine(features, boxes + 0.3, height=40, width=30)
    assert refined.shape == boxes.shape
    assert (refined[:, 2] > refined[:, 0]).all()
    assert (refined[:, 3] > refined[:, 1]).all()
    assert torch.equal(refined, refined.round())  # snapped to the cell grid


def test_refine_empty():
    head = PBRHead(in_channels=16, k=7)
    features = torch.randn(1, 16, 20, 20)
    out = head.refine(features, torch.empty((0, 4)), height=20, width=20)
    assert out.numel() == 0


def test_pbr_learns_to_snap_edges():
    """Overfit the head on one synthetic feature map: given jittered boxes it
    must recover the true boundary. This is the module-level version of the
    EoB-0 promise."""
    torch.manual_seed(42)
    gt = torch.tensor([[8.0, 10.0, 20.0, 30.0]])
    features = torch.zeros(1, 8, 48, 32)
    # Distinct signal inside the table region: boundaries are visible edges.
    features[:, :4, 10:30, 8:20] = 1.0
    features[:, 4:, 10:30, 8:20] = -1.0

    head = PBRHead(in_channels=8, k=7, trunk_channels=32, hidden=64)
    optimizer = torch.optim.Adam(head.parameters(), lr=2e-3)
    for _ in range(300):
        loss = head.loss(features, gt, height=48, width=32)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.manual_seed(7)
    jittered, _ = jitter_boxes(gt.repeat(8, 1), k=7, height=48, width=32)
    with torch.no_grad():
        refined = head.refine(features, jittered, height=48, width=32)
    recovered = (refined == gt).all(dim=1).float().mean()
    assert recovered >= 0.75, f"only {recovered:.0%} of jittered boxes snapped back to GT"