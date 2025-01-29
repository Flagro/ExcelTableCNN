from torch.utils.data import DataLoader
import torch
import torch.optim as optim

from .model.model import TableDetectionModel
from .experimental_model.model import TableDetectionModel as ExperimentalModel


def get_model(in_channels=3, debug=False):
    model = TableDetectionModel(in_channels)
    return model


def get_dataloader(dataset):
    def collate_fn(batch):
        return tuple(zip(*batch))

    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    return loader


def get_experimental_model(in_channels=3, debug=False):
    model = ExperimentalModel(in_channels, debug)
    return model


def get_experimental_dataloader(dataset):
    return DataLoader(dataset, batch_size=None, shuffle=True)


def train_one_epoch(model, dataloader, optimizer, device="cpu"):
    model.train()
    epoch_losses = []
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        gt_boxes = [targets["boxes"].to(device)]
        gt_labels = [targets["labels"].to(device)]
        # if inputs are not in batch form, add a batch dimension
        if len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(0)  # (B=1,C,H,W)

        optimizer.zero_grad()
        finals, losses_dict = model(inputs, gt_boxes, gt_labels)
        total_loss = sum(losses_dict.values())
        total_loss.backward()
        optimizer.step()
        epoch_losses.append(total_loss.item())
    return sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0


def train_model(model, train_dataloader, num_epochs, device="cpu"):
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

    tenth_epoch = num_epochs // 10
    for epoch in range(num_epochs):
        loss_val = train_one_epoch(model, train_dataloader, optimizer, device=device)
        if epoch % tenth_epoch == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch} - Loss: {loss_val:.4f}")


def evaluate_model(model, test_dataset, test_dataloader, device="cpu"):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs = inputs.to(device).unsqueeze(0)
            finals, _ = model(inputs)
            all_preds.append(finals[0].cpu())  # shape (N,5)

    for i, bboxes in enumerate(all_preds):
        rounded_bboxes = set(
            [tuple([round(float(el)) for el in box][:4]) for box in bboxes]
        )
        expected_boxes = set(
            [tuple([int(el) for el in box]) for box in test_dataset[i][1]["boxes"]]
        )
        print(
            f"Spreadsheet {i} predicted final boxes: {rounded_bboxes}, expected: {expected_boxes}"
        )
