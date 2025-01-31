import torch
from torch.utils.data import DataLoader
from collections import defaultdict

from .model import TableDetectionModel


def get_model(num_classes=2):
    model = TableDetectionModel(num_classes)
    return model


def get_dataloader(dataset):
    def collate_fn(batch):
        return tuple(zip(*batch))

    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    return loader


def train_model(model, train_loader, optimizer, num_epochs, device):
    # Send the model to the device (GPU or CPU)
    model.to(device)

    # Set the model in training mode
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()

            # Backpropagation
            losses.backward()
            optimizer.step()

        print(f"Epoch {epoch}: Loss: {epoch_loss / len(train_loader)}")


def calculate_iou(pred_box, gt_box):
    # Determine the coordinates of the intersection rectangle
    xA = max(pred_box[0], gt_box[0])
    yA = max(pred_box[1], gt_box[1])
    xB = min(pred_box[2], gt_box[2])
    yB = min(pred_box[3], gt_box[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    predBoxArea = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    gtBoxArea = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(predBoxArea + gtBoxArea - interArea)

    return iou


def evaluate_model(model, test_loader, device, iou_threshold=0.5):
    model.to(device)
    model.eval()

    all_detections = defaultdict(list)
    all_ground_truths = defaultdict(list)

    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                # Assuming single class; adapt as needed for multiple classes
                for box, score in zip(output["boxes"], output["scores"]):
                    all_detections[i].append((box.cpu().numpy(), score.item()))

                for gt_box in targets[i]["boxes"]:
                    all_ground_truths[i].append(gt_box.cpu().numpy())

    # Calculate TP, FP, FN for each image
    TPs, FPs, FNs = [], [], []
    for image_id, detections in all_detections.items():
        gt_boxes = all_ground_truths[image_id]
        detected = []

        for pred_box, score in sorted(detections, key=lambda x: x[1], reverse=True):
            if score < iou_threshold:
                continue

            for gt_box in gt_boxes:
                iou = calculate_iou(pred_box, gt_box)

                if iou >= iou_threshold:
                    if gt_box not in detected:
                        TPs.append((image_id, pred_box, gt_box))
                        detected.append(gt_box)
                        break
            else:
                FPs.append((image_id, pred_box))

        for gt_box in gt_boxes:
            if gt_box not in detected:
                FNs.append((image_id, gt_box))

    # Calculate precision and recall
    precision = len(TPs) / (len(TPs) + len(FPs)) if TPs or FPs else 0
    recall = len(TPs) / (len(TPs) + len(FNs)) if TPs or FNs else 0

    # Compute AP as the area under the precision-recall curve
    # This can be more complex in practice. Here's a simple version
    AP = precision * recall

    # Assuming single class. For multiple classes, calculate AP for each and then average
    mAP = AP

    return mAP


def get_model_output(model, test_loader, device):
    model.to(device)
    model.eval()  # Set the model in inference mode

    eval_loss = 0
    for images, targets in test_loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            # Forward pass
            outputs = model(images)
            print(outputs)
