import torch


def nms(boxes, scores, iou_threshold=0.5):
    kept = []
    idxs = scores.argsort(descending=True)
    while idxs.numel() > 0:
        i = idxs[0].item()
        kept.append(i)
        if idxs.numel() == 1:
            break
        current = boxes[i]
        rest = boxes[idxs[1:]]

        xx1 = torch.max(current[0], rest[:, 0])
        yy1 = torch.max(current[1], rest[:, 1])
        xx2 = torch.min(current[2], rest[:, 2])
        yy2 = torch.min(current[3], rest[:, 3])
        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        area1 = (current[2] - current[0]) * (current[3] - current[1])
        area2 = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])
        union = area1 + area2 - inter
        iou = inter / union

        remain_mask = iou < iou_threshold
        idxs = idxs[1:][remain_mask]
    return kept
