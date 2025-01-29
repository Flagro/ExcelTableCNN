import torch
import torch.nn as nn
import torch.nn.functional as F
from .fcn_backbone import FCNBackbone
from .cell_featurizer import CellFeaturizer
from .anchor_generator import AnchorGenerator
from .roi_align import RoIAlign
from .detection_head import DetectionHead
from .nms import nms


class TableDetectionModel(nn.Module):
    def __init__(self, in_channels=20, debug=False):
        super().__init__()
        self.debug = debug

        # Featurizer + Backbone
        self.featurizer = CellFeaturizer(in_channels=in_channels, out_channels=32)
        self.backbone = FCNBackbone(in_channels=32, out_channels=64)

        # Anchor generator
        self.anchor_gen = AnchorGenerator(debug=debug)

        # RPN pre conv
        self.rpn_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Build RPN heads after we know #anchors
        # We'll do it dynamically based on anchor_gen, but let's see:
        scales_count = len(self.anchor_gen.scales)
        ratios_count = len(self.anchor_gen.ratios)
        A = scales_count * ratios_count  # #anchors per loc
        self.rpn_obj = nn.Conv2d(64, A, 1)
        self.rpn_bbox = nn.Conv2d(64, A * 4, 1)

        # RoIAlign
        self.roialign = RoIAlign(out_size=7, debug=debug)

        # DetectionHead
        self.det_head = DetectionHead(
            in_channels=64, pool_size=7, hidden_dim=128, debug=debug
        )

        # RPN/Proposal parameters
        self.rpn_nms_thresh = 0.5  # 0.7
        self.rpn_pre_nms_topk = 2000
        self.rpn_post_nms_topk = 200
        self.proposal_match_iou_thr = 0.5

        # Detection parameters
        self.det_nms_thresh = 0.3
        self.det_score_thresh = 0.15  # can lower to see more predictions

    def forward(self, x, gt_boxes=None, gt_labels=None):
        B, _, H, W = x.shape
        if self.debug:
            print(
                f"\n[Forward] Input shape={x.shape}, GT boxes={[b.shape for b in gt_boxes] if gt_boxes else None}"
            )

        # 1) Featurizer + Backbone
        feat = self.featurizer(x)  # (B,32,H,W)
        feat = self.backbone(feat)  # (B,64,H,W)
        if self.debug:
            print(
                f"[Forward] After backbone => feat.shape={feat.shape}, min={feat.min().item()}, max={feat.max().item()}"
            )

        # 2) Generate anchors
        anchors = self.anchor_gen(feat)  # (N,4)

        # 3) RPN forward
        rpn_in = F.relu(self.rpn_conv(feat))
        rpn_obj_logits = self.rpn_obj(rpn_in)  # (B,A,H,W)
        rpn_bbox_deltas = self.rpn_bbox(rpn_in)  # (B,A*4,H,W)

        # Flatten
        A = rpn_obj_logits.shape[1]  # #anchors per loc
        rpn_obj_logits = rpn_obj_logits.permute(0, 2, 3, 1).reshape(B, -1)  # (B,N)
        rpn_bbox_deltas = rpn_bbox_deltas.permute(0, 2, 3, 1).reshape(
            B, -1, 4
        )  # (B,N,4)
        if self.debug:
            print(
                f"[RPN] rpn_obj_logits.shape={rpn_obj_logits.shape}, rpn_bbox_deltas.shape={rpn_bbox_deltas.shape}"
            )

        # 4) Convert obj_logits => prob
        rpn_obj_probs = torch.sigmoid(rpn_obj_logits)  # (B,N)
        if self.debug:
            print(
                f"[RPN] rpn_obj_probs min={rpn_obj_probs.min().item()}, max={rpn_obj_probs.max().item()}"
            )

        # 5) Decode proposals
        proposals_batch = []
        for i in range(B):
            dec = self.decode(anchors, rpn_bbox_deltas[i], H, W)  # clamp to image
            sco = rpn_obj_probs[i]
            # topk
            topk = min(self.rpn_pre_nms_topk, dec.shape[0])
            sco, idxs = sco.topk(topk)
            dec = dec[idxs]

            # nms
            keep_idx = nms(dec, sco, self.rpn_nms_thresh)
            keep_idx = keep_idx[: self.rpn_post_nms_topk]
            dec = dec[keep_idx]
            sco = sco[keep_idx]
            proposals_batch.append((dec, sco))
            if self.debug:
                print(
                    f"[RPN] item={i}, proposals before NMS topk={topk}, kept={len(keep_idx)}, proposal range=({dec.min(dim=0).values}, {dec.max(dim=0).values}), scores=({sco.min().item()}, {sco.max().item()})"
                )

        # 6) If training, assign proposals to GT => ROIAlign => DetectionHead => losses
        if (gt_boxes is not None) and (gt_labels is not None):
            proposals_list = [p[0] for p in proposals_batch]
            roi_boxes, roi_idx, cls_tg, box_tg = self.assign_targets(
                proposals_list, gt_boxes, gt_labels
            )
            roi_feats = self.roialign_for_batch(feat, roi_boxes, roi_idx)
            cls_logits, bbox_deltas, pbr_deltas = self.det_head(roi_feats)
            losses = self.compute_losses(
                cls_logits, bbox_deltas, pbr_deltas, cls_tg, box_tg
            )

            # For final detection, do a simplified approach: same as inference
            final = self.postprocess(feat, proposals_batch)
            return final, losses
        else:
            final = self.postprocess(feat, proposals_batch)
            return final, {}

    def decode(self, anchors, deltas, H, W):
        """
        anchors: (N,4)
        deltas: (N,4)
        clamp to image range [0,W],[0,H]
        """
        ax1, ay1, ax2, ay2 = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
        aw = ax2 - ax1
        ah = ay2 - ay1
        acx = ax1 + 0.5 * aw
        acy = ay1 + 0.5 * ah

        dx, dy, dw, dh = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]
        px = acx + dx * aw
        py = acy + dy * ah
        pw = aw * torch.exp(dw)
        ph = ah * torch.exp(dh)

        x1 = px - 0.5 * pw
        y1 = py - 0.5 * ph
        x2 = px + 0.5 * pw
        y2 = py + 0.5 * ph

        # clamp
        x1 = x1.clamp(min=0, max=W - 1)
        x2 = x2.clamp(min=0, max=W - 1)
        y1 = y1.clamp(min=0, max=H - 1)
        y2 = y2.clamp(min=0, max=H - 1)
        return torch.stack([x1, y1, x2, y2], dim=1)

    def assign_targets(self, proposals_list, gt_boxes_list, gt_labels_list):
        # See prior example for more robust matching. We do IoU≥0.5 => label=1 else 0
        roi_boxes_all = []
        roi_idx_all = []
        cls_all = []
        box_all = []

        for i, (props, gt_bxs, gt_lbls) in enumerate(
            zip(proposals_list, gt_boxes_list, gt_labels_list)
        ):
            if gt_bxs.numel() == 0:
                # background
                labels_i = torch.zeros(
                    (props.shape[0],), dtype=torch.long, device=props.device
                )
                boxes_i = torch.zeros(
                    (props.shape[0], 4), dtype=torch.float32, device=props.device
                )
                roi_boxes_all.append(props)
                roi_idx_all.append(
                    torch.full(
                        (props.shape[0],), i, device=props.device, dtype=torch.long
                    )
                )
                cls_all.append(labels_i)
                box_all.append(boxes_i)
                continue

            ious = self.iou_many(props, gt_bxs)
            max_ious, max_ids = ious.max(dim=1)
            labels_i = torch.zeros(
                (props.shape[0],), dtype=torch.long, device=props.device
            )
            pos_mask = max_ious >= self.proposal_match_iou_thr
            labels_i[pos_mask] = 1
            matched_gt = gt_bxs[max_ids]
            offsets = self.encode(props, matched_gt)
            offsets[~pos_mask] = 0.0

            roi_boxes_all.append(props)
            roi_idx_all.append(
                torch.full((props.shape[0],), i, device=props.device, dtype=torch.long)
            )
            cls_all.append(labels_i)
            box_all.append(offsets)

        rois = torch.cat(roi_boxes_all, dim=0) if roi_boxes_all else torch.empty((0, 4))
        idxs = (
            torch.cat(roi_idx_all, dim=0)
            if roi_idx_all
            else torch.empty((0,), dtype=torch.long)
        )
        clss = (
            torch.cat(cls_all, dim=0)
            if cls_all
            else torch.empty((0,), dtype=torch.long)
        )
        boxes = torch.cat(box_all, dim=0) if box_all else torch.empty((0, 4))
        return rois, idxs, clss, boxes

    def encode(self, anchors, gt):
        ax1, ay1, ax2, ay2 = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
        aw = ax2 - ax1
        ah = ay2 - ay1
        acx = ax1 + 0.5 * aw
        acy = ay1 + 0.5 * ah

        gx1, gy1, gx2, gy2 = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]
        gw = gx2 - gx1
        gh = gy2 - gy1
        gcx = gx1 + 0.5 * gw
        gcy = gy1 + 0.5 * gh

        dx = (gcx - acx) / aw
        dy = (gcy - acy) / ah
        dw = torch.log(gw / aw)
        dh = torch.log(gh / ah)
        return torch.stack([dx, dy, dw, dh], dim=1)

    def roialign_for_batch(self, feat, rois, roi_idxs):
        B, C, H, W = feat.shape
        splitted = [[] for _ in range(B)]
        for j in range(rois.shape[0]):
            i = roi_idxs[j].item()
            splitted[i].append(rois[j])

        pooled = []
        for i in range(B):
            if len(splitted[i]) == 0:
                continue
            local_feat = feat[i].unsqueeze(0)
            for box in splitted[i]:
                x1, y1, x2, y2 = box
                x1c = max(0, min(x1.item(), W - 1))
                x2c = max(0, min(x2.item(), W - 1))
                y1c = max(0, min(y1.item(), H - 1))
                y2c = max(0, min(y2.item(), H - 1))
                sl = local_feat[:, :, int(y1c) : int(y2c + 1), int(x1c) : int(x2c + 1)]
                resized = F.interpolate(
                    sl, size=(7, 7), mode="bilinear", align_corners=False
                )
                pooled.append(resized.squeeze(0))
        if len(pooled) == 0:
            return torch.empty((0, C, 7, 7), device=feat.device)
        out = torch.stack(pooled, dim=0)
        if self.debug:
            print(
                f"[roialign_for_batch] ROI feats shape={out.shape}, min={out.min()}, max={out.max()}"
            )
        return out

    def compute_losses(
        self, cls_logits, bbox_deltas, pbr_deltas, cls_targets, box_targets
    ):
        if cls_logits.shape[0] == 0:
            return {
                "cls_loss": torch.tensor(0.0, device=cls_logits.device),
                "bbox_loss": torch.tensor(0.0, device=cls_logits.device),
                "pbr_loss": torch.tensor(0.0, device=cls_logits.device),
            }
        cls_loss = F.cross_entropy(cls_logits, cls_targets)
        pos_mask = cls_targets == 1
        if pos_mask.sum() > 0:
            bbox_loss = F.smooth_l1_loss(bbox_deltas[pos_mask], box_targets[pos_mask])
            pbr_loss = F.smooth_l1_loss(pbr_deltas[pos_mask], box_targets[pos_mask])
        else:
            bbox_loss = torch.tensor(0.0, device=cls_logits.device)
            pbr_loss = torch.tensor(0.0, device=cls_logits.device)
        return {"cls_loss": cls_loss, "bbox_loss": bbox_loss, "pbr_loss": pbr_loss}

    def postprocess(self, feat, proposals_batch):
        B, C, H, W = feat.shape
        finals = []
        for i in range(B):
            boxes_i, scores_i = proposals_batch[i]
            if boxes_i.shape[0] == 0:
                finals.append(torch.empty((0, 5), device=feat.device))
                continue

            # ROIAlign
            local_feat = feat[i].unsqueeze(0)
            pooled = []
            for box in boxes_i:
                x1, y1, x2, y2 = box
                x1c = max(0, min(x1.item(), W - 1))
                x2c = max(0, min(x2.item(), W - 1))
                y1c = max(0, min(y1.item(), H - 1))
                y2c = max(0, min(y2.item(), H - 1))
                sl = local_feat[:, :, int(y1c) : int(y2c + 1), int(x1c) : int(x2c + 1)]
                resized = F.interpolate(
                    sl, size=(7, 7), mode="bilinear", align_corners=False
                )
                pooled.append(resized.squeeze(0))
            if len(pooled) == 0:
                finals.append(torch.empty((0, 5), device=feat.device))
                continue
            roifeat = torch.stack(pooled, dim=0)
            cls_log, bbox_del, pbr_del = self.det_head(roifeat)
            probs = F.softmax(cls_log, dim=1)
            table_prob = probs[:, 1]
            # decode boxes from the proposals
            decode_coarse = self.decode(boxes_i.to(bbox_del.device), bbox_del, H, W)
            decode_final = self.decode(decode_coarse, pbr_del, H, W)

            # combine with table_prob
            keep = table_prob >= self.det_score_thresh
            decode_final = decode_final[keep]
            table_prob = table_prob[keep]

            if decode_final.shape[0] == 0:
                finals.append(torch.empty((0, 5), device=feat.device))
                continue

            # final NMS
            keep2 = nms(decode_final, table_prob, self.det_nms_thresh)
            finaldets = torch.cat(
                [decode_final[keep2], table_prob[keep2].unsqueeze(1)], dim=1
            )  # (N,5)
            finals.append(finaldets)
            if self.debug:
                print(
                    f"[postprocess] item={i}, finaldets.shape={finaldets.shape}, minCoord={finaldets[:,:4].min()}, maxCoord={finaldets[:,:4].max()}, minScore={finaldets[:,4].min()}, maxScore={finaldets[:,4].max()}"
                )
        return finals

    def iou_many(self, boxes1, boxes2):
        N = boxes1.shape[0]
        M = boxes2.shape[0]
        b1 = boxes1[:, None, :]
        b2 = boxes2[None, :, :]

        ix1 = torch.max(b1[..., 0], b2[..., 0])
        iy1 = torch.max(b1[..., 1], b2[..., 1])
        ix2 = torch.min(b1[..., 2], b2[..., 2])
        iy2 = torch.min(b1[..., 3], b2[..., 3])
        iw = (ix2 - ix1).clamp(min=0)
        ih = (iy2 - iy1).clamp(min=0)
        inter = iw * ih
        area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
        area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
        union = area1 + area2 - inter
        return inter / union
