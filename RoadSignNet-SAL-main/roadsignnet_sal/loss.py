"""
RoadSignNet-SAL: Loss Functions
Focal Loss + CIoU Loss + BCE Loss with Proper Anchor Matching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


def box_iou(box1, box2):
    """Calculate IoU between boxes [N, 4] and [M, 4] in x1y1x2y2 format"""
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    inter_x1 = torch.max(box1[:, None, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[:, 3])
    
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)


def ciou_loss(pred_boxes, target_boxes, eps=1e-7):
    """CIoU loss for bounding boxes in x1y1x2y2 format"""
    # IoU
    inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    union_area = pred_area + target_area - inter_area + eps
    
    iou = inter_area / union_area
    
    # Center distance
    pred_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    target_cx = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    target_cy = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
    
    center_dist = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
    
    # Enclosing box diagonal
    enc_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    enc_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    enc_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    enc_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    enc_diag = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps
    
    # Aspect ratio
    pred_w = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(eps)
    pred_h = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(eps)
    target_w = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(eps)
    target_h = (target_boxes[:, 3] - target_boxes[:, 1]).clamp(eps)
    
    v = (4 / (math.pi ** 2)) * (torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h)) ** 2
    
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    
    ciou = iou - center_dist / enc_diag - alpha * v
    return 1 - ciou


class RoadSignNetLoss(nn.Module):
    """Combined loss for RoadSignNet-SAL with proper anchor matching"""
    def __init__(self, num_classes=3, lambda_cls=1.0, lambda_box=5.0, lambda_obj=1.0, img_size=640):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_cls = lambda_cls
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.img_size = img_size
        
        # Anchors for each scale (relative to feature map stride)
        # Strides: 8, 16, 32 for the 3 detection heads
        self.strides = [8, 16, 32]
        self.anchors = [
            [[10, 13], [16, 30], [33, 23]],    # Small objects
            [[30, 61], [62, 45], [59, 119]],   # Medium objects
            [[116, 90], [156, 198], [373, 326]]  # Large objects
        ]
        self.num_anchors = 3
        
        self.bce_cls = nn.BCEWithLogitsLoss(reduction='none')
        self.bce_obj = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, predictions, targets, gt_boxes, gt_labels):
        """
        Calculate loss with proper anchor-GT matching.
        
        Args:
            predictions: List of (cls_pred, box_pred, obj_pred) from each detection head
            targets: Not used
            gt_boxes: [B, max_boxes, 4] in x1y1x2y2 pixel coords
            gt_labels: [B, max_boxes], -1 for padding
        """
        device = predictions[0][0].device
        batch_size = gt_boxes.shape[0]
        
        total_cls_loss = torch.tensor(0.0, device=device)
        total_box_loss = torch.tensor(0.0, device=device)
        total_obj_loss = torch.tensor(0.0, device=device)
        
        num_pos_total = 0
        
        for scale_idx, (cls_pred, box_pred, obj_pred) in enumerate(predictions):
            B, _, H, W = obj_pred.shape
            stride = self.strides[scale_idx]
            anchors = torch.tensor(self.anchors[scale_idx], device=device, dtype=torch.float32)
            
            # Build targets for this scale
            obj_target = torch.zeros(B, self.num_anchors, H, W, device=device)
            cls_target = torch.zeros(B, self.num_anchors, self.num_classes, H, W, device=device)
            box_target = torch.zeros(B, self.num_anchors, 4, H, W, device=device)
            pos_mask = torch.zeros(B, self.num_anchors, H, W, device=device, dtype=torch.bool)
            
            for b in range(B):
                valid_mask = gt_labels[b] >= 0
                if valid_mask.sum() == 0:
                    continue
                
                boxes = gt_boxes[b][valid_mask]  # [N, 4] x1y1x2y2
                labels = gt_labels[b][valid_mask]  # [N]
                
                # Convert to center format
                cx = (boxes[:, 0] + boxes[:, 2]) / 2
                cy = (boxes[:, 1] + boxes[:, 3]) / 2
                w = boxes[:, 2] - boxes[:, 0]
                h = boxes[:, 3] - boxes[:, 1]
                
                # Grid indices
                gx = (cx / stride).long().clamp(0, W - 1)
                gy = (cy / stride).long().clamp(0, H - 1)
                
                # Assign to best matching anchor based on aspect ratio
                for i in range(len(boxes)):
                    box_w, box_h = w[i].item(), h[i].item()
                    if box_w < 1 or box_h < 1:
                        continue
                    
                    # Find best anchor by IoU of sizes
                    ratios = []
                    for aw, ah in self.anchors[scale_idx]:
                        ratio = min(box_w / aw, aw / box_w) * min(box_h / ah, ah / box_h)
                        ratios.append(ratio)
                    
                    best_anchor = max(range(len(ratios)), key=lambda x: ratios[x])
                    
                    # Assign target
                    gi, gj = gx[i].item(), gy[i].item()
                    
                    obj_target[b, best_anchor, gj, gi] = 1.0
                    pos_mask[b, best_anchor, gj, gi] = True
                    
                    # Class target (one-hot)
                    cls_target[b, best_anchor, labels[i].long(), gj, gi] = 1.0
                    
                    # Box target: tx, ty, tw, th
                    # tx, ty are offsets from grid cell
                    tx = cx[i] / stride - gi
                    ty = cy[i] / stride - gj
                    # tw, th are log-scale relative to anchor
                    aw, ah = anchors[best_anchor]
                    tw = torch.log(w[i] / aw + 1e-6)
                    th = torch.log(h[i] / ah + 1e-6)
                    
                    box_target[b, best_anchor, 0, gj, gi] = tx
                    box_target[b, best_anchor, 1, gj, gi] = ty
                    box_target[b, best_anchor, 2, gj, gi] = tw
                    box_target[b, best_anchor, 3, gj, gi] = th
                    
                    num_pos_total += 1
            
            # Reshape predictions for loss calculation
            # cls_pred: [B, num_anchors * num_classes, H, W]
            # box_pred: [B, num_anchors * 4, H, W]
            # obj_pred: [B, num_anchors, H, W]
            
            cls_pred_r = cls_pred.view(B, self.num_anchors, self.num_classes, H, W)
            box_pred_r = box_pred.view(B, self.num_anchors, 4, H, W)
            
            # Objectness loss - all positions, but weight positives more
            obj_loss = self.bce_obj(obj_pred, obj_target)
            # Balance: positive weight = 1.0, negative weight = 0.5
            obj_weight = torch.where(obj_target > 0.5, 
                                     torch.ones_like(obj_target),
                                     torch.ones_like(obj_target) * 0.5)
            obj_loss = (obj_loss * obj_weight).mean()
            total_obj_loss += obj_loss
            
            # Classification and box loss - only on positive positions
            if pos_mask.sum() > 0:
                # Get positive indices
                pos_indices = torch.where(pos_mask)
                batch_idx, anchor_idx, grid_y, grid_x = pos_indices
                
                # Classification loss - gather predictions and targets at positive locations
                cls_pred_pos = cls_pred_r[batch_idx, anchor_idx, :, grid_y, grid_x]  # [N_pos, num_classes]
                cls_target_pos = cls_target[batch_idx, anchor_idx, :, grid_y, grid_x]  # [N_pos, num_classes]
                cls_loss = self.bce_cls(cls_pred_pos, cls_target_pos).mean()
                total_cls_loss += cls_loss
                
                # Box regression loss
                box_pred_pos = box_pred_r[batch_idx, anchor_idx, :, grid_y, grid_x]  # [N_pos, 4]
                box_target_pos = box_target[batch_idx, anchor_idx, :, grid_y, grid_x]  # [N_pos, 4]
                
                # Smooth L1 on tx, ty (sigmoid for offset)
                xy_loss = F.smooth_l1_loss(torch.sigmoid(box_pred_pos[:, :2]), 
                                           box_target_pos[:, :2], reduction='mean')
                # Smooth L1 on tw, th
                wh_loss = F.smooth_l1_loss(box_pred_pos[:, 2:], 
                                           box_target_pos[:, 2:], reduction='mean')
                box_loss = xy_loss + wh_loss
                total_box_loss += box_loss
        
        num_scales = len(predictions)
        total_obj_loss /= num_scales
        
        if num_pos_total > 0:
            total_cls_loss /= num_scales
            total_box_loss /= num_scales
        else:
            total_cls_loss = torch.tensor(0.0, device=device, requires_grad=True)
            total_box_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Combine losses
        total_loss = (self.lambda_cls * total_cls_loss + 
                     self.lambda_box * total_box_loss + 
                     self.lambda_obj * total_obj_loss)
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'cls_loss': total_cls_loss.item() if isinstance(total_cls_loss, torch.Tensor) else total_cls_loss,
            'box_loss': total_box_loss.item() if isinstance(total_box_loss, torch.Tensor) else total_box_loss,
            'obj_loss': total_obj_loss.item(),
            'num_pos': num_pos_total
        }
        
        return total_loss, loss_dict


class DetectionDecoder:
    """Decode model predictions to bounding boxes with NMS"""
    
    def __init__(self, num_classes=3, conf_thresh=0.25, iou_thresh=0.45, img_size=640):
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.img_size = img_size
        
        self.strides = [8, 16, 32]
        self.anchors = [
            [[10, 13], [16, 30], [33, 23]],
            [[30, 61], [62, 45], [59, 119]],
            [[116, 90], [156, 198], [373, 326]]
        ]
        self.num_anchors = 3
    
    def decode(self, predictions):
        """
        Decode predictions to boxes.
        Returns: boxes [N, 4], scores [N], classes [N]
        """
        device = predictions[0][0].device
        all_boxes = []
        all_scores = []
        all_classes = []
        
        for scale_idx, (cls_pred, box_pred, obj_pred) in enumerate(predictions):
            B, _, H, W = obj_pred.shape
            stride = self.strides[scale_idx]
            anchors = torch.tensor(self.anchors[scale_idx], device=device, dtype=torch.float32)
            
            # Create grid
            yv, xv = torch.meshgrid(torch.arange(H, device=device), 
                                    torch.arange(W, device=device), indexing='ij')
            grid = torch.stack([xv, yv], dim=-1).float()  # [H, W, 2]
            
            # Reshape predictions
            cls_pred = cls_pred.view(B, self.num_anchors, self.num_classes, H, W)
            box_pred = box_pred.view(B, self.num_anchors, 4, H, W)
            
            # Get objectness scores
            obj_scores = torch.sigmoid(obj_pred)  # [B, num_anchors, H, W]
            
            for b in range(B):
                for a in range(self.num_anchors):
                    obj = obj_scores[b, a]  # [H, W]
                    
                    # Find confident positions
                    mask = obj > self.conf_thresh
                    if mask.sum() == 0:
                        continue
                    
                    # Get indices
                    indices = torch.where(mask)
                    gy, gx = indices
                    
                    # Decode boxes
                    tx = torch.sigmoid(box_pred[b, a, 0, gy, gx])
                    ty = torch.sigmoid(box_pred[b, a, 1, gy, gx])
                    tw = box_pred[b, a, 2, gy, gx]
                    th = box_pred[b, a, 3, gy, gx]
                    
                    # Convert to pixel coordinates
                    cx = (gx.float() + tx) * stride
                    cy = (gy.float() + ty) * stride
                    w = torch.exp(tw.clamp(-10, 10)) * anchors[a, 0]
                    h = torch.exp(th.clamp(-10, 10)) * anchors[a, 1]
                    
                    # To x1y1x2y2
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    
                    boxes = torch.stack([x1, y1, x2, y2], dim=-1)
                    
                    # Get class predictions
                    cls_scores = torch.sigmoid(cls_pred[b, a, :, gy, gx])  # [num_classes, N]
                    cls_scores = cls_scores.T  # [N, num_classes]
                    
                    # Combined confidence
                    obj_conf = obj[gy, gx]  # [N]
                    conf = obj_conf.unsqueeze(-1) * cls_scores  # [N, num_classes]
                    
                    # Best class per detection
                    max_conf, max_cls = conf.max(dim=-1)
                    
                    all_boxes.append(boxes)
                    all_scores.append(max_conf)
                    all_classes.append(max_cls)
        
        if len(all_boxes) == 0:
            return torch.empty(0, 4, device=device), torch.empty(0, device=device), torch.empty(0, device=device, dtype=torch.long)
        
        boxes = torch.cat(all_boxes, dim=0)
        scores = torch.cat(all_scores, dim=0)
        classes = torch.cat(all_classes, dim=0)
        
        # Apply NMS per class
        keep_boxes = []
        keep_scores = []
        keep_classes = []
        
        for c in range(self.num_classes):
            cls_mask = classes == c
            if cls_mask.sum() == 0:
                continue
            
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            
            # NMS
            keep_idx = self.nms(cls_boxes, cls_scores, self.iou_thresh)
            
            keep_boxes.append(cls_boxes[keep_idx])
            keep_scores.append(cls_scores[keep_idx])
            keep_classes.append(torch.full((len(keep_idx),), c, device=device, dtype=torch.long))
        
        if len(keep_boxes) == 0:
            return torch.empty(0, 4, device=device), torch.empty(0, device=device), torch.empty(0, device=device, dtype=torch.long)
        
        return torch.cat(keep_boxes), torch.cat(keep_scores), torch.cat(keep_classes)
    
    def nms(self, boxes, scores, iou_threshold):
        """Non-Maximum Suppression"""
        if len(boxes) == 0:
            return torch.empty(0, dtype=torch.long, device=boxes.device)
        
        # Sort by score
        order = scores.argsort(descending=True)
        
        keep = []
        while len(order) > 0:
            i = order[0].item()
            keep.append(i)
            
            if len(order) == 1:
                break
            
            # Calculate IoU with rest
            rest_boxes = boxes[order[1:]]
            ious = self.compute_iou(boxes[i:i+1], rest_boxes).squeeze(0)
            
            # Keep boxes with IoU < threshold
            mask = ious < iou_threshold
            order = order[1:][mask]
        
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)
    
    def compute_iou(self, box1, box2):
        """Compute IoU between box1 [1, 4] and box2 [N, 4]"""
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        
        inter_x1 = torch.max(box1[:, 0], box2[:, 0])
        inter_y1 = torch.max(box1[:, 1], box2[:, 1])
        inter_x2 = torch.min(box1[:, 2], box2[:, 2])
        inter_y2 = torch.min(box1[:, 3], box2[:, 3])
        
        inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        union = area1 + area2 - inter
        
        return inter / (union + 1e-6)