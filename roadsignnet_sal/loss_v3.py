"""
RoadSignNet-SAL V3 Loss Function
Multi-scale anchor-free loss with proper coordinate handling

Fixes from V2:
- Correct normalized-to-pixel coordinate conversion
- Multi-scale target assignment
- Improved loss balancing
- Better positive sample selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: predictions (logits or probabilities)
            targets: ground truth (0 or 1)
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate pt
        pt = torch.exp(-bce_loss)
        
        # Apply focal term
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SimpleCenterAssigner:
    """
    Simple center-based assignment (proven to work)
    Assigns the grid cell containing the object center as positive
    """
    def __init__(self, radius=1):
        self.radius = radius  # Assign neighboring cells within radius
    
    def assign(self, pred_scores, pred_boxes, gt_boxes, gt_labels, stride, grid_h, grid_w):
        """
        Assign ground truth to grid cells based on object centers
        
        Args:
            pred_scores: [B, H*W, num_classes] - predicted class scores (unused, for compatibility)
            pred_boxes: [B, H*W, 4] - predicted boxes (unused, for compatibility)
            gt_boxes: [B, max_obj, 4] - ground truth boxes in pixel coords [x1,y1,x2,y2]
            gt_labels: [B, max_obj] - ground truth labels
            stride: int - downsample stride
            grid_h, grid_w: int - grid dimensions
        
        Returns:
            target_labels: [B, H*W] - assigned class labels (-1 for ignore)
            target_boxes: [B, H*W, 4] - assigned box targets
            target_scores: [B, H*W, num_classes] - target scores (one-hot)
            fg_mask: [B, H*W] - foreground mask
        """
        batch_size = gt_boxes.size(0)
        num_anchors = grid_h * grid_w
        num_classes = pred_scores.size(2)
        device = gt_boxes.device
        
        target_labels = torch.full((batch_size, num_anchors), -1, dtype=torch.long, device=device)
        target_boxes = torch.zeros((batch_size, num_anchors, 4), device=device)
        target_scores = torch.zeros((batch_size, num_anchors, num_classes), device=device)
        fg_mask = torch.zeros((batch_size, num_anchors), dtype=torch.bool, device=device)
        
        # Process each image in batch
        for b in range(batch_size):
            # Get valid ground truth (label >= 0 and not padding)
            valid_mask = gt_labels[b] >= 0
            if not valid_mask.any():
                continue
            
            gt_bbox = gt_boxes[b][valid_mask]  # [n_gt, 4]
            gt_label = gt_labels[b][valid_mask]  # [n_gt]
            
            # Calculate centers
            gt_cx = (gt_bbox[:, 0] + gt_bbox[:, 2]) / 2  # [n_gt]
            gt_cy = (gt_bbox[:, 1] + gt_bbox[:, 3]) / 2  # [n_gt]
            
            # Convert to grid coordinates
            grid_x = (gt_cx / stride).long().clamp(0, grid_w - 1)  # [n_gt]
            grid_y = (gt_cy / stride).long().clamp(0, grid_h - 1)  # [n_gt]
            
            # Assign each GT to its center cell and neighbors
            for gt_idx in range(len(gt_label)):
                cx, cy = grid_x[gt_idx], grid_y[gt_idx]
                
                # Assign center cell and radius neighbors
                for dy in range(-self.radius, self.radius + 1):
                    for dx in range(-self.radius, self.radius + 1):
                        gx = cx + dx
                        gy = cy + dy
                        
                        # Check bounds
                        if 0 <= gx < grid_w and 0 <= gy < grid_h:
                            anchor_idx = gy * grid_w + gx
                            
                            fg_mask[b, anchor_idx] = True
                            target_labels[b, anchor_idx] = gt_label[gt_idx]
                            target_boxes[b, anchor_idx] = gt_bbox[gt_idx]
                            target_scores[b, anchor_idx, gt_label[gt_idx]] = 1.0
        
        return target_labels, target_boxes, target_scores, fg_mask


class V3Loss(nn.Module):
    """
    Multi-scale anchor-free loss for RoadSignNet-SAL V3
    
    Key improvements:
    - Proper coordinate conversion (fixed V2 bug)
    - Multi-scale target assignment
    - Task-aligned assignment (better positive sample selection)
    - Balanced loss weighting
    """
    
    def __init__(self, num_classes=43, strides=[8, 16, 32], 
                 lambda_cls=0.5, lambda_box=7.5, lambda_obj=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.lambda_cls = lambda_cls
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.assigner = SimpleCenterAssigner(radius=1)
    
    def forward(self, predictions, targets, bboxes, labels):
        """
        Compute multi-scale loss
        
        Args:
            predictions: List of 3 tuples [(cls, reg, obj), ...] from 3 scales
            targets: Not used (kept for compatibility)
            bboxes: [B, max_objects, 4] - ground truth boxes in NORMALIZED [0,1] format [x1,y1,x2,y2]
            labels: [B, max_objects] - class labels
        
        Returns:
            total_loss: scalar
            loss_dict: dict with loss components
        """
        device = bboxes.device
        batch_size = bboxes.size(0)
        
        # Convert normalized bboxes to pixel coordinates (416x416 image)
        img_size = 416
        bboxes_pixel = bboxes.clone()
        bboxes_pixel[..., [0, 2]] *= img_size  # x coordinates
        bboxes_pixel[..., [1, 3]] *= img_size  # y coordinates
        
        total_cls_loss = 0
        total_box_loss = 0
        total_obj_loss = 0
        total_num_fg = 0
        
        # Process each scale
        for scale_idx, (pred_cls, pred_reg, pred_obj) in enumerate(predictions):
            stride = self.strides[scale_idx]
            B, C, H, W = pred_cls.shape
            
            # Reshape predictions to [B, H*W, ...]
            pred_cls_flat = pred_cls.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)  # [B, H*W, C]
            pred_reg_flat = pred_reg.permute(0, 2, 3, 1).reshape(B, -1, 4)  # [B, H*W, 4]
            pred_obj_flat = pred_obj.permute(0, 2, 3, 1).reshape(B, -1)  # [B, H*W]
            
            # Generate grid anchors
            grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2).float()  # [H*W, 2]
            
            # Decode predictions to pixel coordinates
            # pred_reg format: [delta_x, delta_y, log_w, log_h] relative to grid
            pred_boxes_decoded = self._decode_boxes(pred_reg_flat, grid_xy, stride)  # [B, H*W, 4] in pixels
            
            # Assign targets using center-based assignment
            target_labels, target_boxes, target_scores, fg_mask = self.assigner.assign(
                pred_cls_flat.detach().sigmoid(),
                pred_boxes_decoded.detach(),
                bboxes_pixel,
                labels,
                stride,
                H,
                W
            )
            
            num_fg = fg_mask.sum()
            total_num_fg += num_fg
            
            if num_fg > 0:
                # Classification loss (only on foreground)
                cls_loss = self.focal_loss(
                    pred_cls_flat[fg_mask],
                    target_scores[fg_mask]
                )
                
                # Box regression loss (IoU loss)
                pred_boxes_fg = pred_boxes_decoded[fg_mask]
                target_boxes_fg = target_boxes[fg_mask]
                box_loss = self._iou_loss(pred_boxes_fg, target_boxes_fg)
                
                # Objectness loss (BCE loss on all samples)
                obj_target = torch.zeros_like(pred_obj_flat)
                obj_target[fg_mask] = 1.0
                obj_loss = F.binary_cross_entropy_with_logits(pred_obj_flat, obj_target, reduction='sum') / num_fg
            else:
                # No foreground samples - still compute background objectness loss
                cls_loss = torch.tensor(0.0, device=device, requires_grad=True)
                box_loss = torch.tensor(0.0, device=device, requires_grad=True)
                obj_target = torch.zeros_like(pred_obj_flat)
                obj_loss = F.binary_cross_entropy_with_logits(pred_obj_flat, obj_target, reduction='mean')
            
            total_cls_loss += cls_loss
            total_box_loss += box_loss
            total_obj_loss += obj_loss
        
        # Average across scales
        num_scales = len(predictions)
        total_cls_loss /= num_scales
        total_box_loss /= num_scales
        total_obj_loss /= num_scales
        
        # Combined loss
        total_loss = (
            self.lambda_cls * total_cls_loss +
            self.lambda_box * total_box_loss +
            self.lambda_obj * total_obj_loss
        )
        
        loss_dict = {
            'cls_loss': total_cls_loss.item(),
            'box_loss': total_box_loss.item(),
            'obj_loss': total_obj_loss.item(),
            'num_fg': int(total_num_fg.item())
        }
        
        return total_loss, loss_dict
    
    def _decode_boxes(self, pred_reg, grid_xy, stride):
        """
        Decode box predictions to pixel coordinates
        
        Args:
            pred_reg: [B, H*W, 4] - predicted offsets [dx, dy, log_w, log_h]
            grid_xy: [H*W, 2] - grid centers
            stride: int - downsample stride
        
        Returns:
            boxes: [B, H*W, 4] - boxes in pixel coords [x1, y1, x2, y2]
        """
        B = pred_reg.size(0)
        grid_xy = grid_xy.unsqueeze(0).expand(B, -1, -1)  # [B, H*W, 2]
        
        # Decode center
        pred_xy = (grid_xy + pred_reg[..., :2]) * stride  # [B, H*W, 2]
        
        # Decode width/height
        pred_wh = torch.exp(pred_reg[..., 2:].clamp(-10, 10)) * stride  # [B, H*W, 2]
        
        # Convert to x1y1x2y2
        pred_x1y1 = pred_xy - pred_wh / 2
        pred_x2y2 = pred_xy + pred_wh / 2
        
        boxes = torch.cat([pred_x1y1, pred_x2y2], dim=-1)
        return boxes
    
    def _iou_loss(self, pred_boxes, target_boxes):
        """
        IoU loss (actually 1 - IoU for minimization)
        
        Args:
            pred_boxes: [N, 4] - predicted boxes [x1, y1, x2, y2]
            target_boxes: [N, 4] - target boxes [x1, y1, x2, y2]
        
        Returns:
            loss: scalar
        """
        # Calculate intersection
        inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        
        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        
        # Calculate union
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        union_area = pred_area + target_area - inter_area
        
        iou = inter_area / (union_area + 1e-7)
        loss = (1 - iou).mean()
        
        return loss


def create_loss_v3(num_classes=43, strides=[8, 16, 32], 
                   lambda_cls=0.5, lambda_box=7.5, lambda_obj=1.0):
    """Factory function to create V3 loss"""
    return V3Loss(
        num_classes=num_classes,
        strides=strides,
        lambda_cls=lambda_cls,
        lambda_box=lambda_box,
        lambda_obj=lambda_obj
    )


class V3Decoder:
    """Decode V3 predictions to bounding boxes"""
    
    def __init__(self, num_classes=43, conf_thresh=0.25, nms_thresh=0.45, max_detections=300, strides=[8, 16, 32]):
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.max_detections = max_detections
        self.strides = strides
    
    def decode(self, predictions):
        """
        Decode multi-scale predictions to boxes
        
        Args:
            predictions: List of 3 tuples [(cls, reg, obj), ...] from 3 scales
        
        Returns:
            boxes: [N, 4] in format [x1, y1, x2, y2]
            scores: [N]
            classes: [N]
        """
        all_boxes = []
        all_scores = []
        all_classes = []
        
        # Process each scale
        for scale_idx, (pred_cls, pred_reg, pred_obj) in enumerate(predictions):
            stride = self.strides[scale_idx]
            B, C, H, W = pred_cls.shape
            device = pred_cls.device
            
            # Apply sigmoid
            pred_cls = pred_cls.sigmoid()
            pred_obj = pred_obj.sigmoid()
            
            # Generate grid
            grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2).float()
            
            # Reshape predictions
            pred_cls_flat = pred_cls[0].permute(1, 2, 0).reshape(-1, self.num_classes)
            pred_reg_flat = pred_reg[0].permute(1, 2, 0).reshape(-1, 4)
            pred_obj_flat = pred_obj[0].reshape(-1)
            
            # Decode boxes
            pred_xy = (grid_xy + pred_reg_flat[:, :2]) * stride
            pred_wh = torch.exp(pred_reg_flat[:, 2:].clamp(-10, 10)) * stride
            pred_x1y1 = pred_xy - pred_wh / 2
            pred_x2y2 = pred_xy + pred_wh / 2
            boxes = torch.cat([pred_x1y1, pred_x2y2], dim=-1)
            
            # Get class scores and classes
            cls_scores, cls_ids = pred_cls_flat.max(dim=1)
            
            # Combine with objectness
            final_scores = cls_scores * pred_obj_flat
            
            # Filter by confidence
            conf_mask = final_scores > self.conf_thresh
            if conf_mask.sum() > 0:
                all_boxes.append(boxes[conf_mask])
                all_scores.append(final_scores[conf_mask])
                all_classes.append(cls_ids[conf_mask])
        
        if len(all_boxes) == 0:
            return (
                torch.zeros((0, 4), device=device),
                torch.zeros(0, device=device),
                torch.zeros(0, dtype=torch.long, device=device)
            )
        
        # Concatenate all scales
        all_boxes = torch.cat(all_boxes, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        all_classes = torch.cat(all_classes, dim=0)
        
        # Apply NMS per class
        keep_boxes = []
        keep_scores = []
        keep_classes = []
        
        for c in range(self.num_classes):
            class_mask = all_classes == c
            if class_mask.sum() == 0:
                continue
            
            c_boxes = all_boxes[class_mask]
            c_scores = all_scores[class_mask]
            
            # NMS
            keep_indices = self._nms(c_boxes, c_scores, self.nms_thresh)
            
            if len(keep_indices) > 0:
                keep_boxes.append(c_boxes[keep_indices])
                keep_scores.append(c_scores[keep_indices])
                keep_classes.append(torch.full((len(keep_indices),), c, dtype=torch.long, device=device))
        
        if len(keep_boxes) == 0:
            return (
                torch.zeros((0, 4), device=device),
                torch.zeros(0, device=device),
                torch.zeros(0, dtype=torch.long, device=device)
            )
        
        final_boxes = torch.cat(keep_boxes, dim=0)
        final_scores = torch.cat(keep_scores, dim=0)
        final_classes = torch.cat(keep_classes, dim=0)
        
        # Limit max detections
        if len(final_scores) > self.max_detections:
            top_indices = final_scores.argsort(descending=True)[:self.max_detections]
            final_boxes = final_boxes[top_indices]
            final_scores = final_scores[top_indices]
            final_classes = final_classes[top_indices]
        
        return final_boxes, final_scores, final_classes
    
    def _nms(self, boxes, scores, iou_threshold):
        """Non-Maximum Suppression"""
        if len(boxes) == 0:
            return torch.tensor([], dtype=torch.long)
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort(descending=True)
        
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break
            
            i = order[0].item()
            keep.append(i)
            
            xx1 = torch.maximum(x1[i], x1[order[1:]])
            yy1 = torch.maximum(y1[i], y1[order[1:]])
            xx2 = torch.minimum(x2[i], x2[order[1:]])
            yy2 = torch.minimum(y2[i], y2[order[1:]])
            
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            
            inds = torch.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return torch.tensor(keep, dtype=torch.long)
