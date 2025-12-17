"""
RoadSignNet-SAL V2: Ultra-Lightweight Anchor-Free Loss
- Focal loss for heatmap (handles class imbalance)
- L1 loss for box regression
- Focal loss for classification
- Minimal memory footprint
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        """
        Args:
            pred: Predictions [B, C, H, W] or [B, 1, H, W]
            target: Ground truth [B, C, H, W] or [B, 1, H, W]
        """
        pred = pred.clamp(1e-7, 1 - 1e-7)
        
        pos_mask = target.gt(0.5).float()  # Changed from eq(1) to gt(0.5) for gaussian targets
        neg_mask = target.le(0.5).float()
        
        # Positive loss (for object centers)
        pos_loss = -self.alpha * torch.pow(1 - pred, self.gamma) * torch.log(pred) * pos_mask * target
        # Negative loss (for background)
        neg_loss = -(1 - self.alpha) * torch.pow(pred, self.gamma) * torch.log(1 - pred) * neg_mask * (1 - target)
        
        loss = pos_loss + neg_loss
        num_pos = pos_mask.sum().clamp(min=1)
        return loss.sum() / num_pos  # Normalize by number of positive samples


class AnchorFreeLoss(nn.Module):
    """
    Memory-efficient anchor-free loss function
    Components:
    1. Heatmap loss: Focal loss for object center detection
    2. Box loss: L1 loss for bbox regression
    3. Class loss: Focal loss for classification
    """
    def __init__(self, num_classes=43, lambda_heat=1.0, lambda_box=1.0, lambda_cls=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_heat = lambda_heat
        self.lambda_box = lambda_box
        self.lambda_cls = lambda_cls
        
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        
    def _generate_heatmap(self, bboxes, labels, output_size, img_size=640):
        """
        OPTIMIZED: Generate heatmap targets using vectorized operations
        """
        batch_size = bboxes.size(0)
        max_objects = bboxes.size(1)
        h, w = output_size
        stride = img_size / h
        device = bboxes.device
        
        # Initialize targets
        heatmap = torch.zeros(batch_size, 1, h, w, device=device)
        box_targets = torch.zeros(batch_size, 4, h, w, device=device)
        class_targets = torch.zeros(batch_size, self.num_classes, h, w, device=device)
        mask = torch.zeros(batch_size, 1, h, w, device=device)
        
        # Vectorized: filter valid boxes (label >= 0 and bbox not all zeros)
        valid_mask = (labels >= 0) & (bboxes.sum(dim=-1) > 0)
        
        for b in range(batch_size):
            valid_objs = valid_mask[b].nonzero(as_tuple=False).squeeze(-1)
            if len(valid_objs) == 0:
                continue
            
            # Get all valid boxes and labels for this image
            valid_bboxes = bboxes[b, valid_objs]  # [N, 4]
            valid_labels = labels[b, valid_objs]  # [N]
            
            # Compute centers and sizes
            cx = (valid_bboxes[:, 0] + valid_bboxes[:, 2]) / 2  # [N]
            cy = (valid_bboxes[:, 1] + valid_bboxes[:, 3]) / 2
            box_w = valid_bboxes[:, 2] - valid_bboxes[:, 0]
            box_h = valid_bboxes[:, 3] - valid_bboxes[:, 1]
            
            # Skip invalid boxes
            valid_size = (box_w > 0) & (box_h > 0)
            if not valid_size.any():
                continue
            
            cx = cx[valid_size]
            cy = cy[valid_size]
            box_w = box_w[valid_size]
            box_h = box_h[valid_size]
            valid_labels = valid_labels[valid_size]
            
            # Grid coordinates
            grid_x = (cx / stride).long().clamp(0, w - 1)
            grid_y = (cy / stride).long().clamp(0, h - 1)
            
            # Fast Gaussian heatmaps: radius based on object size (min 2 for visibility)
            radius = torch.sqrt(box_w * box_h / stride / stride).clamp(min=2).long()
            
            # VECTORIZED: Create coordinate grids
            y_grid = torch.arange(h, device=device).view(-1, 1).float()  # [H, 1]
            x_grid = torch.arange(w, device=device).view(1, -1).float()  # [1, W]
            
            # For each object, compute Gaussian and add to heatmap
            for i in range(len(grid_x)):
                gx, gy = grid_x[i].float(), grid_y[i].float()
                r = radius[i].float()
                
                # Distance from center (vectorized)
                dist_x = (x_grid - gx) ** 2  # [1, W]
                dist_y = (y_grid - gy) ** 2  # [H, 1]
                dist_sq = dist_x + dist_y    # [H, W] broadcast
                
                # Gaussian with sigma = radius/2
                sigma = r / 2.0
                gaussian = torch.exp(-dist_sq / (2 * sigma * sigma))
                
                # Only apply within 2*radius (cutoff for speed)
                radius_mask = dist_sq <= (2 * r) ** 2
                gaussian = gaussian * radius_mask.float()
                
                # Take maximum to handle overlapping objects
                heatmap[b, 0] = torch.maximum(heatmap[b, 0], gaussian)
            
            # Box regression targets (only at object centers)
            grid_cx = (grid_x.float() + 0.5) * stride
            grid_cy = (grid_y.float() + 0.5) * stride
            
            dx = (cx - grid_cx) / stride
            dy = (cy - grid_cy) / stride
            dw = torch.log(box_w / stride + 1e-6)
            dh = torch.log(box_h / stride + 1e-6)
            
            # Use proper indexing for 1D tensors (vectorized)
            for i in range(len(grid_x)):
                gx_idx, gy_idx = grid_x[i].item(), grid_y[i].item()
                box_targets[b, 0, gy_idx, gx_idx] = dx[i]
                box_targets[b, 1, gy_idx, gx_idx] = dy[i]
                box_targets[b, 2, gy_idx, gx_idx] = dw[i]
                box_targets[b, 3, gy_idx, gx_idx] = dh[i]
                
                # Class targets
                class_targets[b, valid_labels[i], gy_idx, gx_idx] = 1.0
                
                # Mask
                mask[b, 0, gy_idx, gx_idx] = 1.0
        
        return heatmap, box_targets, class_targets, mask
    
    def forward(self, predictions, targets, bboxes, labels):
        """
        Compute anchor-free loss
        
        Args:
            predictions: (heatmap, boxes, classes) from model
            targets: Not used (kept for compatibility)
            bboxes: [B, max_objects, 4] ground truth boxes (x1y1x2y2 normalized)
            labels: [B, max_objects] class labels
            
        Returns:
            total_loss: Scalar loss
            loss_dict: Dictionary with loss components
        """
        pred_heatmap, pred_boxes, pred_classes = predictions
        
        batch_size = pred_heatmap.size(0)
        h, w = pred_heatmap.size(2), pred_heatmap.size(3)
        
        # Generate targets
        target_heatmap, target_boxes, target_classes, mask = self._generate_heatmap(
            bboxes, labels, output_size=(h, w)
        )
        
        # 1. Heatmap loss (focal loss) - calculate on all pixels
        heat_loss = self.focal_loss(pred_heatmap, target_heatmap)
        
        # 2. Box regression loss (L1 loss, only at object locations)
        num_pos = mask.sum().clamp(min=1)
        if num_pos > 1:  # Only calculate if we have objects
            box_loss = F.l1_loss(pred_boxes * mask, target_boxes * mask, reduction='sum') / num_pos
        else:
            box_loss = torch.tensor(0.0, device=pred_boxes.device)
        
        # 3. Classification loss (focal loss, only at object locations)
        # Apply sigmoid to predictions
        pred_classes_sig = torch.sigmoid(pred_classes)
        
        # Expand mask to match number of classes
        class_mask = mask.expand(-1, self.num_classes, -1, -1)
        
        # Focal loss for classification
        if num_pos > 1:
            cls_loss = self.focal_loss(pred_classes_sig * class_mask, target_classes * class_mask)
        else:
            cls_loss = torch.tensor(0.0, device=pred_classes.device)
        
        # Total loss
        total_loss = (
            self.lambda_heat * heat_loss +
            self.lambda_box * box_loss +
            self.lambda_cls * cls_loss
        )
        
        loss_dict = {
            'heat_loss': heat_loss.item(),
            'box_loss': box_loss.item(),
            'cls_loss': cls_loss.item(),
            'num_pos': int(num_pos.item())
        }
        
        return total_loss, loss_dict


def create_loss_v2(num_classes=43, lambda_heat=1.0, lambda_box=1.0, lambda_cls=1.0):
    """Factory function to create V2 loss"""
    return AnchorFreeLoss(
        num_classes=num_classes,
        lambda_heat=lambda_heat,
        lambda_box=lambda_box,
        lambda_cls=lambda_cls
    )


class AnchorFreeDecoder:
    """Decode anchor-free predictions to bounding boxes"""
    
    def __init__(self, num_classes=43, conf_thresh=0.1, nms_thresh=0.45, max_detections=100):
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh  # Lower threshold for Gaussian heatmaps
        self.nms_thresh = nms_thresh
        self.max_detections = max_detections
    
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
    
    def decode(self, predictions):
        """
        Decode anchor-free predictions to boxes.
        Compatible with batch_size=1 evaluation (returns unpacked tuple).
        
        Args:
            predictions: tuple of (heatmap, boxes, classes)
                - heatmap: [B, 1, H, W] - object center heatmap
                - boxes: [B, 4, H, W] - box offsets (x, y, w, h)
                - classes: [B, num_classes, H, W] - class predictions
        
        Returns:
            For batch_size=1: (boxes, scores, classes)
            - boxes: [N, 4] in format [x1, y1, x2, y2]
            - scores: [N]
            - classes: [N]
        """
        heatmap, boxes, classes = predictions
        B, _, H, W = heatmap.shape
        device = heatmap.device
        
        # Apply sigmoid to get probabilities
        heatmap = torch.sigmoid(heatmap)
        classes = torch.sigmoid(classes)
        
        # Process first image (batch_size=1 for evaluation)
        b = 0
        heat = heatmap[b, 0]  # [H, W]
        box = boxes[b]  # [4, H, W]
        cls = classes[b]  # [num_classes, H, W]
        
        # Find peaks in heatmap
        peak_mask = heat > self.conf_thresh
        
        if peak_mask.sum() == 0:
            # No detections
            return (
                torch.zeros((0, 4), device=device),
                torch.zeros(0, device=device),
                torch.zeros(0, dtype=torch.long, device=device)
            )
        
        # Get peak positions
        peak_indices = torch.where(peak_mask)
        py, px = peak_indices  # y, x positions
        
        # Get heatmap scores at peaks
        heat_scores = heat[py, px]
        
        # Get boxes at peak positions
        dx = box[0, py, px]  # x offset from grid center (normalized)
        dy = box[1, py, px]  # y offset from grid center (normalized)
        dw = box[2, py, px]  # log(width) (normalized)
        dh = box[3, py, px]  # log(height) (normalized)
        
        # Stride for V2 model (416x416 -> 52x52 output, so stride=8)
        stride = 8.0
        
        # Convert to pixel coordinates
        # Grid cell center in pixels
        grid_cx = (px.float() + 0.5) * stride
        grid_cy = (py.float() + 0.5) * stride
        
        # Add offsets to get actual center
        cx = grid_cx + dx * stride
        cy = grid_cy + dy * stride
        
        # Decode log width/height
        w = torch.exp(dw.clamp(-10, 10)) * stride
        h = torch.exp(dh.clamp(-10, 10)) * stride
        
        # Convert to x1y1x2y2 format
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        det_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        
        # Get class predictions at peak positions
        cls_at_peaks = cls[:, py, px]  # [num_classes, num_peaks]
        cls_scores, cls_ids = cls_at_peaks.max(dim=0)  # [num_peaks]
        
        # Combine heatmap score and class score
        final_scores = heat_scores * cls_scores
        
        # Filter by confidence threshold
        conf_mask = final_scores > self.conf_thresh
        if conf_mask.sum() == 0:
            return (
                torch.zeros((0, 4), device=device),
                torch.zeros(0, device=device),
                torch.zeros(0, dtype=torch.long, device=device)
            )
        
        det_boxes = det_boxes[conf_mask]
        final_scores = final_scores[conf_mask]
        cls_ids = cls_ids[conf_mask]
        
        # Apply NMS per class
        keep_boxes = []
        keep_scores = []
        keep_classes = []
        
        for c in range(self.num_classes):
            class_mask = cls_ids == c
            if class_mask.sum() == 0:
                continue
            
            c_boxes = det_boxes[class_mask]
            c_scores = final_scores[class_mask]
            
            # Apply NMS
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
