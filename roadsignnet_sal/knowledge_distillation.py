"""
RoadSignNet-SAL: Knowledge Distillation Module
Transfer knowledge from teacher YOLO model to student RoadSignNet-SAL

Knowledge Distillation Components:
1. Feature Distillation - Match intermediate feature maps
2. Logit Distillation - Soft label matching
3. Attention Transfer - Match attention maps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDistillationLoss(nn.Module):
    """
    Feature-level knowledge distillation
    Matches intermediate feature maps between teacher and student
    """
    
    def __init__(self, student_channels, teacher_channels, temperature=4.0):
        super().__init__()
        self.temperature = temperature
        
        # Adaptation layers to match channel dimensions
        self.adaptors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(s_ch, t_ch, 1, bias=False),
                nn.BatchNorm2d(t_ch)
            ) for s_ch, t_ch in zip(student_channels, teacher_channels)
        ])
    
    def forward(self, student_features, teacher_features):
        """
        Args:
            student_features: List of student feature maps
            teacher_features: List of teacher feature maps
        
        Returns:
            Feature distillation loss
        """
        total_loss = 0
        
        for adaptor, s_feat, t_feat in zip(self.adaptors, student_features, teacher_features):
            # Adapt student features to teacher dimension
            s_adapted = adaptor(s_feat)
            
            # Resize if needed
            if s_adapted.shape[-2:] != t_feat.shape[-2:]:
                s_adapted = F.interpolate(s_adapted, size=t_feat.shape[-2:], mode='bilinear', align_corners=False)
            
            # L2 loss between normalized features
            s_norm = F.normalize(s_adapted, p=2, dim=1)
            t_norm = F.normalize(t_feat.detach(), p=2, dim=1)
            
            loss = F.mse_loss(s_norm, t_norm)
            total_loss += loss
        
        return total_loss / len(student_features)


class LogitDistillationLoss(nn.Module):
    """
    Logit-level knowledge distillation (Hinton et al.)
    Uses soft labels from teacher for training student
    """
    
    def __init__(self, temperature=3.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # Weight between soft and hard labels
    
    def forward(self, student_logits, teacher_logits, hard_targets=None):
        """
        Args:
            student_logits: Student classification logits
            teacher_logits: Teacher classification logits
            hard_targets: Ground truth labels (optional)
        
        Returns:
            Distillation loss
        """
        # Soft labels from teacher
        soft_targets = F.softmax(teacher_logits.detach() / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # KL divergence loss (scaled by T^2 as per Hinton)
        soft_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean') * (self.temperature ** 2)
        
        if hard_targets is not None:
            # Combine with hard label loss
            hard_loss = F.cross_entropy(student_logits, hard_targets)
            return self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return soft_loss


class AttentionTransferLoss(nn.Module):
    """
    Attention Transfer Loss (Zagoruyko & Komodakis)
    Transfers attention maps from teacher to student
    """
    
    def __init__(self, p=2):
        super().__init__()
        self.p = p  # Power for attention map computation
    
    def attention_map(self, features):
        """Compute attention map as sum of absolute values across channels"""
        return F.normalize(features.pow(self.p).mean(dim=1), p=2, dim=(1, 2))
    
    def forward(self, student_features, teacher_features):
        """
        Args:
            student_features: List of student feature maps
            teacher_features: List of teacher feature maps
        
        Returns:
            Attention transfer loss
        """
        total_loss = 0
        
        for s_feat, t_feat in zip(student_features, teacher_features):
            # Resize if needed
            if s_feat.shape[-2:] != t_feat.shape[-2:]:
                s_feat = F.interpolate(s_feat, size=t_feat.shape[-2:], mode='bilinear', align_corners=False)
            
            s_att = self.attention_map(s_feat)
            t_att = self.attention_map(t_feat.detach())
            
            loss = (s_att - t_att).pow(2).mean()
            total_loss += loss
        
        return total_loss / len(student_features)


class BBoxDistillationLoss(nn.Module):
    """
    Bounding box regression distillation
    Student learns from teacher's localization predictions
    """
    
    def __init__(self, loss_type='l1'):
        super().__init__()
        self.loss_type = loss_type
    
    def forward(self, student_boxes, teacher_boxes, mask=None):
        """
        Args:
            student_boxes: Student box predictions [B, N, 4]
            teacher_boxes: Teacher box predictions [B, N, 4]
            mask: Foreground mask [B, N]
        
        Returns:
            Box distillation loss
        """
        if mask is not None:
            student_boxes = student_boxes[mask]
            teacher_boxes = teacher_boxes[mask]
        
        if len(student_boxes) == 0:
            return torch.tensor(0.0, device=student_boxes.device)
        
        if self.loss_type == 'l1':
            return F.l1_loss(student_boxes, teacher_boxes.detach())
        elif self.loss_type == 'smooth_l1':
            return F.smooth_l1_loss(student_boxes, teacher_boxes.detach())
        else:
            return F.mse_loss(student_boxes, teacher_boxes.detach())


class KnowledgeDistillationLoss(nn.Module):
    """
    Complete Knowledge Distillation Loss
    Combines feature, logit, attention, and bbox distillation
    """
    
    def __init__(self, 
                 student_channels=[64, 128, 256],
                 teacher_channels=[64, 128, 256],
                 temperature=3.0,
                 lambda_feat=0.5,
                 lambda_logit=1.0,
                 lambda_att=0.5,
                 lambda_bbox=0.5):
        super().__init__()
        
        self.lambda_feat = lambda_feat
        self.lambda_logit = lambda_logit
        self.lambda_att = lambda_att
        self.lambda_bbox = lambda_bbox
        
        self.feature_loss = FeatureDistillationLoss(student_channels, teacher_channels, temperature)
        self.logit_loss = LogitDistillationLoss(temperature)
        self.attention_loss = AttentionTransferLoss()
        self.bbox_loss = BBoxDistillationLoss()
    
    def forward(self, student_output, teacher_output, 
                student_features=None, teacher_features=None,
                hard_targets=None, fg_mask=None):
        """
        Compute combined distillation loss
        
        Args:
            student_output: Dict with 'cls', 'box', 'obj' predictions
            teacher_output: Dict with 'cls', 'box', 'obj' predictions
            student_features: List of intermediate feature maps
            teacher_features: List of intermediate feature maps
            hard_targets: Ground truth labels
            fg_mask: Foreground mask
        
        Returns:
            total_loss: Combined distillation loss
            loss_dict: Individual loss components
        """
        total_loss = 0
        loss_dict = {}
        
        # Logit distillation (classification)
        if 'cls' in student_output and 'cls' in teacher_output:
            logit_loss = self.logit_loss(
                student_output['cls'], 
                teacher_output['cls'],
                hard_targets
            )
            total_loss += self.lambda_logit * logit_loss
            loss_dict['logit_distill'] = logit_loss.item()
        
        # Feature distillation
        if student_features is not None and teacher_features is not None:
            feat_loss = self.feature_loss(student_features, teacher_features)
            total_loss += self.lambda_feat * feat_loss
            loss_dict['feat_distill'] = feat_loss.item()
            
            # Attention transfer
            att_loss = self.attention_loss(student_features, teacher_features)
            total_loss += self.lambda_att * att_loss
            loss_dict['att_distill'] = att_loss.item()
        
        # BBox distillation
        if 'box' in student_output and 'box' in teacher_output:
            bbox_loss = self.bbox_loss(
                student_output['box'],
                teacher_output['box'],
                fg_mask
            )
            total_loss += self.lambda_bbox * bbox_loss
            loss_dict['bbox_distill'] = bbox_loss.item()
        
        loss_dict['total_distill'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        
        return total_loss, loss_dict


class YOLOTeacher:
    """
    Wrapper for YOLO model as teacher
    Handles inference and feature extraction
    """
    
    def __init__(self, model_name='yolo11s.pt', device='cuda'):
        from ultralytics import YOLO
        
        self.model = YOLO(model_name)
        self.device = device
        self.model.to(device)
        
        # Set to eval mode
        self.model.model.eval()
        for param in self.model.model.parameters():
            param.requires_grad = False
    
    def get_predictions(self, images, conf_thresh=0.01):
        """
        Get teacher predictions for distillation
        
        Args:
            images: Tensor [B, 3, H, W]
        
        Returns:
            predictions: List of detection results
        """
        with torch.no_grad():
            results = self.model.predict(images, conf=conf_thresh, verbose=False)
        return results
    
    def get_features_and_predictions(self, images):
        """
        Get both intermediate features and final predictions
        
        Args:
            images: Tensor [B, 3, H, W]
        
        Returns:
            features: List of feature maps
            predictions: Detection predictions
        """
        features = []
        
        def hook_fn(module, input, output):
            features.append(output)
        
        # Register hooks on backbone layers
        hooks = []
        backbone = self.model.model.model
        
        # Hook layers 4, 6, 9 for YOLOv8/11 (P3, P4, P5)
        hook_indices = [4, 6, 9]
        for idx in hook_indices:
            if idx < len(backbone):
                hook = backbone[idx].register_forward_hook(hook_fn)
                hooks.append(hook)
        
        with torch.no_grad():
            predictions = self.model.predict(images, conf=0.01, verbose=False)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return features, predictions


def create_distillation_loss(student_model, teacher_model='yolo11s.pt', device='cuda'):
    """
    Factory function to create distillation loss and teacher
    
    Args:
        student_model: RoadSignNet-SAL model
        teacher_model: Path to teacher YOLO model
        device: Device to use
    
    Returns:
        distill_loss: KnowledgeDistillationLoss module
        teacher: YOLOTeacher wrapper
    """
    # Get student channel sizes (from EFP output)
    student_channels = [128, 128, 128]  # EFP outputs 128 channels at all scales
    
    # YOLO teacher channel sizes (YOLOv8n/11n: 64, 128, 256)
    teacher_channels = [64, 128, 256]
    
    distill_loss = KnowledgeDistillationLoss(
        student_channels=student_channels,
        teacher_channels=teacher_channels,
        temperature=3.0,
        lambda_feat=0.5,
        lambda_logit=1.0,
        lambda_att=0.5,
        lambda_bbox=0.5
    )
    
    teacher = YOLOTeacher(teacher_model, device)
    
    return distill_loss, teacher
