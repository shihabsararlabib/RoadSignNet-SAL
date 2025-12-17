"""
Novel Attention Mechanisms for Road Sign Detection
Research Contributions:
1. Spatial Attention for Small Object Enhancement
2. Context-Aware Road Region Focus
3. Multi-Scale Feature Refinement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttentionModule(nn.Module):
    """
    CONTRIBUTION 1: Spatial Attention for Small Sign Detection
    - Emphasizes likely sign regions (upper road area)
    - Enhances small/distant object features
    - Adaptive receptive field
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.conv_spatial = nn.Conv2d(in_channels // 4, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()
        
        # Learnable position bias (signs typically in upper 2/3 of image)
        self.position_bias = nn.Parameter(torch.zeros(1, 1, 1, 1))
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            attended: [B, C, H, W] with spatial attention
        """
        B, C, H, W = x.shape
        
        # Spatial attention map
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv_spatial(attention)
        
        # Add position bias (prefer upper region)
        h_coords = torch.linspace(0, 1, H, device=x.device).view(1, 1, H, 1)
        position_weight = torch.exp(-3 * h_coords) + self.position_bias  # Exponential decay from top
        attention = attention + position_weight
        
        attention_map = self.sigmoid(attention)
        
        # Apply attention
        return x * attention_map


class ContextAwareModule(nn.Module):
    """
    CONTRIBUTION 2: Context-Aware Road Region Detection
    - Segments road/sky/background implicitly
    - Suppresses false positives in irrelevant regions
    - Uses global context for local decisions
    """
    def __init__(self, in_channels):
        super().__init__()
        # Global context branch
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.context_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, in_channels),
            nn.Sigmoid()
        )
        
        # Local feature refinement
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            context_features: [B, C, H, W] with context awareness
        """
        B, C, H, W = x.shape
        
        # Global context (road vs non-road understanding)
        global_context = self.global_pool(x).view(B, C)
        context_weights = self.context_fc(global_context).view(B, C, 1, 1)
        
        # Apply global context to local features
        context_modulated = x * context_weights
        
        # Local refinement
        refined = self.local_conv(context_modulated)
        
        return refined + x  # Residual connection


class SmallObjectEnhancer(nn.Module):
    """
    CONTRIBUTION 3: Small Object Feature Enhancement
    - Specifically designed for detecting distant signs (<32px)
    - High-resolution feature preservation
    - Multi-receptive field fusion
    """
    def __init__(self, in_channels):
        super().__init__()
        mid_channels = in_channels // 2
        
        # Multi-scale receptive fields (3x3, 5x5, 7x7)
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        
        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 5, padding=2),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        
        self.branch_7x7 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 7, padding=3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(mid_channels * 3, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        
        # High-frequency detail preservation (for small signs)
        self.detail_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            enhanced: [B, C, H, W] with enhanced small object features
        """
        # Multi-scale features
        feat_3x3 = self.branch_3x3(x)
        feat_5x5 = self.branch_5x5(x)
        feat_7x7 = self.branch_7x7(x)
        
        # Fuse multi-scale
        multi_scale = torch.cat([feat_3x3, feat_5x5, feat_7x7], dim=1)
        fused = self.fusion(multi_scale)
        
        # Add high-frequency details
        details = self.detail_branch(x)
        
        return fused + details


class AdversarialWeatherModule(nn.Module):
    """
    CONTRIBUTION 4: Adversarial Weather Robustness
    - Adaptive feature normalization for varying conditions
    - Learns to handle rain, fog, night, glare
    - Style-invariant feature learning
    """
    def __init__(self, in_channels):
        super().__init__()
        # Style encoder (estimates weather condition)
        self.style_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Adaptive normalization parameters
        self.gamma_fc = nn.Linear(in_channels // 4, in_channels)
        self.beta_fc = nn.Linear(in_channels // 4, in_channels)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            normalized: [B, C, H, W] weather-invariant features
        """
        B, C, H, W = x.shape
        
        # Estimate style/condition
        style = self.style_encoder(x).view(B, -1)
        gamma = self.gamma_fc(style).view(B, C, 1, 1)
        beta = self.beta_fc(style).view(B, C, 1, 1)
        
        # Instance normalization
        mean = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        normalized = (x - mean) / (var + 1e-5).sqrt()
        
        # Adaptive denormalization
        return gamma * normalized + beta


class EnhancedDetectionHead(nn.Module):
    """
    ENHANCED: Detection head with all novel contributions integrated
    """
    def __init__(self, in_channels, num_classes, hidden_channels=128):
        super().__init__()
        self.num_classes = num_classes
        
        # Novel contributions
        self.spatial_attention = SpatialAttentionModule(in_channels)
        self.context_aware = ContextAwareModule(in_channels)
        self.small_object_enhancer = SmallObjectEnhancer(in_channels)
        self.weather_robust = AdversarialWeatherModule(in_channels)
        
        # Feature reduction
        self.feature_reduce = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU()
        )
        
        # Detection heads (same as before)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        self.box_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 4, 1)
        )
        
        self.class_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)
        )
        
    def forward(self, x):
        """
        Args:
            x: Feature map [B, C, H, W]
        Returns:
            heatmap, boxes, classes with enhanced features
        """
        # Apply novel contributions sequentially
        x = self.spatial_attention(x)        # Focus on likely sign regions
        x = self.context_aware(x)            # Use road context
        x = self.small_object_enhancer(x)    # Enhance small signs
        x = self.weather_robust(x)           # Weather invariance
        
        # Reduce channels
        x = self.feature_reduce(x)
        
        # Detection outputs
        heatmap = self.heatmap_head(x)
        boxes = self.box_head(x)
        classes = self.class_head(x)
        
        return heatmap, boxes, classes
