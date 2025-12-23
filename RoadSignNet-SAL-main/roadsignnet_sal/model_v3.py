"""
RoadSignNet-SAL V3: YOLO-Inspired Ultra-Lightweight Architecture
Based on YOLOv8n/v11n with improvements and fixes for known issues

Key Features:
- CSPDarknet-Nano backbone (proven in YOLO)
- Improved PAN-FPN neck (better multi-scale fusion)
- Decoupled detection head (separate cls/box/obj branches)
- Anchor-free detection (like YOLOv8+)
- ~2M parameters target
- Fixes: coordinate handling, loss balancing, feature alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    """Standard convolution with BN and SiLU activation"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Lightweight bottleneck block"""
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.conv2 = Conv(hidden_channels, out_channels, 3, 1)
        self.shortcut = shortcut and in_channels == out_channels
    
    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.shortcut else self.conv2(self.conv1(x))


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions (from YOLOv8)"""
    def __init__(self, in_channels, out_channels, n=1, shortcut=False, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, 2 * hidden_channels, 1, 1)
        self.conv2 = Conv((2 + n) * hidden_channels, out_channels, 1, 1)
        self.bottlenecks = nn.ModuleList(
            Bottleneck(hidden_channels, hidden_channels, shortcut, expansion=1.0) for _ in range(n)
        )
    
    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(block(y[-1]) for block in self.bottlenecks)
        return self.conv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (from YOLOv5/v8)"""
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.conv2 = Conv(hidden_channels * 4, out_channels, 1, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    
    def forward(self, x):
        x = self.conv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], 1))


class DWConv(nn.Module):
    """Depthwise separable convolution (more efficient)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pw = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x


class RoadSignNetV3(nn.Module):
    """
    Ultra-lightweight YOLO-inspired architecture for road sign detection
    Target: ~2M parameters
    
    Architecture:
    - Backbone: CSPDarknet-Nano (4 stages)
    - Neck: Improved PAN-FPN (feature pyramid with proper fusion)
    - Head: Decoupled anchor-free detection head
    """
    
    def __init__(self, num_classes=43, width_mult=0.25, depth_mult=0.33):
        super().__init__()
        self.num_classes = num_classes
        
        # Calculate channel sizes (scaled by width_mult)
        def make_divisible(x, divisor=8):
            return max(int(x + divisor / 2) // divisor * divisor, divisor)
        
        base_channels = [64, 128, 256, 512, 1024]
        channels = [make_divisible(c * width_mult) for c in base_channels]
        
        # Calculate depth (number of bottlenecks per stage)
        def get_depth(n):
            return max(round(n * depth_mult), 1)
        
        # =================================================================
        # BACKBONE: CSPDarknet-Nano
        # =================================================================
        # Input: 3x416x416
        
        # Stem
        self.stem = Conv(3, channels[0], 3, 2)  # 416 -> 208
        
        # Stage 1: 208 -> 104
        self.stage1_conv = Conv(channels[0], channels[1], 3, 2)
        self.stage1_c2f = C2f(channels[1], channels[1], get_depth(3), shortcut=True)
        
        # Stage 2: 104 -> 52
        self.stage2_conv = Conv(channels[1], channels[2], 3, 2)
        self.stage2_c2f = C2f(channels[2], channels[2], get_depth(6), shortcut=True)
        
        # Stage 3: 52 -> 26
        self.stage3_conv = Conv(channels[2], channels[3], 3, 2)
        self.stage3_c2f = C2f(channels[3], channels[3], get_depth(6), shortcut=True)
        
        # Stage 4: 26 -> 13
        self.stage4_conv = Conv(channels[3], channels[4], 3, 2)
        self.stage4_c2f = C2f(channels[4], channels[4], get_depth(3), shortcut=True)
        self.sppf = SPPF(channels[4], channels[4], 5)
        
        # =================================================================
        # NECK: Improved PAN-FPN
        # =================================================================
        # Feature pyramid sizes: P3 (52x52), P4 (26x26), P5 (13x13)
        
        # Top-down pathway
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.neck_conv1 = Conv(channels[4], channels[3], 1, 1)
        self.neck_c2f1 = C2f(channels[3] * 2, channels[3], get_depth(3), shortcut=False)
        
        self.neck_conv2 = Conv(channels[3], channels[2], 1, 1)
        self.neck_c2f2 = C2f(channels[2] * 2, channels[2], get_depth(3), shortcut=False)
        
        # Bottom-up pathway
        self.neck_conv3 = Conv(channels[2], channels[3], 3, 2)
        self.neck_c2f3 = C2f(channels[3] * 2, channels[3], get_depth(3), shortcut=False)
        
        self.neck_conv4 = Conv(channels[3], channels[4], 3, 2)
        self.neck_c2f4 = C2f(channels[4] * 2, channels[4], get_depth(3), shortcut=False)
        
        # =================================================================
        # HEAD: Decoupled Anchor-Free Detection
        # =================================================================
        # 3 detection scales: P3 (52x52), P4 (26x26), P5 (13x13)
        
        self.heads = nn.ModuleList([
            DecoupledHead(channels[2], num_classes, 256),  # P3
            DecoupledHead(channels[3], num_classes, 256),  # P4
            DecoupledHead(channels[4], num_classes, 256),  # P5
        ])
        
        # Store feature channels for loss computation
        self.feature_channels = [channels[2], channels[3], channels[4]]
        self.strides = [8, 16, 32]  # P3, P4, P5
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights (important for convergence)"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        Returns: List of predictions at 3 scales [(cls, box), (cls, box), (cls, box)]
        """
        # Backbone
        x = self.stem(x)
        
        # Stage 1
        x1 = self.stage1_conv(x)
        x1 = self.stage1_c2f(x1)  # Not used in neck (too early)
        
        # Stage 2 - P3 (52x52)
        x2 = self.stage2_conv(x1)
        p3 = self.stage2_c2f(x2)
        
        # Stage 3 - P4 (26x26)
        x3 = self.stage3_conv(p3)
        p4 = self.stage3_c2f(x3)
        
        # Stage 4 - P5 (13x13)
        x4 = self.stage4_conv(p4)
        x4 = self.stage4_c2f(x4)
        p5 = self.sppf(x4)
        
        # Neck - Top-down pathway
        p5_up = self.upsample(self.neck_conv1(p5))
        p4_fused = torch.cat([p5_up, p4], 1)
        p4_out = self.neck_c2f1(p4_fused)
        
        p4_up = self.upsample(self.neck_conv2(p4_out))
        p3_fused = torch.cat([p4_up, p3], 1)
        p3_out = self.neck_c2f2(p3_fused)
        
        # Neck - Bottom-up pathway
        p3_down = self.neck_conv3(p3_out)
        p4_fused2 = torch.cat([p3_down, p4_out], 1)
        p4_out2 = self.neck_c2f3(p4_fused2)
        
        p4_down = self.neck_conv4(p4_out2)
        p5_fused = torch.cat([p4_down, p5], 1)
        p5_out = self.neck_c2f4(p5_fused)
        
        # Multi-scale detection heads
        outputs = [
            self.heads[0](p3_out),   # 52x52
            self.heads[1](p4_out2),  # 26x26
            self.heads[2](p5_out),   # 13x13
        ]
        
        return outputs


class DecoupledHead(nn.Module):
    """
    Decoupled detection head (separate branches for classification and regression)
    Inspired by YOLOX/YOLOv6/YOLOv8 - proven to improve performance
    """
    def __init__(self, in_channels, num_classes, reg_channels=256):
        super().__init__()
        self.num_classes = num_classes
        
        # Shared stem
        self.stem = Conv(in_channels, reg_channels, 1, 1)
        
        # Classification branch (2 convs)
        self.cls_convs = nn.Sequential(
            DWConv(reg_channels, reg_channels, 3, 1),
            DWConv(reg_channels, reg_channels, 3, 1),
        )
        self.cls_pred = nn.Conv2d(reg_channels, num_classes, 1)
        
        # Regression branch (2 convs)
        self.reg_convs = nn.Sequential(
            DWConv(reg_channels, reg_channels, 3, 1),
            DWConv(reg_channels, reg_channels, 3, 1),
        )
        self.reg_pred = nn.Conv2d(reg_channels, 4, 1)  # 4 values: cx, cy, w, h offsets
        
        # Objectness branch (1 conv)
        self.obj_pred = nn.Conv2d(reg_channels, 1, 1)
        
        self._initialize_biases()
    
    def _initialize_biases(self):
        """Initialize biases for better initial predictions"""
        # Classification bias initialization (reduces initial loss)
        nn.init.constant_(self.cls_pred.bias, -4.6)  # log(0.01 / (1 - 0.01))
        
        # Objectness bias initialization
        nn.init.constant_(self.obj_pred.bias, -4.6)
    
    def forward(self, x):
        """
        Returns:
            cls: [B, num_classes, H, W] - class logits
            reg: [B, 4, H, W] - box offsets (cx, cy, w, h)
            obj: [B, 1, H, W] - objectness score
        """
        x = self.stem(x)
        
        # Classification branch
        cls_feat = self.cls_convs(x)
        cls = self.cls_pred(cls_feat)
        
        # Regression branch
        reg_feat = self.reg_convs(x)
        reg = self.reg_pred(reg_feat)
        
        # Objectness branch
        obj = self.obj_pred(reg_feat)
        
        return cls, reg, obj


def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def create_roadsignnet_v3(num_classes=43, width_mult=0.25, depth_mult=0.33):
    """
    Factory function to create RoadSignNet-SAL V3
    
    Args:
        num_classes: Number of object classes
        width_mult: Width multiplier for channels (0.25 = nano, 0.5 = small, 1.0 = medium)
        depth_mult: Depth multiplier for layers (0.33 = nano, 0.67 = small, 1.0 = medium)
    
    Returns:
        model: RoadSignNetV3 instance
    """
    model = RoadSignNetV3(
        num_classes=num_classes,
        width_mult=width_mult,
        depth_mult=depth_mult
    )
    
    total_params, trainable_params = count_parameters(model)
    
    print(f"✓ Created RoadSignNetV3")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Width Multiplier: {width_mult}")
    print(f"  Depth Multiplier: {depth_mult}")
    
    return model


if __name__ == '__main__':
    # Test the model
    model = create_roadsignnet_v3(num_classes=43, width_mult=0.25, depth_mult=0.33)
    
    # Test forward pass
    x = torch.randn(1, 3, 416, 416)
    outputs = model(x)
    
    print(f"\n✓ Model test passed!")
    print(f"  Input: {x.shape}")
    print(f"  Outputs:")
    for i, (cls, reg, obj) in enumerate(outputs):
        print(f"    Scale {i+1}: cls={cls.shape}, reg={reg.shape}, obj={obj.shape}")
