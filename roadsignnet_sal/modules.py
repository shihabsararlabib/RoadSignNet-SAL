"""
RoadSignNet-SAL: Custom Modules
All architectural components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricConvBlock(nn.Module):
    """Asymmetric Convolutional Block - Novel Component"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        pad = kernel_size // 2
        
        self.conv_h = nn.Conv2d(in_channels, out_channels, (1, kernel_size), 
                                (1, stride), (0, pad), bias=False)
        self.conv_v = nn.Conv2d(out_channels, out_channels, (kernel_size, 1), 
                                (stride, 1), (pad, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        
    def forward(self, x):
        x = self.act(self.bn1(self.conv_h(x)))
        x = self.act(self.bn2(self.conv_v(x)))
        return x


class RoadSignAttentionModule(nn.Module):
    """Road Sign Attention Module - Novel Component"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Channel Attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att
        
        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_conv(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_att
        
        return x


class StemBlock(nn.Module):
    """Efficient stem block for initial feature extraction"""
    def __init__(self, in_channels=3, out_channels=32):
        super().__init__()
        self.conv1 = AsymmetricConvBlock(in_channels, out_channels, kernel_size=3, stride=2)
        self.conv2 = AsymmetricConvBlock(out_channels, out_channels, kernel_size=3, stride=1)
        
    def forward(self, x):
        return self.conv2(self.conv1(x))


class ACBStage(nn.Module):
    """Stage of stacked Asymmetric Convolutional Blocks"""
    def __init__(self, in_channels, out_channels, num_blocks=2, stride=2):
        super().__init__()
        layers = [AsymmetricConvBlock(in_channels, out_channels, kernel_size=3, stride=stride)]
        layers.extend([AsymmetricConvBlock(out_channels, out_channels, kernel_size=3, stride=1) 
                      for _ in range(num_blocks - 1)])
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)


class EfficientFeaturePyramid(nn.Module):
    """Efficient Feature Pyramid - Novel Component"""
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1, bias=False)
            for in_ch in in_channels_list
        ])
        
        self.fpn_convs = nn.ModuleList([
            AsymmetricConvBlock(out_channels, out_channels, kernel_size=3)
            for _ in in_channels_list
        ])
        
        self.attentions = nn.ModuleList([
            RoadSignAttentionModule(out_channels)
            for _ in in_channels_list
        ])
        
    def forward(self, features):
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i-1] += F.interpolate(laterals[i], size=laterals[i-1].shape[-2:], mode='nearest')
        
        outputs = [attention(fpn_conv(lateral)) 
                  for lateral, fpn_conv, attention in zip(laterals, self.fpn_convs, self.attentions)]
        
        return outputs


class LightweightDetectionHead(nn.Module):
    """Lightweight Detection Head - Novel Component"""
    def __init__(self, in_channels, num_classes=50, num_anchors=3):
        super().__init__()
        
        self.stem = nn.Sequential(
            AsymmetricConvBlock(in_channels, in_channels, kernel_size=3),
            AsymmetricConvBlock(in_channels, in_channels, kernel_size=3),
        )
        
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels // 2, num_anchors * num_classes, 1)
        )
        
        self.box_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels // 2, num_anchors * 4, 1)
        )
        
        self.obj_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels // 2, num_anchors, 1)
        )
        
    def forward(self, x):
        x = self.stem(x)
        return self.cls_head(x), self.box_head(x), self.obj_head(x)


class LightweightDetectionHeadKAN(nn.Module):
    """Lightweight Detection Head with KAN-based classification branch."""
    def __init__(self, in_channels, num_classes=50, num_anchors=3, kan_grid=8):
        super().__init__()

        self.stem = nn.Sequential(
            AsymmetricConvBlock(in_channels, in_channels, kernel_size=3),
            AsymmetricConvBlock(in_channels, in_channels, kernel_size=3),
        )

        self.cls_head = KANConv1x1(in_channels, num_anchors * num_classes, grid_size=kan_grid)

        self.box_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels // 2, num_anchors * 4, 1)
        )

        self.obj_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels // 2, num_anchors, 1)
        )

    def forward(self, x):
        x = self.stem(x)
        return self.cls_head(x), self.box_head(x), self.obj_head(x)


class KANLinear(nn.Module):
    """KAN-inspired linear layer using fixed triangular basis functions."""
    def __init__(self, in_features, out_features, grid_size=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size

        centers = torch.linspace(-1.0, 1.0, grid_size)
        self.register_buffer('centers', centers.view(1, 1, grid_size))
        self.delta = 2.0 / (grid_size - 1)

        self.weight = nn.Parameter(torch.randn(out_features, in_features, grid_size) * 0.02)
        self.linear_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # x: [B, in_features]
        x = torch.tanh(x)
        x_exp = x.unsqueeze(-1)
        basis = torch.relu(1.0 - torch.abs(x_exp - self.centers) / self.delta)
        out = torch.einsum('bik,oik->bo', basis, self.weight)
        out = out + x @ self.linear_weight.t() + self.bias
        return out


class KANConv1x1(nn.Module):
    """Apply KANLinear per spatial location like a 1x1 conv."""
    def __init__(self, in_channels, out_channels, grid_size=8):
        super().__init__()
        self.kan = KANLinear(in_channels, out_channels, grid_size=grid_size)

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, c)
        y = self.kan(x_flat)
        y = y.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return y