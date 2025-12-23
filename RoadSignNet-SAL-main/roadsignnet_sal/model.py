"""
RoadSignNet-SAL: Complete Model
"""

import torch
import torch.nn as nn
from .modules import (
    StemBlock, ACBStage, EfficientFeaturePyramid,
    LightweightDetectionHead
)


class RoadSignNetSAL(nn.Module):
    """RoadSignNet-SAL: Novel Architecture (~2.1M parameters)"""
    def __init__(self, num_classes=50, width_multiplier=1.0):
        super().__init__()
        
        self.num_classes = num_classes
        # Reduced channel sizes for ~2.1M parameters (34% lighter than YOLOv8n)
        base_channels = [16, 32, 64, 128, 256]
        channels = [int(c * width_multiplier) for c in base_channels]
        neck_channels = 128  # Reduced from 256
        
        # Backbone
        self.stem = StemBlock(3, channels[0])
        self.stage1 = ACBStage(channels[0], channels[1], 2, 2)
        self.stage2 = ACBStage(channels[1], channels[2], 2, 2)
        self.stage3 = ACBStage(channels[2], channels[3], 2, 2)
        self.stage4 = ACBStage(channels[3], channels[4], 2, 2)
        
        # Neck
        self.neck = EfficientFeaturePyramid([channels[2], channels[3], channels[4]], neck_channels)
        
        # Head
        self.head_p3 = LightweightDetectionHead(neck_channels, num_classes)
        self.head_p4 = LightweightDetectionHead(neck_channels, num_classes)
        self.head_p5 = LightweightDetectionHead(neck_channels, num_classes)
        
        self._initialize_weights()
        
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        p2 = self.stage2(x)
        p3 = self.stage3(p2)
        p4 = self.stage4(p3)
        
        features = self.neck([p2, p3, p4])
        
        return [
            self.head_p3(features[0]),
            self.head_p4(features[1]),
            self.head_p5(features[2])
        ]
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def create_roadsignnet_sal(num_classes=50, width_multiplier=1.0):
    return RoadSignNetSAL(num_classes, width_multiplier)