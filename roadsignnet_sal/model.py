"""
RoadSignNet-SAL: Complete Model with Transfer Learning Support
Supports: YOLOv8n, MobileNetV3, EfficientNet-B0, ResNet18 pretrained backbones
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import (
    MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights,
    EfficientNet_B0_Weights, ResNet18_Weights, DenseNet121_Weights
)
from .modules import (
    StemBlock, ACBStage, EfficientFeaturePyramid,
    LightweightDetectionHead, LightweightDetectionHeadKAN, KANConv1x1
)

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    import timm
    TIMM_AVAILABLE = True
except Exception:
    TIMM_AVAILABLE = False

class FeaturePyramidNeck(nn.Module):
    """Feature Pyramid Network for multi-scale detection"""
    def __init__(self, in_channels_list, out_channels=128):
        super().__init__()
        self.out_channels = out_channels
        
        # Lateral connections (1x1 conv to reduce channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        
        # Output convolutions (3x3 conv for smoothing)
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True)
            ) for _ in in_channels_list
        ])
        
    def forward(self, features):
        # features: [P3, P4, P5] from small to large stride
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        
        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i-1] = laterals[i-1] + F.interpolate(
                laterals[i], size=laterals[i-1].shape[-2:], mode='nearest'
            )
        
        # Output convolutions
        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        return outputs


class DetectionHead(nn.Module):
    """Detection head for each FPN level - outputs (cls, box, obj) tuple"""
    def __init__(self, in_channels, num_classes, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Shared stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
        
        # Separate heads for cls, box, obj (same format as LightweightDetectionHead)
        self.cls_head = nn.Conv2d(in_channels, num_anchors * num_classes, 1)
        self.box_head = nn.Conv2d(in_channels, num_anchors * 4, 1)
        self.obj_head = nn.Conv2d(in_channels, num_anchors * 1, 1)
        
    def forward(self, x):
        x = self.stem(x)
        return self.cls_head(x), self.box_head(x), self.obj_head(x)


class DetectionHeadKAN(nn.Module):
    """Detection head with KAN-based classification branch."""
    def __init__(self, in_channels, num_classes, num_anchors=3, kan_grid=8):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )

        self.cls_head = KANConv1x1(in_channels, num_anchors * num_classes, grid_size=kan_grid)
        self.box_head = nn.Conv2d(in_channels, num_anchors * 4, 1)
        self.obj_head = nn.Conv2d(in_channels, num_anchors * 1, 1)

    def forward(self, x):
        x = self.stem(x)
        return self.cls_head(x), self.box_head(x), self.obj_head(x)


class RoadSignNetTransfer(nn.Module):
    """RoadSignNet with Transfer Learning - Pretrained Backbone"""
    
    # Verified feature channels from actual model inspection
    BACKBONES = {
        'yolov8n': {
            'type': 'yolo',
            # YOLOv8n backbone channels: 64, 128, 256 (for P3, P4, P5)
            'feature_channels': [64, 128, 256],
            'model_name': 'yolov8n.pt',
        },
        'mobilenet_v3_small': {
            'type': 'torchvision',
            'model': models.mobilenet_v3_small,
            'weights': MobileNet_V3_Small_Weights.IMAGENET1K_V1,
            # Layer 3: 24ch (80x80), Layer 8: 48ch (40x40), Layer 11: 96ch (20x20)
            'feature_channels': [24, 48, 96],
            'feature_indices': [3, 8, 11],
        },
        'mobilenet_v3_large': {
            'type': 'torchvision',
            'model': models.mobilenet_v3_large,
            'weights': MobileNet_V3_Large_Weights.IMAGENET1K_V1,
            # Layer 6: 40ch (80x80), Layer 12: 112ch (40x40), Layer 15: 160ch (20x20)
            'feature_channels': [40, 112, 160],
            'feature_indices': [6, 12, 15],
        },
        'efficientnet_b0': {
            'type': 'torchvision',
            'model': models.efficientnet_b0,
            'weights': EfficientNet_B0_Weights.IMAGENET1K_V1,
            # Layer 3: 40ch (80x80), Layer 5: 112ch (40x40), Layer 7: 320ch (20x20)
            'feature_channels': [40, 112, 320],
            'feature_indices': [3, 5, 7],
        },
        'resnet18': {
            'type': 'torchvision',
            'model': models.resnet18,
            'weights': ResNet18_Weights.IMAGENET1K_V1,
            'feature_channels': [128, 256, 512],
            'feature_indices': [5, 6, 7],
        },
        'densenet121': {
            'type': 'torchvision_densenet',
            'model': models.densenet121,
            'weights': DenseNet121_Weights.IMAGENET1K_V1,
            # Use outputs after transition1/2/3 for strides 8/16/32
            'feature_channels': [128, 256, 512],
        },
    }
    
    def __init__(self, num_classes=43, backbone='mobilenet_v3_small', 
                 pretrained=True, freeze_backbone=False, neck_channels=128,
                 use_kan_cls=False, kan_grid=8):
        super().__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        # Support hybrid backbone strings like 'efficientnet_b0+vit_tiny_patch16_224'
        self.hybrid = '+' in backbone
        if not self.hybrid:
            if backbone not in self.BACKBONES:
                raise ValueError(f"Backbone {backbone} not supported. Choose from: {list(self.BACKBONES.keys())} or use 'cnn+vit' hybrid notation")
            backbone_config = self.BACKBONES[backbone]
            self.backbone_type = backbone_config.get('type', 'torchvision')

            # Load pretrained backbone (torchvision / YOLO)
            if self.backbone_type == 'yolo':
                if not ULTRALYTICS_AVAILABLE:
                    raise ImportError("ultralytics package required for YOLOv8. Install with: pip install ultralytics")
                if pretrained:
                    print(f"✓ Loading pretrained YOLOv8n weights...")
                    yolo_model = YOLO(backbone_config['model_name'])
                    self.backbone = yolo_model.model.model[:10]
                else:
                    raise ValueError("YOLOv8 requires pretrained weights")
            else:
                if pretrained:
                    print(f"✓ Loading pretrained {backbone} weights from ImageNet...")
                    self.backbone = backbone_config['model'](weights=backbone_config['weights'])
                else:
                    self.backbone = backbone_config['model'](weights=None)

            self.feature_channels = backbone_config['feature_channels']
            self.feature_indices = backbone_config.get('feature_indices', None)
        else:
            # Hybrid: allow multiple CNNs + one timm transformer
            parts = [p.strip() for p in backbone.split('+') if p.strip()]
            cnn_names = [p for p in parts if p in self.BACKBONES]
            timm_names = [p for p in parts if p not in self.BACKBONES]

            if len(cnn_names) < 1:
                raise ValueError("Hybrid backbone requires at least one CNN backbone")
            if len(timm_names) != 1:
                raise ValueError("Hybrid backbone requires exactly one timm transformer name")
            if not TIMM_AVAILABLE:
                raise ImportError("timm is required for transformer backbones. Install with: pip install timm")

            timm_name = timm_names[0]

            # Load CNN parts
            self.cnn_backbones = nn.ModuleList()
            self.cnn_feature_indices = []
            self.cnn_feature_channels = []
            self.cnn_names = cnn_names

            for cnn_name in cnn_names:
                cnn_config = self.BACKBONES[cnn_name]
                if pretrained:
                    print(f"✓ Loading pretrained {cnn_name} weights from ImageNet...")
                    cnn_model = cnn_config['model'](weights=cnn_config.get('weights', None))
                else:
                    cnn_model = cnn_config['model'](weights=None)

                self.cnn_backbones.append(cnn_model)
                self.cnn_feature_indices.append(cnn_config.get('feature_indices', None))
                self.cnn_feature_channels.append(cnn_config.get('feature_channels', [64, 128, 256]))

            # Load timm model (features_only)
            print(f"✓ Loading timm backbone: {timm_name} (features_only)")
            timm_model = timm.create_model(timm_name, pretrained=pretrained, features_only=True, out_indices=(1,2,3))
            self.timm_backbone = timm_model
            self.timm_feature_channels = [f['num_chs'] for f in timm_model.feature_info.info]

            # Align feature scales across CNNs and timm
            min_cnn_scales = min(len(ch) for ch in self.cnn_feature_channels)
            self.hybrid_num_scales = min(len(self.timm_feature_channels), min_cnn_scales)

            # Save names
            self.hybrid_cnn_names = cnn_names
            self.hybrid_timm_name = timm_name
        
        # Freeze backbone if requested
        if freeze_backbone:
            print("✓ Freezing backbone weights (only training neck and head)...")
            if self.hybrid:
                for cnn in getattr(self, 'cnn_backbones', []):
                    for param in cnn.parameters():
                        param.requires_grad = False
                for param in self.timm_backbone.parameters():
                    param.requires_grad = False
            else:
                for param in self.backbone.parameters():
                    param.requires_grad = False
        
        # Feature Pyramid Neck (for hybrid we will create a projection to neck_channels)
        if not self.hybrid:
            self.neck = FeaturePyramidNeck(self.feature_channels, neck_channels)
        else:
            # Create simple projection convs for fused features (one per scale)
            fused_in_channels = []
            for scale_idx in range(self.hybrid_num_scales):
                total = self.timm_feature_channels[scale_idx]
                for cnn_ch in self.cnn_feature_channels:
                    total += cnn_ch[scale_idx]
                fused_in_channels.append(total)

            self.fuse_projs = nn.ModuleList([
                nn.Conv2d(in_ch, neck_channels, 1, bias=False)
                for in_ch in fused_in_channels
            ])
            self.neck = FeaturePyramidNeck([neck_channels, neck_channels, neck_channels], neck_channels)
        
        # Detection Heads for 3 scales
        head_cls = DetectionHeadKAN if use_kan_cls else DetectionHead
        self.head_p3 = head_cls(neck_channels, num_classes, kan_grid=kan_grid) if use_kan_cls else head_cls(neck_channels, num_classes)
        self.head_p4 = head_cls(neck_channels, num_classes, kan_grid=kan_grid) if use_kan_cls else head_cls(neck_channels, num_classes)
        self.head_p5 = head_cls(neck_channels, num_classes, kan_grid=kan_grid) if use_kan_cls else head_cls(neck_channels, num_classes)
        
        self._initialize_new_layers()
    
    def _extract_features_yolo(self, x):
        """Extract multi-scale features from YOLOv8 backbone"""
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            # YOLOv8n backbone outputs at layers 4, 6, 9 for P3, P4, P5
            if i in [4, 6, 9]:
                features.append(x)
        return features
        
    def _extract_features_mobilenet(self, x):
        """Extract multi-scale features from MobileNetV3"""
        features = []
        for i, layer in enumerate(self.backbone.features):
            x = layer(x)
            if i in self.feature_indices:
                features.append(x)
        return features
    
    def _extract_features_efficientnet(self, x):
        """Extract multi-scale features from EfficientNet"""
        features = []
        for i, layer in enumerate(self.backbone.features):
            x = layer(x)
            if i in self.feature_indices:
                features.append(x)
        return features
    
    def _extract_features_resnet(self, x, backbone=None):
        """Extract multi-scale features from ResNet"""
        backbone = backbone or self.backbone
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)
        
        x = backbone.layer1(x)
        x = backbone.layer2(x)
        p3 = x  # stride 8
        x = backbone.layer3(x)
        p4 = x  # stride 16
        x = backbone.layer4(x)
        p5 = x  # stride 32
        
        return [p3, p4, p5]

    def _extract_features_densenet(self, x, backbone=None):
        """Extract multi-scale features from DenseNet-121"""
        backbone = backbone or self.backbone
        f = backbone.features
        x = f.conv0(x)
        x = f.norm0(x)
        x = f.relu0(x)
        x = f.pool0(x)

        x = f.denseblock1(x)
        x = f.transition1(x)
        p3 = x  # stride 8

        x = f.denseblock2(x)
        x = f.transition2(x)
        p4 = x  # stride 16

        x = f.denseblock3(x)
        x = f.transition3(x)
        p5 = x  # stride 32

        return [p3, p4, p5]

    def _extract_features_timm(self, x):
        """Extract multi-scale features from a timm features_only model"""
        # timm features_only models return list of feature maps
        return self.timm_backbone(x)

    def _extract_cnn_features(self, x, cnn_name, cnn_model, feature_indices):
        """Extract multi-scale features from a CNN backbone instance."""
        if 'mobilenet' in cnn_name:
            features = []
            xx = x
            for i, layer in enumerate(cnn_model.features):
                xx = layer(xx)
                if i in feature_indices:
                    features.append(xx)
            return features

        if 'efficientnet' in cnn_name:
            features = []
            xx = x
            for i, layer in enumerate(cnn_model.features):
                xx = layer(xx)
                if i in feature_indices:
                    features.append(xx)
            return features

        if 'densenet' in cnn_name:
            return self._extract_features_densenet(x, backbone=cnn_model)

        if 'resnet' in cnn_name:
            return self._extract_features_resnet(x, backbone=cnn_model)

        raise ValueError(f"Unsupported CNN backbone: {cnn_name}")

    def _extract_features_hybrid(self, x):
        """Extract features from CNN and timm backbones and fuse them"""
        # Extract CNN features for each backbone
        cnn_features_list = []
        for cnn_name, cnn_model, feat_idx in zip(self.cnn_names, self.cnn_backbones, self.cnn_feature_indices):
            cnn_features_list.append(self._extract_cnn_features(x, cnn_name, cnn_model, feat_idx))

        # Extract timm features (list)
        feat_timm = self._extract_features_timm(x)

        num_scales = getattr(self, 'hybrid_num_scales', len(self.fuse_projs))
        for cnn_feats in cnn_features_list:
            if len(cnn_feats) < num_scales:
                raise ValueError("CNN backbone produced fewer feature maps than expected")
        if len(feat_timm) < num_scales:
            raise ValueError("timm backbone produced fewer feature maps than expected")

        # Fuse corresponding scales by concatenation and projection
        fused = []
        for scale_idx, proj in enumerate(self.fuse_projs[:num_scales]):
            feats = [cnn_feats[scale_idx] for cnn_feats in cnn_features_list]
            feats.append(feat_timm[scale_idx])

            target_size = feats[0].shape[-2:]
            aligned = []
            for f in feats:
                if f.shape[-2:] != target_size:
                    f = F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
                aligned.append(f)

            cat = torch.cat(aligned, dim=1)
            fused.append(proj(cat))

        return fused
    
    def forward(self, x):
        # Extract backbone features (support hybrid)
        if not self.hybrid:
            if self.backbone_type == 'yolo':
                features = self._extract_features_yolo(x)
            elif 'mobilenet' in self.backbone_name:
                features = self._extract_features_mobilenet(x)
            elif 'efficientnet' in self.backbone_name:
                features = self._extract_features_efficientnet(x)
            elif 'resnet' in self.backbone_name:
                features = self._extract_features_resnet(x)
            elif 'densenet' in self.backbone_name:
                features = self._extract_features_densenet(x)
            else:
                raise ValueError(f"Unknown backbone: {self.backbone_name}")
        else:
            features = self._extract_features_hybrid(x)
        
        # FPN neck
        fpn_features = self.neck(features)
        
        # Detection heads
        return [
            self.head_p3(fpn_features[0]),
            self.head_p4(fpn_features[1]),
            self.head_p5(fpn_features[2]),
        ]
    
    def _initialize_new_layers(self):
        """Initialize neck and head layers"""
        for module in [self.neck, self.head_p3, self.head_p4, self.head_p5]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


class RoadSignNetSAL(nn.Module):
    """RoadSignNet-SAL: Original Novel Architecture (~2.1M parameters)"""
    def __init__(self, num_classes=50, width_multiplier=1.0, use_kan_cls=False, kan_grid=8):
        super().__init__()
        
        self.num_classes = num_classes
        base_channels = [16, 32, 64, 128, 256]
        channels = [int(c * width_multiplier) for c in base_channels]
        neck_channels = 128
        
        # Backbone
        self.stem = StemBlock(3, channels[0])
        self.stage1 = ACBStage(channels[0], channels[1], 2, 2)
        self.stage2 = ACBStage(channels[1], channels[2], 2, 2)
        self.stage3 = ACBStage(channels[2], channels[3], 2, 2)
        self.stage4 = ACBStage(channels[3], channels[4], 2, 2)
        
        # Neck
        self.neck = EfficientFeaturePyramid([channels[2], channels[3], channels[4]], neck_channels)
        
        # Head
        head_cls = LightweightDetectionHeadKAN if use_kan_cls else LightweightDetectionHead
        if use_kan_cls:
            self.head_p3 = head_cls(neck_channels, num_classes, kan_grid=kan_grid)
            self.head_p4 = head_cls(neck_channels, num_classes, kan_grid=kan_grid)
            self.head_p5 = head_cls(neck_channels, num_classes, kan_grid=kan_grid)
        else:
            self.head_p3 = head_cls(neck_channels, num_classes)
            self.head_p4 = head_cls(neck_channels, num_classes)
            self.head_p5 = head_cls(neck_channels, num_classes)
        
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


def create_roadsignnet_sal(num_classes=50, width_multiplier=1.0, use_kan_cls=False, kan_grid=8):
    """Create original RoadSignNet-SAL (no pretrained weights)"""
    return RoadSignNetSAL(num_classes, width_multiplier, use_kan_cls=use_kan_cls, kan_grid=kan_grid)


def create_roadsignnet_transfer(num_classes=43, backbone='mobilenet_v3_small', 
                                 pretrained=True, freeze_backbone=False,
                                 use_kan_cls=False, kan_grid=8):
    """
    Create RoadSignNet with Transfer Learning
    
    Args:
        num_classes: Number of detection classes
        backbone: Backbone name or hybrid string (e.g. 'efficientnet_b0+vit_tiny_patch16_224')
        pretrained: Load ImageNet pretrained weights
        freeze_backbone: Freeze backbone (only train neck/head)
    
    Returns:
        RoadSignNetTransfer model
    """
    return RoadSignNetTransfer(
        num_classes, backbone, pretrained, freeze_backbone,
        use_kan_cls=use_kan_cls, kan_grid=kan_grid
    )