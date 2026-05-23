"""
RoadSignNet-SAL: Novel Lightweight Architecture for Road Sign Detection

Novel Contributions:
1. Asymmetric Convolutional Block (ACB) - 33% parameter reduction
2. Road Sign Attention Module (RSAM) - Domain-specific attention
3. Spatial Position Bias - Emphasizes upper image regions
4. Context-Aware Module - Global-local feature interaction
5. Small Object Enhancer - Multi-receptive field fusion
6. Adversarial Weather Module - Robust to weather variations
7. Efficient Feature Pyramid (EFP) - Bidirectional FPN with attention
8. Knowledge Distillation - Transfer from YOLO teachers
"""

__version__ = "2.0.0"
__author__ = "thesis-2025-team"

from .model import RoadSignNetSAL, RoadSignNetTransfer
from .model_optimized import RoadSignNetOptimized, create_roadsignnet_optimized
from .model_v2 import RoadSignNetV2
from .model_v3 import RoadSignNetV3
from .loss import RoadSignNetLoss
from .loss_v2 import AnchorFreeLoss, AnchorFreeDecoder
from .loss_v3 import V3Loss
from .dataset import RoadSignDataset, create_dataloader
from .modules import AsymmetricConvBlock, RoadSignAttentionModule, EfficientFeaturePyramid, KANLinear, KANConv1x1
from .attention_modules import (
    SpatialAttentionModule, 
    ContextAwareModule, 
    SmallObjectEnhancer,
    AdversarialWeatherModule,
    EnhancedDetectionHead
)
from .augmentations import MosaicAugmentation, MixUpAugmentation, AdvancedAugmentationPipeline

__all__ = [
    # Models
    'RoadSignNetSAL',
    'RoadSignNetTransfer',
    'RoadSignNetOptimized',
    'create_roadsignnet_optimized',
    'RoadSignNetV2',
    'RoadSignNetV3',
    
    # Loss Functions
    'RoadSignNetLoss',
    'AnchorFreeLoss',
    'AnchorFreeDecoder',
    'V3Loss',
    
    # Dataset
    'RoadSignDataset',
    'create_dataloader',
    
    # Novel Modules
    'AsymmetricConvBlock',
    'RoadSignAttentionModule',
    'EfficientFeaturePyramid',
    'KANLinear',
    'KANConv1x1',
    'SpatialAttentionModule',
    'ContextAwareModule',
    'SmallObjectEnhancer',
    'AdversarialWeatherModule',
    'EnhancedDetectionHead',
    
    # Augmentations
    'MosaicAugmentation',
    'MixUpAugmentation',
    'AdvancedAugmentationPipeline',
]
