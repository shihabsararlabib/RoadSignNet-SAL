"""
RoadSignNet-SAL: Data Preprocessing and Augmentation
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch


class Preprocessor:
    """Data preprocessing and augmentation utilities"""
    
    def __init__(self, img_size=640):
        self.img_size = img_size
    
    def training_transform(self):
        """OPTIMIZED: Fast training augmentation pipeline (V4 - light)"""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Resize(self.img_size, self.img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.3))
    
    def training_transform_strong(self):
        """V5: STRONG augmentation pipeline to reduce overfitting"""
        return A.Compose([
            # Geometric augmentations
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,      # 10% shift
                scale_limit=0.15,     # 15% scale
                rotate_limit=10,      # ±10 degrees
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),
            A.Affine(
                shear=(-5, 5),        # slight shear
                p=0.3
            ),
            
            # Color augmentations
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
            ], p=0.7),
            
            # Noise/blur augmentations
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
                A.GaussNoise(var_limit=(10.0, 30.0), p=1.0),
            ], p=0.3),
            
            # Dropout augmentation (Cutout-like)
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=2,
                min_height=16,
                min_width=16,
                fill_value=0,
                p=0.4
            ),
            
            A.Resize(self.img_size, self.img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.3))
    
    def validation_transform(self):
        """Validation/test augmentation pipeline"""
        return A.Compose([
            A.PadIfNeeded(self.img_size, self.img_size, border_mode=cv2.BORDER_CONSTANT),
            A.Resize(self.img_size, self.img_size),
            A.ToFloat(max_value=255.0),  # Convert to float before normalize to avoid LUT issues
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=1.0  # Already converted to 0-1 range by ToFloat
            ),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    
    def inference_transform(self):
        """Inference augmentation (minimal)"""
        return A.Compose([
            A.PadIfNeeded(self.img_size, self.img_size, border_mode=cv2.BORDER_CONSTANT),
            A.Resize(self.img_size, self.img_size),
            A.ToFloat(max_value=255.0),  # Convert to float before normalize to avoid LUT issues
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=1.0  # Already converted to 0-1 range by ToFloat
            ),
            ToTensorV2(),
        ])
    
    @staticmethod
    def convert_labels_to_pascal_voc(center_x, center_y, width, height, img_width, img_height):
        """Convert YOLO format (center, width, height) to Pascal VOC format (x1, y1, x2, y2)"""
        x_min = (center_x - width / 2) * img_width
        y_min = (center_y - height / 2) * img_height
        x_max = (center_x + width / 2) * img_width
        y_max = (center_y + height / 2) * img_height
        return x_min, y_min, x_max, y_max
    
    @staticmethod
    def convert_labels_to_yolo(x_min, y_min, x_max, y_max, img_width, img_height):
        """Convert Pascal VOC format to YOLO format"""
        center_x = ((x_min + x_max) / 2) / img_width
        center_y = ((y_min + y_max) / 2) / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        return center_x, center_y, width, height


# Global preprocessor instance
PREPROCESSOR = Preprocessor(img_size=640)
