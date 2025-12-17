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
        """OPTIMIZED: Fast training augmentation pipeline"""
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
