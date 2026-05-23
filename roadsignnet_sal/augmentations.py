"""
RoadSignNet-SAL: Advanced Augmentations
Mosaic, MixUp, Copy-Paste for improved training
Inspired by YOLO augmentation pipeline
"""

import torch
import numpy as np
import cv2
import random
from pathlib import Path


class MosaicAugmentation:
    """
    Mosaic Augmentation (from YOLOv4/v5)
    Combines 4 images into one, improving:
    - Multi-scale object detection
    - Batch normalization statistics
    - Context understanding
    """
    
    def __init__(self, img_size=640, mosaic_prob=0.5):
        self.img_size = img_size
        self.mosaic_prob = mosaic_prob
        
    def __call__(self, images, all_bboxes, all_labels):
        """
        Apply mosaic augmentation to a batch of images
        
        Args:
            images: List of numpy arrays [H, W, 3]
            all_bboxes: List of numpy arrays [N, 4] in x1y1x2y2 format (pixel coords)
            all_labels: List of numpy arrays [N]
            
        Returns:
            mosaic_img: [img_size, img_size, 3]
            mosaic_bboxes: [M, 4] combined bboxes
            mosaic_labels: [M] combined labels
        """
        if random.random() > self.mosaic_prob or len(images) < 4:
            # Return first image without mosaic
            return images[0], all_bboxes[0], all_labels[0]
        
        s = self.img_size
        # Random center point
        xc = int(random.uniform(s * 0.25, s * 0.75))
        yc = int(random.uniform(s * 0.25, s * 0.75))
        
        # Initialize mosaic image
        mosaic_img = np.full((s, s, 3), 114, dtype=np.uint8)
        
        combined_bboxes = []
        combined_labels = []
        
        # Select 4 images (with replacement if needed)
        indices = random.choices(range(len(images)), k=4)
        
        for i, idx in enumerate(indices):
            img = images[idx]
            bboxes = all_bboxes[idx].copy() if len(all_bboxes[idx]) > 0 else np.zeros((0, 4))
            labels = all_labels[idx].copy() if len(all_labels[idx]) > 0 else np.zeros(0)
            
            h, w = img.shape[:2]
            
            # Place image in mosaic
            if i == 0:  # Top-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # Top-right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # Bottom-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # Bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s), min(s, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            
            # Copy image region
            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            
            # Adjust bboxes
            padw = x1a - x1b
            padh = y1a - y1b
            
            if len(bboxes) > 0:
                # Scale bboxes to mosaic coordinates
                scale_x = (x2a - x1a) / (x2b - x1b) if (x2b - x1b) > 0 else 1
                scale_y = (y2a - y1a) / (y2b - y1b) if (y2b - y1b) > 0 else 1
                
                bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * scale_x + padw
                bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * scale_y + padh
                
                # Clip to mosaic boundaries
                bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, s)
                bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, s)
                
                # Filter valid boxes (min size)
                valid = (bboxes[:, 2] - bboxes[:, 0] > 2) & (bboxes[:, 3] - bboxes[:, 1] > 2)
                bboxes = bboxes[valid]
                labels = labels[valid]
                
                combined_bboxes.append(bboxes)
                combined_labels.append(labels)
        
        # Combine all bboxes and labels
        if combined_bboxes:
            mosaic_bboxes = np.concatenate(combined_bboxes, axis=0)
            mosaic_labels = np.concatenate(combined_labels, axis=0)
        else:
            mosaic_bboxes = np.zeros((0, 4))
            mosaic_labels = np.zeros(0)
        
        return mosaic_img, mosaic_bboxes, mosaic_labels


class MixUpAugmentation:
    """
    MixUp Augmentation
    Blends two images and their labels for regularization
    """
    
    def __init__(self, alpha=0.5, mixup_prob=0.3):
        self.alpha = alpha
        self.mixup_prob = mixup_prob
    
    def __call__(self, img1, bboxes1, labels1, img2, bboxes2, labels2):
        """
        Apply mixup between two images
        
        Returns:
            mixed_img, combined_bboxes, combined_labels
        """
        if random.random() > self.mixup_prob:
            return img1, bboxes1, labels1
        
        # Beta distribution for mixing ratio
        lam = np.random.beta(self.alpha, self.alpha)
        lam = max(lam, 1 - lam)  # Ensure dominant image
        
        # Resize img2 to match img1
        h, w = img1.shape[:2]
        img2_resized = cv2.resize(img2, (w, h))
        
        # Mix images
        mixed_img = (img1 * lam + img2_resized * (1 - lam)).astype(np.uint8)
        
        # Combine bboxes (keep all from both images)
        combined_bboxes = np.concatenate([bboxes1, bboxes2], axis=0) if len(bboxes2) > 0 else bboxes1
        combined_labels = np.concatenate([labels1, labels2], axis=0) if len(labels2) > 0 else labels1
        
        return mixed_img, combined_bboxes, combined_labels


class CopyPasteAugmentation:
    """
    Copy-Paste Augmentation
    Copies objects from one image to another for data augmentation
    Especially useful for rare classes
    """
    
    def __init__(self, paste_prob=0.3, max_paste=3):
        self.paste_prob = paste_prob
        self.max_paste = max_paste
    
    def __call__(self, target_img, target_bboxes, target_labels, 
                 source_img, source_bboxes, source_labels):
        """
        Paste objects from source image to target image
        """
        if random.random() > self.paste_prob or len(source_bboxes) == 0:
            return target_img, target_bboxes, target_labels
        
        result_img = target_img.copy()
        result_bboxes = list(target_bboxes)
        result_labels = list(target_labels)
        
        # Select random objects to paste
        num_paste = min(self.max_paste, len(source_bboxes))
        paste_indices = random.sample(range(len(source_bboxes)), num_paste)
        
        h, w = target_img.shape[:2]
        
        for idx in paste_indices:
            bbox = source_bboxes[idx].astype(int)
            label = source_labels[idx]
            
            # Extract object region
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(source_img.shape[1], x2), min(source_img.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            obj_region = source_img[y1:y2, x1:x2].copy()
            obj_h, obj_w = obj_region.shape[:2]
            
            if obj_h < 5 or obj_w < 5:
                continue
            
            # Random scale
            scale = random.uniform(0.5, 1.5)
            new_w = int(obj_w * scale)
            new_h = int(obj_h * scale)
            
            if new_w < 5 or new_h < 5 or new_w > w or new_h > h:
                continue
            
            obj_region = cv2.resize(obj_region, (new_w, new_h))
            
            # Random position
            paste_x = random.randint(0, max(0, w - new_w))
            paste_y = random.randint(0, max(0, h - new_h))
            
            # Paste object
            result_img[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = obj_region
            
            # Add bbox
            new_bbox = np.array([paste_x, paste_y, paste_x + new_w, paste_y + new_h])
            result_bboxes.append(new_bbox)
            result_labels.append(label)
        
        return result_img, np.array(result_bboxes), np.array(result_labels)


class AdvancedAugmentationPipeline:
    """
    Complete augmentation pipeline combining Mosaic, MixUp, and other augmentations
    """
    
    def __init__(self, img_size=640, mosaic_prob=0.5, mixup_prob=0.15, 
                 hsv_prob=0.5, flip_prob=0.5):
        self.img_size = img_size
        self.mosaic = MosaicAugmentation(img_size, mosaic_prob)
        self.mixup = MixUpAugmentation(mixup_prob=mixup_prob)
        self.hsv_prob = hsv_prob
        self.flip_prob = flip_prob
    
    def hsv_augment(self, img, hgain=0.015, sgain=0.7, vgain=0.4):
        """HSV color-space augmentation"""
        if random.random() > self.hsv_prob:
            return img
        
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
        
        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(np.uint8)
        lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)
        
        img_hsv = cv2.merge([
            cv2.LUT(hue, lut_hue),
            cv2.LUT(sat, lut_sat),
            cv2.LUT(val, lut_val)
        ]).astype(np.uint8)
        
        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    
    def horizontal_flip(self, img, bboxes):
        """Horizontal flip augmentation"""
        if random.random() > self.flip_prob:
            return img, bboxes
        
        img = img[:, ::-1, :]
        w = img.shape[1]
        
        if len(bboxes) > 0:
            bboxes = bboxes.copy()
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]
        
        return img, bboxes
    
    def __call__(self, images, all_bboxes, all_labels):
        """
        Apply full augmentation pipeline
        
        Args:
            images: List of images [H, W, 3]
            all_bboxes: List of bbox arrays [N, 4]
            all_labels: List of label arrays [N]
        
        Returns:
            Augmented image, bboxes, labels
        """
        # 1. Mosaic augmentation
        img, bboxes, labels = self.mosaic(images, all_bboxes, all_labels)
        
        # 2. MixUp (if we have more images)
        if len(images) >= 2:
            idx2 = random.randint(0, len(images) - 1)
            img, bboxes, labels = self.mixup(
                img, bboxes, labels,
                images[idx2], all_bboxes[idx2], all_labels[idx2]
            )
        
        # 3. HSV augmentation
        img = self.hsv_augment(img)
        
        # 4. Horizontal flip
        img, bboxes = self.horizontal_flip(img, bboxes)
        
        # 5. Resize to target size
        h, w = img.shape[:2]
        if h != self.img_size or w != self.img_size:
            scale_x = self.img_size / w
            scale_y = self.img_size / h
            img = cv2.resize(img, (self.img_size, self.img_size))
            if len(bboxes) > 0:
                bboxes[:, [0, 2]] *= scale_x
                bboxes[:, [1, 3]] *= scale_y
        
        return img, bboxes, labels
