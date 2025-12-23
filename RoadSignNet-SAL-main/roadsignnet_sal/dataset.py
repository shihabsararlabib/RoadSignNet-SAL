import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocess import PREPROCESSOR
import cv2


class RoadSignDataset(Dataset):
    """Road Sign Dataset with Complete Preprocessing"""
    
    def __init__(self, img_dir: str, label_dir: str, img_size: int = 640, 
                 augment: bool = True, split: str = 'train'):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.augment = augment
        self.split = split
        
        # Get all image files
        self.img_files = list(self.img_dir.glob('*.jpg')) + \
                        list(self.img_dir.glob('*.png')) + \
                        list(self.img_dir.glob('*.jpeg')) + \
                        list(self.img_dir.glob('*.JPG')) + \
                        list(self.img_dir.glob('*.PNG'))
        
        # Use preprocessor transforms
        if augment and split == 'train':
            self.transform = PREPROCESSOR.training_transform()
        elif split == 'val' or split == 'test':
            self.transform = PREPROCESSOR.validation_transform()
        else:
            self.transform = PREPROCESSOR.validation_transform()
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        image = cv2.imread(str(img_path))
        
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        
        h, w = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load labels
        label_path = self.label_dir / f"{img_path.stem}.txt"
        bboxes = []
        class_labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        # YOLO format: center_x, center_y, width, height (normalized)
                        x_center, y_center, width, height = map(float, parts[1:5])
                        
                        # Convert to Pascal VOC format (x1, y1, x2, y2)
                        x_min = (x_center - width / 2) * w
                        y_min = (y_center - height / 2) * h
                        x_max = (x_center + width / 2) * w
                        y_max = (y_center + height / 2) * h
                        
                        # Clip to image boundaries
                        x_min = max(0, min(w - 1, x_min))
                        y_min = max(0, min(h - 1, y_min))
                        x_max = max(0, min(w, x_max))
                        y_max = max(0, min(h, y_max))
                        
                        # Skip invalid boxes
                        if x_max <= x_min or y_max <= y_min:
                            continue
                        
                        bboxes.append([x_min, y_min, x_max, y_max])
                        class_labels.append(class_id)
        
        # Apply preprocessing + augmentation
        # Always pass bboxes and class_labels (even if empty) since transform has bbox_params
        transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
        image = transformed['image']
        bboxes = torch.tensor(transformed['bboxes'], dtype=torch.float32) if len(transformed['bboxes']) > 0 else torch.zeros((0, 4))
        class_labels = torch.tensor(transformed['class_labels'], dtype=torch.long) if len(transformed['class_labels']) > 0 else torch.zeros(0, dtype=torch.long)
        
        return image, bboxes, class_labels


def create_dataloader(img_dir: str, label_dir: str, batch_size: int = 16, 
                     img_size: int = 640, augment: bool = True, 
                     num_workers: int = 4, shuffle: bool = True, 
                     split: str = 'train') -> DataLoader:
    """Create dataloader with complete preprocessing"""
    dataset = RoadSignDataset(img_dir, label_dir, img_size, augment, split)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )


def collate_fn(batch):
    """Custom collate function for variable-length bounding boxes"""
    images, bboxes, labels = zip(*batch)
    images = torch.stack(images)
    
    # Pad bboxes and labels to same length
    max_boxes = max(len(b) for b in bboxes) if len(bboxes) > 0 else 1
    padded_bboxes = torch.zeros(len(batch), max_boxes, 4)
    padded_labels = torch.full((len(batch), max_boxes), -1, dtype=torch.long)
    
    for i, (bb, lbl) in enumerate(zip(bboxes, labels)):
        if len(bb) > 0:
            padded_bboxes[i, :len(bb)] = bb
            padded_labels[i, :len(lbl)] = lbl
    
    return images, padded_bboxes, padded_labels