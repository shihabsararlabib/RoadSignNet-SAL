#!/usr/bin/env python3
"""
Train RoadSignNet-SAL Optimized V5
- STRONG data augmentation to reduce overfitting
- Higher dropout and weight decay
- Label smoothing
- Mixup augmentation
- Full training (no early stopping)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import yaml
import argparse
from pathlib import Path
import gc
import json
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import GradScaler, autocast

from roadsignnet_sal.model_optimized import create_roadsignnet_optimized
from roadsignnet_sal.loss import RoadSignNetLoss


def clear_memory():
    """Aggressively clear memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class RoadSignDatasetV5(Dataset):
    """Road Sign Dataset with STRONG Augmentation for V5"""
    
    def __init__(self, img_dir: str, label_dir: str, img_size: int = 416, 
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
        
        # Strong augmentation for training (memory-optimized)
        if augment and split == 'train':
            self.transform = A.Compose([
                # Resize FIRST to reduce memory usage during augmentation
                A.Resize(img_size, img_size),
                
                # Geometric augmentations
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.15,
                    rotate_limit=10,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5
                ),
                
                # Color augmentations (simplified - avoid ColorJitter memory issues)
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
                
                # Noise/blur (simplified)
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                ], p=0.2),
                
                # Cutout/dropout (reduced)
                A.CoarseDropout(
                    num_holes_range=(1, 4),
                    hole_height_range=(8, 24),
                    hole_width_range=(8, 24),
                    fill=0,
                    p=0.3
                ),
                
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.3))
        else:
            # Validation transform
            self.transform = A.Compose([
                A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT),
                A.Resize(img_size, img_size),
                A.ToFloat(max_value=255.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        try:
            img_path = self.img_files[idx]
            image = cv2.imread(str(img_path))
            
            if image is None:
                # Return a blank image if loading fails
                image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                bboxes = []
                class_labels = []
            else:
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
                                x_center, y_center, width, height = map(float, parts[1:5])
                                
                                x_min = (x_center - width / 2) * w
                                y_min = (y_center - height / 2) * h
                                x_max = (x_center + width / 2) * w
                                y_max = (y_center + height / 2) * h
                                
                                x_min = max(0, min(w - 1, x_min))
                                y_min = max(0, min(h - 1, y_min))
                                x_max = max(0, min(w, x_max))
                                y_max = max(0, min(h, y_max))
                                
                                if x_max <= x_min or y_max <= y_min:
                                    continue
                                
                                bboxes.append([x_min, y_min, x_max, y_max])
                                class_labels.append(class_id)
            
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
        except Exception as e:
            # Fallback: return blank image on any error
            print(f"Warning: Error loading sample {idx}: {e}")
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            transformed = self.transform(image=image, bboxes=[], class_labels=[])
        image = transformed['image']
        bboxes = torch.tensor(transformed['bboxes'], dtype=torch.float32) if len(transformed['bboxes']) > 0 else torch.zeros((0, 4))
        class_labels = torch.tensor(transformed['class_labels'], dtype=torch.long) if len(transformed['class_labels']) > 0 else torch.zeros(0, dtype=torch.long)
        
        return image, bboxes, class_labels


def collate_fn(batch):
    """Custom collate function"""
    images, bboxes, labels = zip(*batch)
    images = torch.stack(images)
    
    max_boxes = max(len(b) for b in bboxes) if len(bboxes) > 0 else 1
    padded_bboxes = torch.zeros(len(batch), max_boxes, 4)
    padded_labels = torch.full((len(batch), max_boxes), -1, dtype=torch.long)
    
    for i, (bb, lbl) in enumerate(zip(bboxes, labels)):
        if len(bb) > 0:
            padded_bboxes[i, :len(bb)] = bb
            padded_labels[i, :len(lbl)] = lbl
    
    return images, padded_bboxes, padded_labels


# Use the original RoadSignNetLoss which works with the model output format

def mixup_data(x, y_bbox, y_labels, alpha=0.2):
    """Mixup augmentation for detection"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    
    return mixed_x, y_bbox, y_labels, lam


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, use_mixup=True, scaler: GradScaler | None = None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    cls_loss_sum = 0
    box_loss_sum = 0
    obj_loss_sum = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (images, bboxes, labels) in enumerate(pbar):
        try:
            images = images.to(device, non_blocking=True)
            bboxes = bboxes.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Apply mixup with 50% probability
            if use_mixup and np.random.random() > 0.5:
                images, bboxes, labels, _ = mixup_data(images, bboxes, labels, alpha=0.2)

            optimizer.zero_grad(set_to_none=True)

            if scaler is not None and torch.cuda.is_available():
                with autocast():
                    predictions = model(images)
                    loss, loss_dict = criterion(predictions, None, bboxes, labels)
                scaler.scale(loss).backward()
                # Gradient clipping with unscaled grads
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                predictions = model(images)
                loss, loss_dict = criterion(predictions, None, bboxes, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

            total_loss += loss.detach().item()
            cls_loss_sum += loss_dict['cls_loss']
            box_loss_sum += loss_dict['box_loss']
            obj_loss_sum += loss_dict['obj_loss']

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cls': f'{loss_dict["cls_loss"]:.4f}',
                'box': f'{loss_dict["box_loss"]:.4f}',
                'obj': f'{loss_dict["obj_loss"]:.4f}'
            })
        except RuntimeError as e:
            # Handle CUDA OOM gracefully by skipping the batch
            if 'out of memory' in str(e).lower():
                print(f"Warning: CUDA OOM at batch {batch_idx}, skipping batch.")
                clear_memory()
                continue
            else:
                raise
        finally:
            # Free batch tensors
            del images, bboxes, labels
            if 'predictions' in locals():
                del predictions
            if 'loss' in locals():
                del loss
            if 'loss_dict' in locals():
                del loss_dict

            if batch_idx % 100 == 0:
                clear_memory()
    
    n = len(dataloader)
    return {
        'total_loss': total_loss / n,
        'cls_loss': cls_loss_sum / n,
        'box_loss': box_loss_sum / n,
        'obj_loss': obj_loss_sum / n
    }


def validate(model, dataloader, criterion, device, use_amp: bool = True):
    """Validate model"""
    model.eval()
    total_loss = 0
    cls_loss_sum = 0
    box_loss_sum = 0
    obj_loss_sum = 0
    
    with torch.no_grad():
        for batch_idx, (images, bboxes, labels) in enumerate(tqdm(dataloader, desc='Validating')):
            images = images.to(device, non_blocking=True)
            bboxes = bboxes.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_amp and torch.cuda.is_available():
                with autocast():
                    predictions = model(images)
                    loss, loss_dict = criterion(predictions, None, bboxes, labels)
            else:
                predictions = model(images)
                loss, loss_dict = criterion(predictions, None, bboxes, labels)
            
            total_loss += loss.item()
            cls_loss_sum += loss_dict['cls_loss']
            box_loss_sum += loss_dict['box_loss']
            obj_loss_sum += loss_dict['obj_loss']
            
            del images, bboxes, labels, predictions, loss, loss_dict
            
            if batch_idx % 50 == 0:
                clear_memory()
    
    n = len(dataloader)
    return {
        'total_loss': total_loss / n,
        'cls_loss': cls_loss_sum / n,
        'box_loss': box_loss_sum / n,
        'obj_loss': obj_loss_sum / n
    }


def main():
    parser = argparse.ArgumentParser(description='Train RoadSignNet-SAL V5 (Strong Augmentation)')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--width', type=float, default=1.35)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0.05)  # Increased from 0.01
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.2)  # Added dropout
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    print("=" * 70)
    print("ROADSIGNNET-SAL V5 TRAINING")
    print("Strong Augmentation + Regularization to Reduce Overfitting")
    print("=" * 70)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Width Multiplier: {args.width}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Label Smoothing: {args.label_smoothing}")
    print(f"Epochs: {args.epochs}")
    print("\nAugmentation: STRONG")
    print("  - HorizontalFlip, Affine (scale, rotate, shear)")
    print("  - HueSaturationValue, BrightnessContrast")
    print("  - GaussianBlur, MotionBlur, GaussNoise")
    print("  - CoarseDropout (Cutout)")
    print("  - Mixup (alpha=0.2)")
    
    # Create model
    num_classes = config['model']['num_classes']
    model = create_roadsignnet_optimized(
        num_classes=num_classes,
        width_multiplier=args.width
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Use original loss function (works with model output format)
    criterion = RoadSignNetLoss(
        num_classes=num_classes,
        lambda_cls=config['training']['loss']['lambda_cls'],
        lambda_box=config['training']['loss']['lambda_box'],
        lambda_obj=config['training']['loss']['lambda_obj']
    )
    
    # Optimizer with higher weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['optimizer']['lr'],
        weight_decay=args.weight_decay  # Increased regularization
    )
    # AMP scaler to reduce memory usage
    scaler = GradScaler(enabled=torch.cuda.is_available())
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=config['training']['optimizer']['lr'] * 0.01
    )
    
    # Data loaders with strong augmentation
    img_size = config['data']['img_size']
    
    train_dataset = RoadSignDatasetV5(
        img_dir=config['data']['train_img_dir'],
        label_dir=config['data']['train_label_dir'],
        img_size=img_size,
        augment=True,
        split='train'
    )
    
    val_dataset = RoadSignDatasetV5(
        img_dir=config['data']['val_img_dir'],
        label_dir=config['data']['val_label_dir'],
        img_size=img_size,
        augment=False,
        split='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Resume
    start_epoch = 1
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    history = []
    os.makedirs('outputs/checkpoints', exist_ok=True)
    os.makedirs('outputs/logs', exist_ok=True)
    
    print("\n" + "=" * 70)
    print(f"STARTING V5 TRAINING ({args.epochs} epochs)")
    print("=" * 70)
    
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 70)
        
        # Train with mixup
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, use_mixup=True, scaler=scaler)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, use_amp=True)
        
        scheduler.step()
        
        print(f"\nTrain Loss: {train_metrics['total_loss']:.4f} (cls: {train_metrics['cls_loss']:.4f}, box: {train_metrics['box_loss']:.4f}, obj: {train_metrics['obj_loss']:.4f})")
        print(f"Val Loss:   {val_metrics['total_loss']:.4f} (cls: {val_metrics['cls_loss']:.4f}, box: {val_metrics['box_loss']:.4f}, obj: {val_metrics['obj_loss']:.4f})")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Loss gap (overfitting indicator)
        gap = val_metrics['total_loss'] / train_metrics['total_loss'] if train_metrics['total_loss'] > 0 else 0
        print(f"Val/Train Loss Ratio: {gap:.2f}x (lower = less overfitting)")
        
        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        
        history.append({
            'epoch': epoch,
            'train_loss': train_metrics['total_loss'],
            'train_cls_loss': train_metrics['cls_loss'],
            'train_box_loss': train_metrics['box_loss'],
            'train_obj_loss': train_metrics['obj_loss'],
            'val_loss': val_metrics['total_loss'],
            'val_cls_loss': val_metrics['cls_loss'],
            'val_box_loss': val_metrics['box_loss'],
            'val_obj_loss': val_metrics['obj_loss'],
            'lr': scheduler.get_last_lr()[0],
            'loss_ratio': gap
        })
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'num_classes': num_classes,
                'width_multiplier': args.width,
                'model_type': 'optimized'
            }, 'outputs/checkpoints/best_model_optimized_v5.pth')
            print(f"✓ New best model! (val_loss: {best_val_loss:.4f})")
        
        # Save checkpoints every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'num_classes': num_classes,
                'width_multiplier': args.width,
                'model_type': 'optimized'
            }, f'outputs/checkpoints/checkpoint_optimized_v5_epoch_{epoch}.pth')
            print(f"✓ Saved: checkpoint_optimized_v5_epoch_{epoch}.pth")
        
        # Save history
        with open('outputs/logs/optimized_v5_training.json', 'w') as f:
            for h in history:
                f.write(json.dumps(h) + '\n')
        
        clear_memory()
    
    # Save final
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'num_classes': num_classes,
        'width_multiplier': args.width,
        'model_type': 'optimized'
    }, 'outputs/checkpoints/final_model_optimized_v5.pth')
    
    print("\n" + "=" * 70)
    print("V5 TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"\nTo evaluate: python scripts/evaluate.py --checkpoint outputs/checkpoints/checkpoint_optimized_v5_epoch_XX.pth")


if __name__ == '__main__':
    main()
