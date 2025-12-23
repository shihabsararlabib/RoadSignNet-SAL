#!/usr/bin/env python3
"""
Train RoadSignNet-SAL Optimized
Enhanced V1 with novel attention mechanisms and improved feature pyramid
Target: ~2M parameters, >55% mAP (baseline V1: 50.71% mAP)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import argparse
from pathlib import Path
import gc

from roadsignnet_sal.model_optimized import create_roadsignnet_optimized
from roadsignnet_sal.loss import RoadSignNetLoss
from roadsignnet_sal.dataset import create_dataloader
import json


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    cls_loss_sum = 0
    box_loss_sum = 0
    obj_loss_sum = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
    for batch_idx, (images, bboxes, labels) in enumerate(pbar):
        images = images.to(device)
        bboxes = bboxes.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        predictions = model(images)
        
        # Compute loss (note: criterion expects (predictions, None, gt_boxes, gt_labels))
        loss, loss_dict = criterion(predictions, None, bboxes, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        cls_loss_sum += loss_dict['cls_loss']
        box_loss_sum += loss_dict['box_loss']
        obj_loss_sum += loss_dict['obj_loss']
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cls': f'{loss_dict["cls_loss"]:.4f}',
            'box': f'{loss_dict["box_loss"]:.4f}',
            'obj': f'{loss_dict["obj_loss"]:.4f}'
        })
    
    n = len(dataloader)
    return {
        'total_loss': total_loss / n,
        'cls_loss': cls_loss_sum / n,
        'box_loss': box_loss_sum / n,
        'obj_loss': obj_loss_sum / n
    }


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    cls_loss_sum = 0
    box_loss_sum = 0
    obj_loss_sum = 0
    
    with torch.no_grad():
        for images, bboxes, labels in tqdm(dataloader, desc='Validating'):
            images = images.to(device)
            bboxes = bboxes.to(device)
            labels = labels.to(device)
            
            predictions = model(images)
            loss, loss_dict = criterion(predictions, None, bboxes, labels)
            
            total_loss += loss.item()
            cls_loss_sum += loss_dict['cls_loss']
            box_loss_sum += loss_dict['box_loss']
            obj_loss_sum += loss_dict['obj_loss']
    
    n = len(dataloader)
    return {
        'total_loss': total_loss / n,
        'cls_loss': cls_loss_sum / n,
        'box_loss': box_loss_sum / n,
        'obj_loss': obj_loss_sum / n
    }


def main():
    parser = argparse.ArgumentParser(description='Train RoadSignNet-SAL Optimized')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--width', type=float, default=1.35,
                       help='Width multiplier (1.35 for ~2M params)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("ROADSIGNNET-SAL OPTIMIZED TRAINING")
    print("="*70)
    print(f"Configuration: {args.config}")
    print(f"Device: {device}")
    print(f"Width Multiplier: {args.width}")
    print(f"Target: ~2M params, >55% mAP (Baseline: 50.71% mAP)")
    print("="*70)
    
    # Create model
    num_classes = config['model']['num_classes']
    model = create_roadsignnet_optimized(num_classes=num_classes, width_multiplier=args.width)
    model = model.to(device)
    
    # Loss function (same as V1 baseline)
    criterion = RoadSignNetLoss(
        num_classes=num_classes,
        lambda_cls=config['training']['loss']['lambda_cls'],
        lambda_box=config['training']['loss']['lambda_box'],
        lambda_obj=config['training']['loss']['lambda_obj']
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['optimizer']['lr'],
        weight_decay=config['training']['optimizer']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=config['training']['optimizer']['lr'] * 0.01
    )
    
    # Data loaders
    img_size = config['data']['img_size']
    batch_size = config['training']['batch_size']
    num_workers = 0  # Reduce to avoid memory accumulation
    augment = config['training']['augmentation']['enable']
    
    train_loader = create_dataloader(
        img_dir='data/train/images',
        label_dir='data/train/labels',
        img_size=img_size,
        batch_size=batch_size,
        augment=augment,
        num_workers=num_workers,
        shuffle=True,
        split='train'
    )
    
    val_loader = create_dataloader(
        img_dir='data/valid/images',
        label_dir='data/valid/labels',
        img_size=img_size,
        batch_size=batch_size,
        augment=False,
        num_workers=num_workers,
        shuffle=False,
        split='valid'
    )
    
    print(f"\n✓ Training samples: {len(train_loader.dataset)}")
    print(f"✓ Validation samples: {len(val_loader.dataset)}")
    print(f"✓ Batch size: {config['training']['batch_size']}")
    print(f"✓ Total epochs: {config['training']['epochs']}\n")
    
    # Create log directory
    Path('outputs/logs').mkdir(parents=True, exist_ok=True)
    log_file = open('outputs/logs/optimized_training.json', 'a')
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"✓ Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"✓ Starting from epoch {start_epoch+1}\n")
    
    # Training loop
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        print("-" * 70)
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Clear memory after each epoch
        torch.cuda.empty_cache()
        gc.collect()
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        print(f"\nTrain Loss: {train_metrics['total_loss']:.4f} "
              f"(cls: {train_metrics['cls_loss']:.4f}, "
              f"box: {train_metrics['box_loss']:.4f}, "
              f"obj: {train_metrics['obj_loss']:.4f})")
        print(f"Val Loss:   {val_metrics['total_loss']:.4f} "
              f"(cls: {val_metrics['cls_loss']:.4f}, "
              f"box: {val_metrics['box_loss']:.4f}, "
              f"obj: {val_metrics['obj_loss']:.4f})")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Log metrics to JSON
        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['total_loss'],
            'train_cls_loss': train_metrics['cls_loss'],
            'train_box_loss': train_metrics['box_loss'],
            'train_obj_loss': train_metrics['obj_loss'],
            'val_loss': val_metrics['total_loss'],
            'val_cls_loss': val_metrics['cls_loss'],
            'val_box_loss': val_metrics['box_loss'],
            'val_obj_loss': val_metrics['obj_loss'],
            'lr': current_lr
        }
        log_file.write(json.dumps(log_entry) + '\n')
        log_file.flush()
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_metrics['total_loss'],
            'val_loss': val_metrics['total_loss'],
            'config': config
        }
        
        # Save periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'outputs/checkpoints/checkpoint_optimized_epoch_{epoch+1}.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            best_checkpoint_path = 'outputs/checkpoints/best_model_optimized.pth'
            torch.save(checkpoint, best_checkpoint_path)
            print(f"✓ New best model saved! Val loss: {best_val_loss:.4f}")
    
    log_file.close()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved: outputs/checkpoints/best_model_optimized.pth")
    print("="*70)


if __name__ == '__main__':
    main()
