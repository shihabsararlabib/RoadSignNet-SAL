#!/usr/bin/env python3
"""
Train RoadSignNet-SAL Optimized V3
- Early stopping to prevent overfitting  
- Data augmentation enabled for better generalization
- Memory-efficient with aggressive cleanup
- Based on train_optimized_v2.py structure
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
import json

from roadsignnet_sal.model_optimized import create_roadsignnet_optimized
from roadsignnet_sal.loss import RoadSignNetLoss
from roadsignnet_sal.dataset import create_dataloader


def clear_memory():
    """Aggressively clear memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    def __init__(self, patience=15, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: {self.counter}/{self.patience} (best: {self.best_loss:.4f} at epoch {self.best_epoch})")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            
        return self.early_stop


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch with memory management"""
    model.train()
    total_loss = 0
    cls_loss_sum = 0
    box_loss_sum = 0
    obj_loss_sum = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (images, bboxes, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        bboxes = bboxes.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Forward
        predictions = model(images)
        loss, loss_dict = criterion(predictions, None, bboxes, labels)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        
        # Accumulate losses
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
        
        # Clear intermediate tensors
        del images, bboxes, labels, predictions, loss, loss_dict
        
        # Clear memory every 100 batches
        if batch_idx % 100 == 0:
            clear_memory()
    
    n = len(dataloader)
    return {
        'total_loss': total_loss / n,
        'cls_loss': cls_loss_sum / n,
        'box_loss': box_loss_sum / n,
        'obj_loss': obj_loss_sum / n
    }


def validate(model, dataloader, criterion, device):
    """Validate model with memory management"""
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
    parser = argparse.ArgumentParser(description='Train RoadSignNet-SAL Optimized V3')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--width', type=float, default=1.35)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--augment', action='store_true', default=True, help='Enable augmentation')
    parser.add_argument('--no-augment', action='store_false', dest='augment')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    print("=" * 70)
    print("ROADSIGNNET-SAL OPTIMIZED V3 TRAINING")
    print("With Early Stopping & Data Augmentation")
    print("=" * 70)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Width Multiplier: {args.width}")
    print(f"Augmentation: {'Enabled' if args.augment else 'Disabled'}")
    print(f"Early Stopping Patience: {args.patience} epochs")
    print(f"Max Epochs: {args.epochs}")
    
    # Create model
    num_classes = config['model']['num_classes']
    model = create_roadsignnet_optimized(
        num_classes=num_classes,
        width_multiplier=args.width
    ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Loss function
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
    
    # Scheduler - adjusted for potentially shorter training
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=config['training']['optimizer']['lr'] * 0.01
    )
    
    # Data loaders
    img_size = config['data']['img_size']
    
    train_loader = create_dataloader(
        img_dir=config['data']['train_img_dir'],
        label_dir=config['data']['train_label_dir'],
        batch_size=args.batch_size,
        img_size=img_size,
        augment=args.augment,  # Enable augmentation!
        num_workers=0,
        shuffle=True,
        split='train'
    )
    
    val_loader = create_dataloader(
        img_dir=config['data']['val_img_dir'],
        label_dir=config['data']['val_label_dir'],
        batch_size=args.batch_size,
        img_size=img_size,
        augment=False,
        num_workers=0,
        shuffle=False,
        split='val'
    )
    
    print(f"\nTraining samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001)
    
    # Resume from checkpoint
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
        print(f"Resuming from epoch {start_epoch}")
    
    # Training history
    history = []
    
    # Create output dirs
    os.makedirs('outputs/checkpoints', exist_ok=True)
    os.makedirs('outputs/logs', exist_ok=True)
    
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 70)
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Print metrics
        print(f"\nTrain Loss: {train_metrics['total_loss']:.4f} (cls: {train_metrics['cls_loss']:.4f}, box: {train_metrics['box_loss']:.4f}, obj: {train_metrics['obj_loss']:.4f})")
        print(f"Val Loss:   {val_metrics['total_loss']:.4f} (cls: {val_metrics['cls_loss']:.4f}, box: {val_metrics['box_loss']:.4f}, obj: {val_metrics['obj_loss']:.4f})")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated, {torch.cuda.memory_reserved()/1e9:.2f}GB cached")
        
        # Save history
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
            'lr': scheduler.get_last_lr()[0]
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
            }, 'outputs/checkpoints/best_model_optimized_v3.pth')
            print(f"✓ New best model saved! (val_loss: {best_val_loss:.4f})")
        
        # Save checkpoints: every 10 epochs + more frequently in optimal range (45-75)
        save_checkpoint = False
        if epoch % 10 == 0:
            save_checkpoint = True
        elif 45 <= epoch <= 75 and epoch % 5 == 0:
            save_checkpoint = True
        
        if save_checkpoint:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'num_classes': num_classes,
                'width_multiplier': args.width,
                'model_type': 'optimized'
            }, f'outputs/checkpoints/checkpoint_optimized_v3_epoch_{epoch}.pth')
            print(f"✓ Saved checkpoint: checkpoint_optimized_v3_epoch_{epoch}.pth")
        
        # Save history
        with open('outputs/logs/optimized_v3_training.json', 'w') as f:
            for h in history:
                f.write(json.dumps(h) + '\n')
        
        # Early stopping check
        if early_stopping(val_metrics['total_loss'], epoch):
            print(f"\n{'='*70}")
            print(f"EARLY STOPPING triggered at epoch {epoch}")
            print(f"Best validation loss: {early_stopping.best_loss:.4f} at epoch {early_stopping.best_epoch}")
            print(f"{'='*70}")
            break
        
        # Memory cleanup
        clear_memory()
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: outputs/checkpoints/best_model_optimized_v3.pth")
    print(f"\nTo evaluate: python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model_optimized_v3.pth")


if __name__ == '__main__':
    main()
