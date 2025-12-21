#!/usr/bin/env python3
"""
Train RoadSignNet-SAL Optimized - Memory Efficient Version
Fixed memory allocation issues that occur after 60+ epochs
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


def clear_memory():
    """Aggressively clear memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch with memory management"""
    model.train()
    total_loss = 0
    cls_loss_sum = 0
    box_loss_sum = 0
    obj_loss_sum = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
    for batch_idx, (images, bboxes, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        bboxes = bboxes.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # More memory efficient
        
        # Forward
        predictions = model(images)
        loss, loss_dict = criterion(predictions, None, bboxes, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Accumulate losses (detach to avoid memory leak)
        total_loss += loss.detach().item()
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
            
            # Clear tensors
            del images, bboxes, labels, predictions, loss, loss_dict
            
            # Clear every 50 batches
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
    parser = argparse.ArgumentParser(description='Train RoadSignNet-SAL Optimized (Memory Efficient)')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--width', type=float, default=1.35)
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (default: 16)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override batch size
    batch_size = args.batch_size
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Enable memory efficient settings
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    print("="*70)
    print("ROADSIGNNET-SAL OPTIMIZED TRAINING (Memory Efficient)")
    print("="*70)
    print(f"Device: {device}")
    print(f"Batch Size: {batch_size}")
    print(f"Width Multiplier: {args.width}")
    
    # Create model
    num_classes = config['model']['num_classes']
    model = create_roadsignnet_optimized(
        num_classes=num_classes,
        width_multiplier=args.width
    ).to(device)
    
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
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=config['training']['optimizer']['lr'] * 0.01
    )
    
    # Data loaders - MINIMAL WORKERS TO SAVE MEMORY
    img_size = config['data']['img_size']
    
    train_loader = create_dataloader(
        img_dir=config['data']['train_img_dir'],
        label_dir=config['data']['train_label_dir'],
        batch_size=batch_size,
        img_size=img_size,
        augment=False,  # Disable augmentation to save memory
        num_workers=0,  # No multiprocessing
        shuffle=True
    )
    
    val_loader = create_dataloader(
        img_dir=config['data']['val_img_dir'],
        label_dir=config['data']['val_label_dir'],
        batch_size=batch_size,
        img_size=img_size,
        augment=False,
        num_workers=0,
        shuffle=False
    )
    
    print(f"\n✓ Training samples: {len(train_loader.dataset)}")
    print(f"✓ Validation samples: {len(val_loader.dataset)}")
    print(f"✓ Batch size: {batch_size}")
    print(f"✓ Total epochs: {config['training']['epochs']}")
    
    # Resume handling
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"✓ Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        
        # Adjust scheduler
        for _ in range(start_epoch):
            scheduler.step()
        
        print(f"✓ Starting from epoch {start_epoch + 1}\n")
        
        # Clear checkpoint from memory
        del checkpoint
        clear_memory()
    
    # Training loop
    checkpoint_dir = Path('outputs/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        print("-" * 70)
        
        # Clear memory before epoch
        clear_memory()
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Clear memory after training
        clear_memory()
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Clear memory after validation
        clear_memory()
        
        # Update LR
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print metrics
        print(f"\nTrain Loss: {train_metrics['total_loss']:.4f} "
              f"(cls: {train_metrics['cls_loss']:.4f}, "
              f"box: {train_metrics['box_loss']:.4f}, "
              f"obj: {train_metrics['obj_loss']:.4f})")
        print(f"Val Loss:   {val_metrics['total_loss']:.4f} "
              f"(cls: {val_metrics['cls_loss']:.4f}, "
              f"box: {val_metrics['box_loss']:.4f}, "
              f"obj: {val_metrics['obj_loss']:.4f})")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Print memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            cached = torch.cuda.memory_reserved(device) / 1024**3
            print(f"GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'width_multiplier': args.width,
            }, checkpoint_dir / 'best_model_optimized.pth')
            print(f"✓ Saved best model (val_loss: {best_val_loss:.4f})")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['total_loss'],
                'width_multiplier': args.width,
            }, checkpoint_dir / f'checkpoint_optimized_epoch_{epoch+1}.pth')
            print(f"✓ Saved checkpoint: checkpoint_optimized_epoch_{epoch+1}.pth")
        
        # Force memory cleanup at end of each epoch
        clear_memory()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: outputs/checkpoints/best_model_optimized.pth")


if __name__ == '__main__':
    main()
