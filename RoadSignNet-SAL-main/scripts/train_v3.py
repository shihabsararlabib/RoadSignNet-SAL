"""
Training script for RoadSignNet-SAL V3
YOLO-inspired ultra-lightweight architecture
"""

import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roadsignnet_sal.model_v3 import create_roadsignnet_v3
from roadsignnet_sal.loss_v3 import create_loss_v3
from roadsignnet_sal.dataset import RoadSignDataset


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_dataloaders(config):
    """Create train and validation dataloaders"""
    # Training dataset
    train_dataset = RoadSignDataset(
        img_dir='data/train/images',
        label_dir='data/train/labels',
        img_size=416,
        augment=True,
        split='train'
    )
    
    # Validation dataset
    val_dataset = RoadSignDataset(
        img_dir='data/valid/images',
        label_dir='data/valid/labels',
        img_size=416,
        augment=False,
        split='val'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        collate_fn=collate_fn,
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


def collate_fn(batch):
    """Custom collate function to handle variable number of objects"""
    images = []
    bboxes = []
    labels = []
    
    max_objects = max(len(item[1]) for item in batch)
    
    for img, bbox, label in batch:
        images.append(img)
        
        # Pad to max_objects
        if len(bbox) < max_objects:
            pad_size = max_objects - len(bbox)
            bbox = torch.cat([bbox, torch.full((pad_size, 4), -1.0)], dim=0)
            label = torch.cat([label, torch.full((pad_size,), -1, dtype=torch.long)], dim=0)
        
        bboxes.append(bbox)
        labels.append(label)
    
    images = torch.stack(images, dim=0)
    bboxes = torch.stack(bboxes, dim=0)
    labels = torch.stack(labels, dim=0)
    
    return images, None, bboxes, labels


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_cls_loss = 0
    total_box_loss = 0
    total_obj_loss = 0
    total_num_fg = 0
    
    num_batches = len(train_loader)
    
    for batch_idx, (images, _, bboxes, labels) in enumerate(train_loader):
        images = images.to(device)
        bboxes = bboxes.to(device)
        labels = labels.to(device)
        
        # Forward pass
        predictions = model(images)
        
        # Compute loss
        loss, loss_dict = criterion(predictions, None, bboxes, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_cls_loss += loss_dict['cls_loss']
        total_box_loss += loss_dict['box_loss']
        total_obj_loss += loss_dict['obj_loss']
        total_num_fg += loss_dict['num_fg']
        
        # Print progress
        if (batch_idx + 1) % 50 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            avg_fg = total_num_fg / (batch_idx + 1)
            print(f"  [{batch_idx+1}/{num_batches}] Loss: {avg_loss:.4f}, FG: {avg_fg:.1f}")
        
        # Memory cleanup
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()
    
    # Compute averages
    avg_loss = total_loss / num_batches
    avg_cls_loss = total_cls_loss / num_batches
    avg_box_loss = total_box_loss / num_batches
    avg_obj_loss = total_obj_loss / num_batches
    avg_num_fg = total_num_fg / num_batches
    
    return avg_loss, avg_cls_loss, avg_box_loss, avg_obj_loss, avg_num_fg


def validate(model, val_loader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    
    total_loss = 0
    total_cls_loss = 0
    total_box_loss = 0
    total_obj_loss = 0
    total_num_fg = 0
    
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for batch_idx, (images, _, bboxes, labels) in enumerate(val_loader):
            images = images.to(device)
            bboxes = bboxes.to(device)
            labels = labels.to(device)
            
            # Forward pass
            predictions = model(images)
            
            # Compute loss
            loss, loss_dict = criterion(predictions, None, bboxes, labels)
            
            # Accumulate losses
            total_loss += loss.item()
            total_cls_loss += loss_dict['cls_loss']
            total_box_loss += loss_dict['box_loss']
            total_obj_loss += loss_dict['obj_loss']
            total_num_fg += loss_dict['num_fg']
    
    # Compute averages
    avg_loss = total_loss / num_batches
    avg_cls_loss = total_cls_loss / num_batches
    avg_box_loss = total_box_loss / num_batches
    avg_obj_loss = total_obj_loss / num_batches
    avg_num_fg = total_num_fg / num_batches
    
    return avg_loss, avg_cls_loss, avg_box_loss, avg_obj_loss, avg_num_fg


def main():
    parser = argparse.ArgumentParser(description='Train RoadSignNet-SAL V3')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("ROADSIGNNET-SAL V3 TRAINING (YOLO-INSPIRED)")
    print("="*70)
    print(f"Device: {device}")
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print(f"Architecture: CSPDarknet-Nano + PAN-FPN + Decoupled Head")
    
    # Create model
    model = create_roadsignnet_v3(
        num_classes=config['model']['num_classes'],
        width_mult=0.25,
        depth_mult=0.33
    )
    model.to(device)
    
    # Create loss
    criterion = create_loss_v3(
        num_classes=config['model']['num_classes'],
        strides=[8, 16, 32],
        lambda_cls=config['training']['loss']['lambda_cls'],
        lambda_box=config['training']['loss']['lambda_box'],
        lambda_obj=config['training']['loss']['lambda_obj']
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['optimizer']['lr'],
        weight_decay=config['training']['optimizer']['weight_decay'],
        betas=config['training']['optimizer']['betas']
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['scheduler']['T_max'],
        eta_min=config['training']['scheduler']['eta_min']
    )
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    
    print(f"\nTrain samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Training loop
    start_epoch = 1
    best_val_loss = float('inf')
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"\n✓ Resumed from epoch {start_epoch-1}")
    
    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{config['training']['epochs']}")
        print(f"{'='*70}")
        
        # Train
        train_loss, train_cls, train_box, train_obj, train_fg = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_cls, val_box, val_obj, val_fg = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Step scheduler
        scheduler.step()
        
        # Print epoch summary
        print(f"\nTrain Loss: {train_loss:.4f} (cls: {train_cls:.4f}, box: {train_box:.4f}, obj: {train_obj:.4f}, fg: {train_fg:.1f})")
        print(f"Val Loss:   {val_loss:.4f} (cls: {val_cls:.4f}, box: {val_box:.4f}, obj: {val_obj:.4f}, fg: {val_fg:.1f})")
        print(f"LR:         {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = 'outputs/checkpoints/best_model_v3.pth'
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config
            }, checkpoint_path)
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
        
        # Save periodic checkpoint
        if epoch % 10 == 0:
            checkpoint_path = f'outputs/checkpoints/checkpoint_v3_epoch_{epoch}.pth'
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config
            }, checkpoint_path)
            print(f"✓ Saved checkpoint: epoch_{epoch}")
        
        # Memory cleanup
        torch.cuda.empty_cache()
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
