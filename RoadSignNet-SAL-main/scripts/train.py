#!/usr/bin/env python3
"""
RoadSignNet-SAL Training Script
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml
from pathlib import Path
import argparse
import time
from datetime import datetime

from roadsignnet_sal.model import create_roadsignnet_sal
from roadsignnet_sal.loss import RoadSignNetLoss
from roadsignnet_sal.dataset import create_dataloader


def load_config(config_path):
    """Load configuration from YAML file"""
    config_path = Path(config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert relative paths to absolute
    base_dir = config_path.parent.parent
    for key in ['train_img_dir', 'train_label_dir', 'val_img_dir', 'val_label_dir', 'test_img_dir', 'test_label_dir']:
        if key in config.get('data', {}):
            path = config['data'][key]
            if not os.path.isabs(path):
                config['data'][key] = str(base_dir / path)
    
    for key in ['save_dir', 'log_dir']:
        if key in config.get('checkpoint', {}):
            path = config['checkpoint'][key]
            if not os.path.isabs(path):
                config['checkpoint'][key] = str(base_dir / path)
        if key in config.get('logging', {}):
            path = config['logging'][key]
            if not os.path.isabs(path):
                config['logging'][key] = str(base_dir / path)
        if key in config.get('export', {}):
            path = config['export'][key]
            if not os.path.isabs(path):
                config['export'][key] = str(base_dir / path)
    
    return config


def setup_directories(config):
    """Create necessary directories"""
    dirs = [
        config['checkpoint']['save_dir'],
        config['logging']['log_dir'],
        config['export']['save_dir']
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def train(config):
    """Main training loop"""
    
    # Setup
    setup_directories(config)
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("ROADSIGNNET-SAL TRAINING")
    print("="*70)
    print(f"Device: {device}")
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch Size: {config['training']['batch_size']}")
    
    # Model
    model = create_roadsignnet_sal(
        num_classes=config['model']['num_classes'],
        width_multiplier=config['model']['width_multiplier']
    ).to(device)
    
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss
    criterion = RoadSignNetLoss(
        num_classes=config['model']['num_classes'],
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
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=config['training']['scheduler']['eta_min']
    )
    
    # Data loaders
    train_loader = create_dataloader(
        img_dir=config['data']['train_img_dir'],
        label_dir=config['data']['train_label_dir'],
        batch_size=config['training']['batch_size'],
        img_size=config['data']['img_size'],
        augment=True,
        num_workers=config['training']['num_workers'],
        shuffle=True
    )
    
    val_loader = create_dataloader(
        img_dir=config['data']['val_img_dir'],
        label_dir=config['data']['val_label_dir'],
        batch_size=config['training']['batch_size'],
        img_size=config['data']['img_size'],
        augment=False,
        num_workers=config['training']['num_workers'],
        shuffle=False
    )
    
    print(f"\nTrain samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # TensorBoard
    if config['logging']['use_tensorboard']:
        writer = SummaryWriter(config['logging']['log_dir'])
    
    # Training history
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(config['training']['epochs']):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{config['training']['epochs']}")
        print(f"{'='*70}")
        
        # TRAIN
        model.train()
        train_loss = 0
        train_loss_dict_sum = {}
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (images, bboxes, labels) in enumerate(pbar):
            images = images.to(device)
            bboxes = bboxes.to(device)
            labels = labels.to(device)
            
            # Forward
            predictions = model(images)
            loss, loss_dict = criterion(predictions, None, bboxes, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()
            
            # Accumulate
            train_loss += loss.item()
            for key, value in loss_dict.items():
                if key != 'num_pos':
                    train_loss_dict_sum[key] = train_loss_dict_sum.get(key, 0) + value
            
            num_pos = loss_dict.get('num_pos', 0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'pos': num_pos})
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_loss_dict = {k: v / len(train_loader) for k, v in train_loss_dict_sum.items()}
        
        # VALIDATE
        model.eval()
        val_loss = 0
        val_loss_dict_sum = {}
        
        with torch.no_grad():
            for images, bboxes, labels in tqdm(val_loader, desc='Validation'):
                images = images.to(device)
                bboxes = bboxes.to(device)
                labels = labels.to(device)
                predictions = model(images)
                loss, loss_dict = criterion(predictions, None, bboxes, labels)
                val_loss += loss.item()
                for key, value in loss_dict.items():
                    if key != 'num_pos':
                        val_loss_dict_sum[key] = val_loss_dict_sum.get(key, 0) + value
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_loss_dict = {k: v / len(val_loader) for k, v in val_loss_dict_sum.items()}
        
        # Scheduler step
        scheduler.step()
        
        # Print summary
        print(f"\nTrain Loss: {avg_train_loss:.4f} (cls: {avg_train_loss_dict.get('cls_loss',0):.4f}, box: {avg_train_loss_dict.get('box_loss',0):.4f}, obj: {avg_train_loss_dict.get('obj_loss',0):.4f})")
        print(f"Val Loss:   {avg_val_loss:.4f} (cls: {avg_val_loss_dict.get('cls_loss',0):.4f}, box: {avg_val_loss_dict.get('box_loss',0):.4f}, obj: {avg_val_loss_dict.get('obj_loss',0):.4f})")
        print(f"LR:         {optimizer.param_groups[0]['lr']:.6f}")
        
        # TensorBoard logging
        if config['logging']['use_tensorboard']:
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = Path(config['checkpoint']['save_dir']) / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'config': config
            }, save_path)
            print(f"✓ Best model saved: {save_path}")
        
        # Periodic checkpoint
        if (epoch + 1) % config['checkpoint']['save_interval'] == 0:
            save_path = Path(config['checkpoint']['save_dir']) / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)
    
    if config['logging']['use_tensorboard']:
        writer.close()
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED!")
    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train RoadSignNet-SAL')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    train(config)