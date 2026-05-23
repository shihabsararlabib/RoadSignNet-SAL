#!/usr/bin/env python3
"""
Train RoadSignNet-SAL with hybrid teacher distillation.
Teacher: hybrid backbone (CNN + TinyViT via timm)
Student: RoadSignNet-SAL or transfer backbone
"""

import sys
import os
from pathlib import Path
import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import gc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from roadsignnet_sal.model import create_roadsignnet_sal, create_roadsignnet_transfer
from roadsignnet_sal.loss import RoadSignNetLoss
from roadsignnet_sal.dataset import create_dataloader


def load_config(config_path):
    """Load configuration from YAML file and resolve paths."""
    config_path = Path(config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

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

    return config


def setup_directories(config):
    """Create necessary directories."""
    dirs = [
        config['checkpoint']['save_dir'],
        config['logging']['log_dir']
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def align_like(src, ref):
    """Align spatial dimensions of src to ref if needed."""
    if src.shape[-2:] != ref.shape[-2:]:
        src = F.interpolate(src, size=ref.shape[-2:], mode='bilinear', align_corners=False)
    return src


def distill_sigmoid_mse(student_logits, teacher_logits, temperature=1.0):
    """Sigmoid MSE distillation for multi-label logits."""
    s = torch.sigmoid(student_logits / temperature)
    t = torch.sigmoid(teacher_logits / temperature)
    return F.mse_loss(s, t)


def build_teacher(config, device):
    """Create teacher model for distillation."""
    distill_cfg = config.get('distillation', {})
    teacher_backbone = distill_cfg.get('teacher_backbone', 'efficientnet_b0+vit_tiny_patch16_224')
    teacher_ckpt = distill_cfg.get('teacher_checkpoint', None)
    use_kan_cls = distill_cfg.get('teacher_use_kan_cls', config.get('model', {}).get('use_kan_cls', False))
    kan_grid = distill_cfg.get('teacher_kan_grid', config.get('model', {}).get('kan_grid', 8))

    checkpoint = None
    if teacher_ckpt:
        checkpoint = torch.load(teacher_ckpt, map_location=device)
        if any('cls_head.kan.' in k for k in checkpoint['model_state_dict'].keys()):
            use_kan_cls = True
            print("✓ Detected KAN head in teacher checkpoint")

    teacher = create_roadsignnet_transfer(
        num_classes=config['model']['num_classes'],
        backbone=teacher_backbone,
        pretrained=False,
        use_kan_cls=use_kan_cls,
        kan_grid=kan_grid
    ).to(device)

    if teacher_ckpt:
        teacher.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded teacher checkpoint: {teacher_ckpt}")
    else:
        print("⚠️  Teacher checkpoint not provided; distillation uses untrained teacher outputs")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    return teacher


def build_student(config, device):
    """Create student model for distillation."""
    distill_cfg = config.get('distillation', {})
    student_type = distill_cfg.get('student_type', 'sal')
    student_backbone = distill_cfg.get('student_backbone', 'mobilenet_v3_small')
    use_kan_cls = distill_cfg.get('student_use_kan_cls', config.get('model', {}).get('use_kan_cls', False))
    kan_grid = distill_cfg.get('student_kan_grid', config.get('model', {}).get('kan_grid', 8))

    if student_type == 'transfer':
        student = create_roadsignnet_transfer(
            num_classes=config['model']['num_classes'],
            backbone=student_backbone,
            pretrained=True,
            freeze_backbone=False,
            use_kan_cls=use_kan_cls,
            kan_grid=kan_grid
        ).to(device)
        print(f"✓ Student: transfer backbone = {student_backbone}")
    else:
        student = create_roadsignnet_sal(
            num_classes=config['model']['num_classes'],
            width_multiplier=config['model'].get('width_multiplier', 1.0),
            use_kan_cls=use_kan_cls,
            kan_grid=kan_grid
        ).to(device)
        print("✓ Student: RoadSignNet-SAL")

    if use_kan_cls:
        print(f"✓ Student KAN head enabled (grid={kan_grid})")

    return student


def train(config, args):
    setup_directories(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Build teacher and student
    teacher = build_teacher(config, device)
    student = build_student(config, device)

    # Losses
    criterion = RoadSignNetLoss(
        num_classes=config['model']['num_classes'],
        lambda_cls=config['training']['loss']['lambda_cls'],
        lambda_box=config['training']['loss']['lambda_box'],
        lambda_obj=config['training']['loss']['lambda_obj']
    )

    # Distillation weights
    distill_cfg = config.get('distillation', {})
    t = distill_cfg.get('temperature', 2.0)
    lambda_cls = distill_cfg.get('lambda_cls', 1.0)
    lambda_box = distill_cfg.get('lambda_box', 0.5)
    lambda_obj = distill_cfg.get('lambda_obj', 0.5)
    lambda_total = distill_cfg.get('lambda_total', 1.0)

    # Optimizer
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config['training']['optimizer']['lr'],
        weight_decay=config['training']['optimizer']['weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=config['training']['scheduler'].get('eta_min', 1e-6)
    )

    # Dataloaders
    if not args.smoke:
        train_loader = create_dataloader(
            img_dir=config['data']['train_img_dir'],
            label_dir=config['data']['train_label_dir'],
            batch_size=config['training']['batch_size'],
            img_size=config['data']['img_size'],
            augment=True,
            num_workers=config['training'].get('num_workers', 0),
            shuffle=True
        )

        val_loader = create_dataloader(
            img_dir=config['data']['val_img_dir'],
            label_dir=config['data']['val_label_dir'],
            batch_size=config['training']['batch_size'],
            img_size=config['data']['img_size'],
            augment=False,
            num_workers=config['training'].get('num_workers', 0),
            shuffle=False
        )
    else:
        # Synthetic loader for smoke test
        train_loader = [(
            torch.randn(2, 3, 224, 224),
            torch.zeros(2, 1, 4),
            torch.full((2, 1), -1, dtype=torch.long)
        )]
        val_loader = train_loader

    # TensorBoard
    writer = None
    if config.get('logging', {}).get('tensorboard', True):
        writer = SummaryWriter(config['logging']['log_dir'])

    best_val = float('inf')

    for epoch in range(config['training']['epochs']):
        student.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for images, bboxes, labels in pbar:
            images = images.to(device)
            bboxes = bboxes.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            # Student predictions
            student_preds = student(images)
            det_loss, det_dict = criterion(student_preds, None, bboxes, labels)

            # Teacher predictions (no grad)
            with torch.no_grad():
                teacher_preds = teacher(images)

            # Distillation loss across scales
            distill_loss = 0.0
            for (s_cls, s_box, s_obj), (t_cls, t_box, t_obj) in zip(student_preds, teacher_preds):
                t_cls = align_like(t_cls, s_cls)
                t_box = align_like(t_box, s_box)
                t_obj = align_like(t_obj, s_obj)

                distill_loss += (
                    lambda_cls * distill_sigmoid_mse(s_cls, t_cls, t) +
                    lambda_box * F.l1_loss(s_box, t_box) +
                    lambda_obj * distill_sigmoid_mse(s_obj, t_obj, t)
                )

            total_loss = det_loss + lambda_total * distill_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=10.0)
            optimizer.step()

            train_loss += total_loss.item()
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'det': f"{det_loss.item():.4f}",
                'distill': f"{distill_loss.item():.4f}"
            })

        avg_train = train_loss / max(1, len(train_loader))

        # Validation (det loss only)
        student.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, bboxes, labels in val_loader:
                images = images.to(device)
                bboxes = bboxes.to(device)
                labels = labels.to(device)
                preds = student(images)
                loss, _ = criterion(preds, None, bboxes, labels)
                val_loss += loss.item()

        avg_val = val_loss / max(1, len(val_loader))

        if writer:
            writer.add_scalar('Loss/train_total', avg_train, epoch)
            writer.add_scalar('Loss/val_det', avg_val, epoch)

        scheduler.step()

        # Save best
        if avg_val < best_val:
            best_val = avg_val
            save_path = Path(config['checkpoint']['save_dir']) / 'best_model_distill.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val,
                'config': config
            }, save_path)
            print(f"✓ Best distill model saved: {save_path}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if args.smoke:
            print("✓ Smoke test complete")
            break

    if writer:
        writer.close()


def main():
    parser = argparse.ArgumentParser(description='Hybrid distillation training')
    parser.add_argument('--config', type=str, default='config/config_v6.yaml')
    parser.add_argument('--smoke', action='store_true', help='Run a 1-iteration synthetic smoke test')
    args = parser.parse_args()

    # Resolve config path
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = Path(__file__).parent.parent / config_path

    config = load_config(config_path)
    train(config, args)


if __name__ == '__main__':
    main()
