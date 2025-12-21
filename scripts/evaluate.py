#!/usr/bin/env python3
"""
RoadSignNet-SAL Evaluation Script with Accuracy Metrics
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from tqdm import tqdm
import yaml
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from roadsignnet_sal.model import create_roadsignnet_sal, create_roadsignnet_transfer
from roadsignnet_sal.model_v2 import create_roadsignnet_v2
from roadsignnet_sal.model_v3 import create_roadsignnet_v3
from roadsignnet_sal.model_optimized import create_roadsignnet_optimized
from roadsignnet_sal.loss import RoadSignNetLoss, DetectionDecoder
from roadsignnet_sal.loss_v2 import create_loss_v2, AnchorFreeDecoder
from roadsignnet_sal.loss_v3 import V3Loss, V3Decoder
from roadsignnet_sal.dataset import create_dataloader


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area + 1e-6
    
    return inter_area / union_area


def calculate_ap(precisions, recalls):
    """Calculate AP using all-point interpolation"""
    # Sort by recall
    sorted_indices = np.argsort(recalls)
    recalls = np.array(recalls)[sorted_indices]
    precisions = np.array(precisions)[sorted_indices]
    
    # Add sentinel values
    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[0], precisions, [0]])
    
    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # Calculate AP
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    return ap


def detect_model_type(checkpoint):
    """Detect if checkpoint is from optimized, V3, V2, transfer learning, or original model"""
    state_dict = checkpoint['model_state_dict']
    
    # Collect key patterns
    has_spatial_att = any('spatial_att' in k for k in state_dict.keys())
    has_bu_convs = any('neck.bu_convs' in k for k in state_dict.keys())
    has_stage_v3 = any(k.startswith('neck_c2f') for k in state_dict.keys())
    has_v2_attention = any(k.startswith('detection_head.spatial_attention') for k in state_dict.keys())
    has_transfer = any(k.startswith('backbone.features') for k in state_dict.keys())
    
    # Check for optimized model (has spatial_att + bottom-up convs in neck)
    if has_spatial_att and has_bu_convs:
        return 'optimized'
    
    # Check for V3 model
    if has_stage_v3:
        return 'v3'
    
    # Check for V2 model
    if has_v2_attention:
        return 'v2'
    
    # Check for transfer learning
    if has_transfer:
        return 'transfer'
    
    return 'original'


def evaluate(config, checkpoint_path, backbone='mobilenet_v3_small'):
    """Evaluate model on test set with accuracy metrics"""
    
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("ROADSIGNNET-SAL EVALUATION")
    print("="*70)
    
    # Load checkpoint first to detect model type
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_type = detect_model_type(checkpoint)
    
    num_classes = config['model']['num_classes']
    
    # Create appropriate model
    if model_type == 'optimized':
        print("✓ Detected RoadSignNet-SAL Optimized (enhanced V1 with novel contributions)")
        # Get width multiplier from checkpoint config or default to 1.35
        width_mult = checkpoint.get('config', {}).get('width_multiplier', 1.35)
        model = create_roadsignnet_optimized(
            num_classes=num_classes,
            width_multiplier=width_mult
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✓ Model loaded from: {checkpoint_path}")
        print(f"✓ Trained for {checkpoint['epoch']+1} epochs")
        print(f"✓ Number of classes: {num_classes}")
        print(f"✓ Device: {device}")
        
        # Uses same loss and decoder as V1
        criterion = RoadSignNetLoss(
            num_classes=num_classes,
            lambda_cls=config['training']['loss']['lambda_cls'],
            lambda_box=config['training']['loss']['lambda_box'],
            lambda_obj=config['training']['loss']['lambda_obj']
        )
        
        decoder = DetectionDecoder(
            num_classes=num_classes,
            conf_thresh=0.42,  # Balanced threshold to reduce both FP and FN
            iou_thresh=0.45
        )
        
    elif model_type == 'v3':
        print("✓ Detected RoadSignNet-SAL V3 (YOLO-inspired multi-scale)")
        model = create_roadsignnet_v3(
            num_classes=num_classes,
            width_mult=0.25,
            depth_mult=0.33
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✓ Model loaded from: {checkpoint_path}")
        print(f"✓ Trained for {checkpoint['epoch']+1} epochs")
        print(f"✓ Number of classes: {num_classes}")
        print(f"✓ Device: {device}")
        
        # V3 uses multi-scale loss and decoder
        criterion = V3Loss(num_classes=num_classes, strides=[8, 16, 32])
        decoder = V3Decoder(
            num_classes=num_classes,
            conf_thresh=0.25,
            nms_thresh=0.45,
            max_detections=300,
            strides=[8, 16, 32]
        )
        
    elif model_type == 'v2':
        print("✓ Detected RoadSignNet-SAL V2 (anchor-free with novel contributions)")
        model = create_roadsignnet_v2(
            num_classes=num_classes,
            enhanced=True  # Use enhanced mode with novel attention modules
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✓ Model loaded from: {checkpoint_path}")
        print(f"✓ Trained for {checkpoint['epoch']+1} epochs")
        print(f"✓ Number of classes: {num_classes}")
        print(f"✓ Device: {device}")
        
        # V2 uses anchor-free loss and decoder
        criterion = create_loss_v2(num_classes=num_classes)
        decoder = AnchorFreeDecoder(
            num_classes=num_classes,
            conf_thresh=0.1,  # Lower threshold for Gaussian heatmaps
            nms_thresh=0.45
        )
        
    elif model_type == 'transfer':
        print(f"✓ Detected transfer learning model (backbone: {backbone})")
        model = create_roadsignnet_transfer(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=False  # Don't need pretrained weights, we'll load from checkpoint
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✓ Model loaded from: {checkpoint_path}")
        print(f"✓ Trained for {checkpoint['epoch']+1} epochs")
        print(f"✓ Number of classes: {num_classes}")
        print(f"✓ Device: {device}")
        
        # Loss function
        criterion = RoadSignNetLoss(
            num_classes=num_classes,
            lambda_cls=config['training']['loss']['lambda_cls'],
            lambda_box=config['training']['loss']['lambda_box'],
            lambda_obj=config['training']['loss']['lambda_obj']
        )
        
        # Decoder
        decoder = DetectionDecoder(
            num_classes=num_classes,
            conf_thresh=0.25,
            iou_thresh=0.45
        )
        
    else:
        print("✓ Detected original RoadSignNet-SAL model")
        model = create_roadsignnet_sal(
            num_classes=num_classes,
            width_multiplier=config['model']['width_multiplier']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✓ Model loaded from: {checkpoint_path}")
        print(f"✓ Trained for {checkpoint['epoch']+1} epochs")
        print(f"✓ Number of classes: {num_classes}")
        print(f"✓ Device: {device}")
        
        # Loss function
        criterion = RoadSignNetLoss(
            num_classes=num_classes,
            lambda_cls=config['training']['loss']['lambda_cls'],
            lambda_box=config['training']['loss']['lambda_box'],
            lambda_obj=config['training']['loss']['lambda_obj']
        )
        
        # Decoder
        decoder = DetectionDecoder(
            num_classes=num_classes,
            conf_thresh=0.42,
        iou_thresh=0.45,
        img_size=config['data']['img_size']
    )
    
    # Test dataloader
    test_loader = create_dataloader(
        img_dir=config['data']['test_img_dir'],
        label_dir=config['data']['test_label_dir'],
        batch_size=1,
        img_size=config['data']['img_size'],
        augment=False,
        num_workers=0,
        shuffle=False
    )
    
    print(f"✓ Test samples: {len(test_loader.dataset)}")
    
    # Metrics tracking
    test_loss = 0
    loss_dict_sum = {}
    
    # Per-class metrics
    class_names = config['model'].get('class_names', [f'class_{i}' for i in range(num_classes)])
    
    # Store all predictions and ground truths for mAP calculation
    all_predictions = {c: [] for c in range(num_classes)}  # (confidence, is_tp)
    total_gt_per_class = {c: 0 for c in range(num_classes)}
    
    iou_threshold = 0.5
    
    print("\nEvaluating...")
    with torch.no_grad():
        for images, bboxes, labels in tqdm(test_loader):
            images = images.to(device)
            bboxes_dev = bboxes.to(device)
            labels_dev = labels.to(device)
            
            predictions = model(images)
            loss, loss_dict = criterion(predictions, None, bboxes_dev, labels_dev)
            
            test_loss += loss.item()
            for key, value in loss_dict.items():
                if key != 'num_pos':
                    loss_dict_sum[key] = loss_dict_sum.get(key, 0) + value
            
            # Decode predictions
            pred_boxes, pred_scores, pred_classes = decoder.decode(predictions)
            
            # Get ground truth
            gt_boxes = []
            gt_classes = []
            for i in range(bboxes.shape[1]):
                if labels[0, i] >= 0:
                    box = bboxes[0, i].cpu().numpy()
                    gt_boxes.append(box)
                    gt_classes.append(labels[0, i].item())
                    total_gt_per_class[labels[0, i].item()] += 1
            
            # Match predictions to GT
            pred_boxes_np = pred_boxes.cpu().numpy() if len(pred_boxes) > 0 else []
            pred_scores_np = pred_scores.cpu().numpy() if len(pred_scores) > 0 else []
            pred_classes_np = pred_classes.cpu().numpy() if len(pred_classes) > 0 else []
            
            matched_gt = set()
            
            # Sort predictions by confidence (already sorted from NMS but ensure)
            if len(pred_scores_np) > 0:
                sort_idx = np.argsort(-pred_scores_np)
                pred_boxes_np = [pred_boxes_np[i] for i in sort_idx]
                pred_scores_np = pred_scores_np[sort_idx]
                pred_classes_np = pred_classes_np[sort_idx]
            
            for pred_idx in range(len(pred_boxes_np)):
                pred_box = pred_boxes_np[pred_idx]
                pred_score = pred_scores_np[pred_idx]
                pred_cls = int(pred_classes_np[pred_idx])
                
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
                    if gt_idx in matched_gt:
                        continue
                    if gt_cls != pred_cls:
                        continue
                    
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    all_predictions[pred_cls].append((pred_score, True))
                    matched_gt.add(best_gt_idx)
                else:
                    all_predictions[pred_cls].append((pred_score, False))
    
    # Calculate per-class AP
    aps = []
    print("\n" + "="*70)
    print("PER-CLASS RESULTS (IoU threshold = {:.1f})".format(iou_threshold))
    print("="*70)
    
    for c in range(num_classes):
        preds = all_predictions[c]
        n_gt = total_gt_per_class[c]
        
        if n_gt == 0:
            print(f"  {class_names[c]}: No GT samples")
            continue
        
        if len(preds) == 0:
            ap = 0.0
            precision = 0.0
            recall = 0.0
        else:
            # Sort by confidence
            preds = sorted(preds, key=lambda x: -x[0])
            
            tp_cumsum = 0
            fp_cumsum = 0
            precisions = []
            recalls = []
            
            for conf, is_tp in preds:
                if is_tp:
                    tp_cumsum += 1
                else:
                    fp_cumsum += 1
                
                precision = tp_cumsum / (tp_cumsum + fp_cumsum)
                recall = tp_cumsum / n_gt
                precisions.append(precision)
                recalls.append(recall)
            
            ap = calculate_ap(precisions, recalls)
            precision = tp_cumsum / (tp_cumsum + fp_cumsum) if (tp_cumsum + fp_cumsum) > 0 else 0
            recall = tp_cumsum / n_gt
        
        aps.append(ap)
        print(f"  {class_names[c]:15s}: AP={ap*100:5.1f}%  P={precision*100:5.1f}%  R={recall*100:5.1f}%  GT={n_gt}")
    
    # Calculate mAP
    mAP = np.mean(aps) if aps else 0.0
    
    # Calculate overall metrics
    total_tp = sum(sum(1 for _, is_tp in preds if is_tp) for preds in all_predictions.values())
    total_fp = sum(sum(1 for _, is_tp in preds if not is_tp) for preds in all_predictions.values())
    total_gt = sum(total_gt_per_class.values())
    
    overall_precision = total_tp / (total_tp + total_fp + 1e-6)
    overall_recall = total_tp / (total_gt + 1e-6)
    f1_score = 2 * overall_precision * overall_recall / (overall_precision + overall_recall + 1e-6)
    
    # Calculate averages
    avg_test_loss = test_loss / len(test_loader)
    avg_loss_dict = {k: v / len(test_loader) for k, v in loss_dict_sum.items()}
    
    # Print results
    print("\n" + "="*70)
    print("OVERALL RESULTS")
    print("="*70)
    
    print("\n📊 LOSS METRICS:")
    print(f"  Total Loss:     {avg_test_loss:.4f}")
    print(f"  Class Loss:     {avg_loss_dict.get('cls_loss', 0):.4f}")
    print(f"  Box Loss:       {avg_loss_dict.get('box_loss', 0):.4f}")
    print(f"  Obj Loss:       {avg_loss_dict.get('obj_loss', 0):.4f}")
    
    print("\n🎯 DETECTION METRICS:")
    print(f"  mAP@0.5:          {mAP*100:.2f}%")
    print(f"  Precision:        {overall_precision*100:.2f}%")
    print(f"  Recall:           {overall_recall*100:.2f}%")
    print(f"  F1-Score:         {f1_score*100:.2f}%")
    
    print("\n📈 COUNTS:")
    print(f"  True Positives:   {total_tp}")
    print(f"  False Positives:  {total_fp}")
    print(f"  Total GT Objects: {total_gt}")
    
    # Save results
    results = {
        'checkpoint': checkpoint_path,
        'test_loss': avg_test_loss,
        'mAP@0.5': mAP,
        'precision': overall_precision,
        'recall': overall_recall,
        'f1_score': f1_score,
        'true_positives': total_tp,
        'false_positives': total_fp,
        'total_gt': total_gt,
        **avg_loss_dict
    }
    
    results_df = pd.DataFrame([results])
    results_path = Path('outputs') / 'evaluation_results.csv'
    results_df.to_csv(results_path, index=False)
    
    print(f"\n✓ Results saved: {results_path}")
    print("="*70)
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--backbone', type=str, default='mobilenet_v3_small',
                       choices=['mobilenet_v3_small', 'mobilenet_v3_large', 'efficientnet_b0', 'resnet18'],
                       help='Backbone used for transfer learning model')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    evaluate(config, args.checkpoint, args.backbone)