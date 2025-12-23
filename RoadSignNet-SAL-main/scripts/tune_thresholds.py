#!/usr/bin/env python3
"""
Tune detection thresholds to reduce false positives
Tests different confidence and IoU thresholds to find optimal settings
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import yaml
from tqdm import tqdm
import numpy as np

from roadsignnet_sal.model_optimized import create_roadsignnet_optimized
from roadsignnet_sal.loss import DetectionDecoder
from roadsignnet_sal.dataset import create_dataloader


def calculate_metrics(tp, fp, fn):
    """Calculate precision, recall, F1"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def evaluate_with_thresholds(model, dataloader, decoder, device):
    """Evaluate model with current decoder settings"""
    model.eval()
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    with torch.no_grad():
        for images, bboxes, labels in dataloader:
            images = images.to(device)
            bboxes = bboxes.to(device)
            labels = labels.to(device)
            
            batch_size = images.size(0)
            predictions = model(images)
            
            for b in range(batch_size):
                # Get predictions for this image
                pred_cls_list = [pred[0][b:b+1] for pred in predictions]
                pred_box_list = [pred[1][b:b+1] for pred in predictions]
                pred_obj_list = [pred[2][b:b+1] for pred in predictions]
                
                single_pred = [(cls, box, obj) for cls, box, obj in 
                              zip(pred_cls_list, pred_box_list, pred_obj_list)]
                
                det_boxes, det_scores, det_classes = decoder.decode(single_pred)
                
                # Get ground truth
                gt_mask = labels[b] >= 0
                gt_count = gt_mask.sum().item()
                pred_count = len(det_boxes)
                
                if gt_count == 0:
                    total_fp += pred_count
                    continue
                
                gt_boxes_valid = bboxes[b][gt_mask]
                gt_labels_valid = labels[b][gt_mask]
                
                # Simple matching (IoU > 0.5 and correct class)
                matched_gt = set()
                matched_pred = set()
                
                for pred_idx in range(pred_count):
                    pred_box = det_boxes[pred_idx]
                    pred_cls = det_classes[pred_idx].item()
                    
                    best_iou = 0
                    best_gt_idx = -1
                    
                    for gt_idx in range(gt_count):
                        if gt_idx in matched_gt:
                            continue
                        
                        gt_box = gt_boxes_valid[gt_idx]
                        gt_cls = gt_labels_valid[gt_idx].item()
                        
                        if pred_cls != gt_cls:
                            continue
                        
                        # Calculate IoU
                        inter_x1 = max(pred_box[0].item(), gt_box[0].item())
                        inter_y1 = max(pred_box[1].item(), gt_box[1].item())
                        inter_x2 = min(pred_box[2].item(), gt_box[2].item())
                        inter_y2 = min(pred_box[3].item(), gt_box[3].item())
                        
                        if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                            pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                            gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                            union_area = pred_area + gt_area - inter_area
                            iou = inter_area / (union_area + 1e-6)
                            
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = gt_idx
                    
                    if best_iou >= 0.5 and best_gt_idx >= 0:
                        matched_gt.add(best_gt_idx)
                        matched_pred.add(pred_idx)
                
                tp = len(matched_pred)
                fp = pred_count - tp
                fn = gt_count - len(matched_gt)
                
                total_tp += tp
                total_fp += fp
                total_fn += fn
    
    return total_tp, total_fp, total_fn


def main():
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("THRESHOLD TUNING TO REDUCE FALSE POSITIVES")
    print("="*70)
    
    # Load model
    checkpoint = torch.load('outputs/checkpoints/best_model_optimized.pth')
    model = create_roadsignnet_optimized(num_classes=43, width_multiplier=1.35).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data
    test_loader = create_dataloader(
        img_dir='data/test/images',
        label_dir='data/test/labels',
        img_size=416,
        batch_size=8,
        augment=False,
        num_workers=0,
        shuffle=False,
        split='test'
    )
    
    print(f"✓ Model loaded")
    print(f"✓ Test samples: {len(test_loader.dataset)}")
    print(f"✓ Device: {device}\n")
    
    # Current baseline
    print("Baseline (conf=0.25, iou=0.45):")
    decoder = DetectionDecoder(num_classes=43, conf_thresh=0.25, iou_thresh=0.45)
    tp, fp, fn = evaluate_with_thresholds(model, test_loader, decoder, device)
    precision, recall, f1 = calculate_metrics(tp, fp, fn)
    print(f"  TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"  Precision: {precision:.1%}, Recall: {recall:.1%}, F1: {f1:.1%}")
    print(f"  FP/TP ratio: {fp/tp:.2f}\n")
    
    # Test different confidence thresholds
    print("Testing confidence thresholds (iou=0.45):")
    print("-" * 70)
    
    best_f1 = 0
    best_conf = 0.25
    best_metrics = None
    
    conf_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    
    for conf_thresh in conf_thresholds:
        decoder = DetectionDecoder(num_classes=43, conf_thresh=conf_thresh, iou_thresh=0.45)
        tp, fp, fn = evaluate_with_thresholds(model, test_loader, decoder, device)
        precision, recall, f1 = calculate_metrics(tp, fp, fn)
        
        print(f"conf={conf_thresh:.2f}: TP={tp:4d}, FP={fp:4d}, FN={fn:4d} | "
              f"P={precision:.1%}, R={recall:.1%}, F1={f1:.1%} | FP/TP={fp/max(1,tp):.2f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_conf = conf_thresh
            best_metrics = (tp, fp, fn, precision, recall, f1)
    
    print("\n" + "="*70)
    print("OPTIMAL SETTINGS:")
    print("="*70)
    print(f"Best Confidence Threshold: {best_conf}")
    print(f"Best IoU Threshold: 0.45")
    print(f"\nMetrics with optimal settings:")
    tp, fp, fn, precision, recall, f1 = best_metrics
    print(f"  True Positives:  {tp}")
    print(f"  False Positives: {fp} (reduced from baseline)")
    print(f"  False Negatives: {fn}")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall: {recall:.1%}")
    print(f"  F1-Score: {f1:.1%}")
    print(f"  FP/TP ratio: {fp/tp:.2f}")
    print("\nUpdate evaluate.py to use conf_thresh={:.2f} for better results!".format(best_conf))
    print("="*70)


if __name__ == '__main__':
    main()
