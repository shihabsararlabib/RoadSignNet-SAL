"""
Fine-tune thresholds between 0.40-0.45 to find optimal FP/FN balance
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import numpy as np

from roadsignnet_sal.model_optimized import create_roadsignnet_optimized
from roadsignnet_sal.dataset import create_dataloader
from roadsignnet_sal.loss import DetectionDecoder, RoadSignNetLoss


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


def evaluate_threshold(model, test_loader, criterion, num_classes, conf_thresh, iou_thresh, device):
    """Evaluate model at specific threshold"""
    decoder = DetectionDecoder(
        num_classes=num_classes,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh
    )
    
    # Storage for per-class predictions
    all_predictions = {c: [] for c in range(num_classes)}
    total_gt_per_class = {c: 0 for c in range(num_classes)}
    
    model.eval()
    with torch.no_grad():
        for images, bboxes, labels in test_loader:
            images = images.to(device)
            predictions = model(images)
            
            # Decode predictions
            pred_boxes, pred_scores, pred_classes = decoder.decode(predictions)
            
            # Process ground truth for this batch (batch_size=1)
            gt_boxes = []
            gt_labels = []
            for i in range(bboxes.shape[1]):
                if labels[0, i] >= 0:  # Valid label
                    box = bboxes[0, i].cpu().numpy()
                    gt_boxes.append(box)
                    gt_labels.append(labels[0, i].item())
                    total_gt_per_class[int(labels[0, i].item())] += 1
            
            # Convert predictions to numpy
            pred_boxes_np = pred_boxes.cpu().numpy()
            pred_scores_np = pred_scores.cpu().numpy()
            pred_classes_np = pred_classes.cpu().numpy()
            
            # Match predictions to ground truth
            gt_matched = [False] * len(gt_boxes)
            
            for pred_box, pred_score, pred_class in zip(pred_boxes_np, pred_scores_np, pred_classes_np):
                pred_label = int(pred_class)
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                    if gt_label != pred_label:
                        continue
                    if gt_matched[gt_idx]:
                        continue
                    
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                is_tp = best_iou >= 0.5 and best_gt_idx >= 0
                if is_tp:
                    gt_matched[best_gt_idx] = True
                
                all_predictions[pred_label].append((float(pred_score), is_tp))
    
    # Calculate metrics
    total_tp = sum(sum(1 for _, is_tp in preds if is_tp) for preds in all_predictions.values())
    total_fp = sum(sum(1 for _, is_tp in preds if not is_tp) for preds in all_predictions.values())
    total_gt = sum(total_gt_per_class.values())
    total_fn = total_gt - total_tp
    
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_gt + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return {
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main():
    print("="*70)
    print("FINE-TUNING THRESHOLDS (0.40 - 0.45)")
    print("="*70)
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = config['model']['num_classes']
    
    # Load model
    checkpoint = torch.load('outputs/checkpoints/best_model_optimized.pth', map_location=device)
    width_mult = checkpoint.get('width_multiplier', 1.35)
    
    print("✓ Creating RoadSignNet-SAL Optimized")
    print(f"  Width Multiplier: {width_mult}")
    
    model = create_roadsignnet_optimized(
        num_classes=num_classes,
        width_multiplier=width_mult
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Model loaded")
    
    # Create loss for evaluation
    criterion = RoadSignNetLoss(
        num_classes=num_classes,
        lambda_cls=config['training']['loss']['lambda_cls'],
        lambda_box=config['training']['loss']['lambda_box'],
        lambda_obj=config['training']['loss']['lambda_obj']
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
    print(f"✓ Device: {device}\n")
    
    # Test thresholds
    thresholds = [0.40, 0.41, 0.42, 0.43, 0.44, 0.45]
    iou_thresh = 0.45
    
    print(f"Testing confidence thresholds (iou={iou_thresh}):")
    print("-"*70)
    
    results = []
    for conf in thresholds:
        print(f"Testing conf={conf:.2f}...", end=" ", flush=True)
        
        metrics = evaluate_threshold(
            model, test_loader, criterion, num_classes, conf, iou_thresh, device
        )
        
        tp = metrics['tp']
        fp = metrics['fp']
        fn = metrics['fn']
        precision = metrics['precision']
        recall = metrics['recall']
        f1 = metrics['f1']
        fp_tp_ratio = fp / tp if tp > 0 else 0
        
        # Calculate total errors
        total_errors = fp + fn
        
        results.append({
            'conf': conf,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fp_tp_ratio': fp_tp_ratio,
            'total_errors': total_errors
        })
        
        print(f"TP={tp:4d}, FP={fp:4d}, FN={fn:4d} | "
              f"P={precision:5.1f}%, R={recall:5.1f}%, F1={f1:5.1f}% | "
              f"FP/TP={fp_tp_ratio:.2f} | Total Errors={total_errors}")
    
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    # Find best by different criteria
    best_f1 = max(results, key=lambda x: x['f1'])
    best_balance = min(results, key=lambda x: x['total_errors'])
    best_recall = max(results, key=lambda x: x['recall'])
    
    print(f"\n📊 Best F1-Score:")
    print(f"   conf={best_f1['conf']:.2f}: F1={best_f1['f1']:.1f}%, "
          f"FP={best_f1['fp']}, FN={best_f1['fn']}")
    
    print(f"\n⚖️  Best Error Balance (lowest FP+FN):")
    print(f"   conf={best_balance['conf']:.2f}: Total Errors={best_balance['total_errors']}, "
          f"FP={best_balance['fp']}, FN={best_balance['fn']}, F1={best_balance['f1']:.1f}%")
    
    print(f"\n🎯 Best Recall (fewest FN):")
    print(f"   conf={best_recall['conf']:.2f}: Recall={best_recall['recall']:.1f}%, "
          f"FN={best_recall['fn']}, FP={best_recall['fp']}, F1={best_recall['f1']:.1f}%")
    
    print("\n" + "="*70)
    print("RECOMMENDATION:")
    print("="*70)
    
    # Recommend based on balanced approach
    if best_balance['conf'] == best_f1['conf']:
        print(f"✓ Use conf={best_balance['conf']:.2f} (best on both F1 and error balance)")
    else:
        print(f"✓ For best F1: Use conf={best_f1['conf']:.2f}")
        print(f"✓ For balanced errors: Use conf={best_balance['conf']:.2f}")
    
    print("="*70)

if __name__ == '__main__':
    main()
