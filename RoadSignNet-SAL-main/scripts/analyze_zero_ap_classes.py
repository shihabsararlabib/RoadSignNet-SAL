"""
Analyze zero-AP classes to understand why model fails on them
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
from pathlib import Path
import numpy as np
from collections import defaultdict
import torch

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class_names = config['model']['class_names']
num_classes = config['model']['num_classes']

# Zero-AP classes from evaluation
zero_ap_classes = ['narrow_road', 'school_nearby', 'speed_limit_100', 'speed_limit_20']
zero_ap_indices = [class_names.index(c) for c in zero_ap_classes]

print("="*70)
print("ANALYZING ZERO-AP CLASSES")
print("="*70)
print(f"Classes with 0% AP: {zero_ap_classes}")
print(f"Class indices: {zero_ap_indices}\n")

# Analyze training data distribution
def count_class_samples(label_dir, dataset_name):
    """Count how many samples per class in dataset"""
    class_counts = defaultdict(int)
    label_path = Path(label_dir)
    
    for label_file in label_path.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
    
    return class_counts

print("📊 CLASS DISTRIBUTION ANALYSIS")
print("-"*70)

# Training set
train_counts = count_class_samples(config['data']['train_label_dir'], 'train')
print("\n🔹 TRAINING SET:")
print(f"{'Class':<20} {'Count':<10} {'% of Total':<12}")
print("-"*70)
total_train = sum(train_counts.values())
for class_name in zero_ap_classes:
    class_id = class_names.index(class_name)
    count = train_counts.get(class_id, 0)
    percentage = (count / total_train * 100) if total_train > 0 else 0
    print(f"{class_name:<20} {count:<10} {percentage:>10.2f}%")

# Test set
test_counts = count_class_samples(config['data']['test_label_dir'], 'test')
print("\n🔹 TEST SET:")
print(f"{'Class':<20} {'Count':<10}")
print("-"*70)
for class_name in zero_ap_classes:
    class_id = class_names.index(class_name)
    count = test_counts.get(class_id, 0)
    print(f"{class_name:<20} {count:<10}")

# Validation set (use correct key)
valid_label_dir = 'data/valid/labels'
valid_counts = count_class_samples(valid_label_dir, 'valid')
print("\n🔹 VALIDATION SET:")
print(f"{'Class':<20} {'Count':<10}")
print("-"*70)
for class_name in zero_ap_classes:
    class_id = class_names.index(class_name)
    count = valid_counts.get(class_id, 0)
    print(f"{class_name:<20} {count:<10}")

# Compare with well-performing classes
print("\n\n📈 COMPARISON WITH HIGH-AP CLASSES")
print("-"*70)
high_ap_classes = ['parking', 'u_turn', 'bus_stop', 'red_light']  # Classes with >80% AP
print(f"{'Class':<20} {'Train':<10} {'Valid':<10} {'Test':<10}")
print("-"*70)
for class_name in high_ap_classes:
    class_id = class_names.index(class_name)
    train_c = train_counts.get(class_id, 0)
    valid_c = valid_counts.get(class_id, 0)
    test_c = test_counts.get(class_id, 0)
    print(f"{class_name:<20} {train_c:<10} {valid_c:<10} {test_c:<10}")

print("\n🔹 ZERO-AP CLASSES:")
print("-"*70)
for class_name in zero_ap_classes:
    class_id = class_names.index(class_name)
    train_c = train_counts.get(class_id, 0)
    valid_c = valid_counts.get(class_id, 0)
    test_c = test_counts.get(class_id, 0)
    print(f"{class_name:<20} {train_c:<10} {valid_c:<10} {test_c:<10}")

# Analyze class imbalance
print("\n\n⚖️ CLASS IMBALANCE ANALYSIS")
print("-"*70)
all_train_counts = sorted(train_counts.items(), key=lambda x: x[1], reverse=True)
max_samples = all_train_counts[0][1]
min_samples = all_train_counts[-1][1]
print(f"Most frequent class: {class_names[all_train_counts[0][0]]} ({max_samples} samples)")
print(f"Least frequent class: {class_names[all_train_counts[-1][0]]} ({min_samples} samples)")
print(f"Imbalance ratio: {max_samples / max(min_samples, 1):.1f}x")

print("\n🔹 Zero-AP classes ranking:")
for class_name in zero_ap_classes:
    class_id = class_names.index(class_name)
    count = train_counts.get(class_id, 0)
    if count > 0:
        rank = next(i for i, (cid, _) in enumerate(all_train_counts, 1) if cid == class_id)
        print(f"{class_name:<20}: {count:>5} samples (rank {rank}/{len(all_train_counts)})")
    else:
        print(f"{class_name:<20}: {count:>5} samples (NOT IN TRAINING SET)")

# Check predictions from model
print("\n\n🔍 CHECKING MODEL PREDICTIONS")
print("-"*70)
print("Loading model and checking what it predicts for zero-AP classes...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from roadsignnet_sal.model_optimized import create_roadsignnet_optimized
from roadsignnet_sal.loss import DetectionDecoder
from roadsignnet_sal.dataset import create_dataloader

# Load model
checkpoint = torch.load('outputs/checkpoints/checkpoint_optimized_epoch_100.pth', map_location=device)
width_mult = checkpoint.get('width_multiplier', 1.35)
model = create_roadsignnet_optimized(num_classes=num_classes, width_multiplier=width_mult).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Decoder
decoder = DetectionDecoder(num_classes=num_classes, conf_thresh=0.25, iou_thresh=0.45)

# Test loader
test_loader = create_dataloader(
    img_dir=config['data']['test_img_dir'],
    label_dir=config['data']['test_label_dir'],
    batch_size=1,
    img_size=config['data']['img_size'],
    augment=False,
    num_workers=0,
    shuffle=False
)

# Track predictions for zero-AP classes
zero_ap_predictions = defaultdict(list)  # {class_id: [confidence_scores]}
zero_ap_gt_count = defaultdict(int)

with torch.no_grad():
    for images, bboxes, labels in test_loader:
        # Check if this image has any zero-AP class GT
        has_zero_ap = False
        for label in labels[0]:
            if label.item() in zero_ap_indices:
                has_zero_ap = True
                zero_ap_gt_count[label.item()] += 1
        
        if not has_zero_ap:
            continue
        
        images = images.to(device)
        predictions = model(images)
        pred_boxes, pred_scores, pred_classes = decoder.decode(predictions)
        
        # Check if model predicted any zero-AP class
        for score, cls in zip(pred_scores, pred_classes):
            if cls in zero_ap_indices:
                zero_ap_predictions[cls].append(score)

print(f"\n📋 PREDICTION STATISTICS:")
print("-"*70)
print(f"{'Class':<20} {'GT Count':<12} {'Pred Count':<12} {'Max Conf':<12} {'Avg Conf':<12}")
print("-"*70)
for class_name in zero_ap_classes:
    class_id = class_names.index(class_name)
    gt_count = zero_ap_gt_count.get(class_id, 0)
    pred_count = len(zero_ap_predictions.get(class_id, []))
    preds = zero_ap_predictions.get(class_id, [])
    max_conf = max(preds) if preds else 0.0
    avg_conf = np.mean(preds) if preds else 0.0
    print(f"{class_name:<20} {gt_count:<12} {pred_count:<12} {max_conf:<12.3f} {avg_conf:<12.3f}")

print("\n\n" + "="*70)
print("DIAGNOSIS & RECOMMENDATIONS")
print("="*70)

# Diagnosis
issues = []
for class_name in zero_ap_classes:
    class_id = class_names.index(class_name)
    train_c = train_counts.get(class_id, 0)
    test_c = test_counts.get(class_id, 0)
    pred_count = len(zero_ap_predictions.get(class_id, []))
    
    if train_c == 0:
        issues.append(f"❌ {class_name}: NO TRAINING DATA (has {test_c} test samples)")
    elif train_c < 50:
        issues.append(f"⚠️  {class_name}: SEVERELY UNDERREPRESENTED ({train_c} train, {test_c} test)")
    elif pred_count == 0:
        issues.append(f"🔴 {class_name}: MODEL NEVER PREDICTS THIS CLASS")
    else:
        max_conf = max(zero_ap_predictions.get(class_id, [0]))
        if max_conf < 0.25:
            issues.append(f"📉 {class_name}: LOW CONFIDENCE (max={max_conf:.3f}, predictions={pred_count})")

print("\n🔍 IDENTIFIED ISSUES:")
for issue in issues:
    print(f"  {issue}")

print("\n\n💡 RECOMMENDED SOLUTIONS:")
print("-"*70)
print("1. CLASS WEIGHTS: Increase loss weights for underrepresented classes")
print("   - Implement focal loss or weighted cross-entropy")
print("   - Weight inversely proportional to frequency")
print()
print("2. DATA AUGMENTATION: Apply aggressive augmentation to rare classes")
print("   - Mixup/CutMix with rare class samples")
print("   - Copy-paste augmentation")
print("   - Oversampling during training")
print()
print("3. CONFIDENCE THRESHOLD: Use lower threshold for rare classes")
print("   - Per-class confidence thresholds")
print("   - Adaptive thresholding based on class difficulty")
print()
print("4. TRAINING STRATEGY:")
print("   - Two-stage training: general → fine-tune on rare classes")
print("   - Class-balanced sampling")
print("   - Hard example mining")
print("="*70)
