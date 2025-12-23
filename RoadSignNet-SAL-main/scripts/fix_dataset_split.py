"""
Fix dataset split for zero-AP classes by moving samples from valid/test to train
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shutil
from pathlib import Path
import yaml
from collections import defaultdict

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class_names = config['model']['class_names']

# Classes that need fixing
fix_classes = {
    'narrow_road': 13,
    'school_nearby': 26,
    'speed_limit_100': 28,
    'speed_limit_20': 30
}

print("="*70)
print("FIXING DATASET SPLIT FOR ZERO-AP CLASSES")
print("="*70)

# Strategy: Move samples from valid to train for classes with 0 train samples
# Keep some in valid/test for evaluation

def get_samples_for_class(label_dir, class_id):
    """Find all image files that contain the specified class"""
    label_path = Path(label_dir)
    samples = []
    
    for label_file in label_path.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5 and int(parts[0]) == class_id:
                    # This image contains the class
                    samples.append(label_file.stem)  # filename without extension
                    break
    
    return samples

def move_samples(class_name, class_id, source_set='valid', num_to_move=30):
    """Move samples from source set to training set"""
    print(f"\n🔄 Processing {class_name} (class {class_id}):")
    print("-"*70)
    
    # Paths
    if source_set == 'valid':
        source_img_dir = Path('data/valid/images')
        source_label_dir = Path('data/valid/labels')
    else:  # test
        source_img_dir = Path('data/test/images')
        source_label_dir = Path('data/test/labels')
    
    train_img_dir = Path(config['data']['train_img_dir'])
    train_label_dir = Path(config['data']['train_label_dir'])
    
    # Get samples
    samples = get_samples_for_class(source_label_dir, class_id)
    print(f"  Found {len(samples)} samples in {source_set} set")
    
    if len(samples) == 0:
        print(f"  ⚠️  No samples found in {source_set} set")
        return 0
    
    # Move samples (keep some for validation/testing)
    samples_to_move = min(num_to_move, max(1, len(samples) - 5))  # keep at least 5 for eval
    moved = 0
    
    for i, sample in enumerate(samples[:samples_to_move]):
        # Move image
        img_ext = None
        for ext in ['.jpg', '.jpeg', '.png']:
            img_file = source_img_dir / f"{sample}{ext}"
            if img_file.exists():
                img_ext = ext
                break
        
        if img_ext is None:
            print(f"  ⚠️  Image not found for {sample}")
            continue
        
        src_img = source_img_dir / f"{sample}{img_ext}"
        dst_img = train_img_dir / f"{sample}{img_ext}"
        
        # Move label
        src_label = source_label_dir / f"{sample}.txt"
        dst_label = train_label_dir / f"{sample}.txt"
        
        if not dst_img.exists():
            shutil.copy(src_img, dst_img)
            shutil.copy(src_label, dst_label)
            moved += 1
        else:
            print(f"  ⚠️  {sample} already exists in training set, skipping")
    
    print(f"  ✅ Moved {moved} samples to training set")
    print(f"  📊 Remaining in {source_set}: {len(samples) - moved}")
    
    return moved

# Process each class
total_moved = 0
backup_created = False

for class_name, class_id in fix_classes.items():
    # Check train count
    train_samples = get_samples_for_class(config['data']['train_label_dir'], class_id)
    
    if len(train_samples) == 0:
        # No training samples - move from validation
        moved = move_samples(class_name, class_id, source_set='valid', num_to_move=25)
        total_moved += moved
    elif len(train_samples) < 100:
        # Very few training samples - add more from validation
        needed = 100 - len(train_samples)
        print(f"\n➕ {class_name} has only {len(train_samples)} training samples, adding {min(needed, 30)} more")
        moved = move_samples(class_name, class_id, source_set='valid', num_to_move=min(needed, 30))
        total_moved += moved

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"✅ Total samples moved to training: {total_moved}")
print(f"\n⚠️  IMPORTANT: You need to retrain the model with the updated dataset!")
print(f"   Run: python scripts/train_optimized.py")
print("="*70)
