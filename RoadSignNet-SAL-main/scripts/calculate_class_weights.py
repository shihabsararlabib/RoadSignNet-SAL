"""
Add class weights to loss function to handle imbalanced classes
This can improve performance on underrepresented classes
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import numpy as np
from pathlib import Path
from collections import defaultdict

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class_names = config['model']['class_names']
num_classes = config['model']['num_classes']

print("="*70)
print("CALCULATING CLASS WEIGHTS FOR LOSS FUNCTION")
print("="*70)

def count_class_samples(label_dir):
    """Count samples per class"""
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

# Count training samples
train_counts = count_class_samples(config['data']['train_label_dir'])

# Calculate class weights
print("\n📊 Training Sample Distribution:")
print("-"*70)
print(f"{'Class ID':<10} {'Class Name':<20} {'Count':<10} {'Weight':<10}")
print("-"*70)

class_weights = []
total_samples = sum(train_counts.values())

for class_id in range(num_classes):
    count = train_counts.get(class_id, 0)
    class_name = class_names[class_id]
    
    # Calculate weight (inverse frequency with smoothing)
    if count == 0:
        # For classes with no samples, assign very high weight
        weight = 10.0
    else:
        # Effective number of samples (from paper: Class-Balanced Loss)
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, count)
        weight = (1.0 - beta) / effective_num
    
    class_weights.append(weight)
    print(f"{class_id:<10} {class_name:<20} {count:<10} {weight:<10.4f}")

# Normalize weights
class_weights = np.array(class_weights)
class_weights = class_weights / class_weights.mean()

print("\n" + "="*70)
print("NORMALIZED CLASS WEIGHTS")
print("="*70)
print("These weights can be used in the loss function to handle imbalance.")
print(f"\nclass_weights = {list(class_weights.round(4))}")

# Highlight problem classes
print("\n🔍 PROBLEM CLASSES (high weights):")
print("-"*70)
problem_classes = [(i, class_names[i], class_weights[i], train_counts.get(i, 0)) 
                   for i in range(num_classes) if class_weights[i] > 3.0]
problem_classes.sort(key=lambda x: x[2], reverse=True)

for class_id, class_name, weight, count in problem_classes:
    print(f"  {class_name:<20}: weight={weight:.2f}, train_samples={count}")

# Save weights to file
weights_str = "[\n    " + ",\n    ".join([f"{w:.6f}" for w in class_weights]) + "\n]"
weights_code = f"""
# Class weights for handling imbalanced dataset
# Generated automatically - add to loss.py

import torch

CLASS_WEIGHTS = torch.tensor({weights_str}, dtype=torch.float32)

# Use in loss function:
# cls_loss = F.binary_cross_entropy_with_logits(
#     cls_pred, cls_target, 
#     weight=CLASS_WEIGHTS[target_classes].to(device)
# )
"""

with open('class_weights.py', 'w') as f:
    f.write(weights_code)

print("\n✅ Class weights saved to: class_weights.py")
print("="*70)
