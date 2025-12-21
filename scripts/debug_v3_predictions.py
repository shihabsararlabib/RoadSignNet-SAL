"""
Debug V3 predictions to understand why decoder isn't working
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from roadsignnet_sal.model_v3 import create_roadsignnet_v3
from roadsignnet_sal.loss_v3 import V3Decoder
from roadsignnet_sal.dataset import RoadSignDataset
import numpy as np

device = torch.device('cuda')

# Load model
model = create_roadsignnet_v3(num_classes=43, width_mult=0.25, depth_mult=0.33).to(device)
checkpoint = torch.load('outputs/checkpoints/best_model_v3.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Create decoder
decoder = V3Decoder(num_classes=43, conf_thresh=0.01, nms_thresh=0.45, max_detections=300, strides=[8, 16, 32])

# Load test dataset
dataset = RoadSignDataset('data/test/images', 'data/test/labels', img_size=416, augment=False, split='test')

print("="*70)
print("V3 PREDICTION ANALYSIS")
print("="*70)

# Check a few samples
with torch.no_grad():
    for idx in [0, 100, 500, 1000]:
        img, bboxes, labels = dataset[idx]
        img_batch = img.unsqueeze(0).to(device)
        
        # Get predictions from model
        preds = model(img_batch)  # Returns list of 3 scales, each is (cls, reg, obj)
        
        print(f"\n--- Sample {idx} ---")
        print(f"GT: {(labels >= 0).sum().item()} objects")
        
        # Check each scale
        for scale_idx, ((cls_pred, box_pred, obj_pred), stride) in enumerate(zip(preds, [8, 16, 32])):
            B, num_classes, H, W = cls_pred.shape
            
            # Get max confidence
            max_cls_conf = cls_pred.max().item()
            max_obj_conf = torch.sigmoid(obj_pred).max().item()
            mean_obj_conf = torch.sigmoid(obj_pred).mean().item()
            
            print(f"  Scale {scale_idx+1} ({H}x{W}):")
            print(f"    Max cls conf: {max_cls_conf:.4f}")
            print(f"    Max obj conf: {max_obj_conf:.4f}")
            print(f"    Mean obj conf: {mean_obj_conf:.4f}")
            
            # Count how many cells have high objectness
            high_obj = (torch.sigmoid(obj_pred) > 0.5).sum().item()
            print(f"    High obj cells (>0.5): {high_obj}")
        
        # Try decoding
        boxes, scores, classes = decoder.decode(preds)
        print(f"  Decoded: {len(boxes)} detections")
        
        if len(boxes) > 0:
            print(f"  Top 3 detections:")
            for i in range(min(3, len(boxes))):
                print(f"    {i+1}. Class={int(classes[i])}, Score={scores[i]:.4f}, Box={boxes[i]}")

print("\n" + "="*70)
print("KEY INSIGHTS:")
print("  - If obj confidences are all low, model isn't learning to predict objects")
print("  - If cls confidences are high but no detections, decoder threshold issue")
print("  - If boxes are decoded but mAP=0, IoU/NMS issue")
print("="*70)
