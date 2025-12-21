"""
Debug script to visualize model predictions and understand why detection fails
"""

import os
import sys
import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roadsignnet_sal.model_v2 import create_roadsignnet_v2
from roadsignnet_sal.dataset import RoadSignDataset
from roadsignnet_sal.loss_v2 import AnchorFreeDecoder


def load_config(config_path='config/config.yaml'):
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def visualize_predictions(model, dataset, device, num_samples=3):
    """
    Visualize raw model outputs to understand what's happening
    """
    model.eval()
    
    for i in range(min(num_samples, len(dataset))):
        # Get sample
        img, bbox, label = dataset[i]
        img = img.unsqueeze(0).to(device)
        
        # Get predictions
        with torch.no_grad():
            pred_heatmap, pred_boxes, pred_classes = model(img)
        
        # Move to CPU and apply sigmoid
        heatmap = torch.sigmoid(pred_heatmap[0, 0]).cpu().numpy()
        classes = torch.sigmoid(pred_classes[0]).cpu().numpy()  # [num_classes, H, W]
        
        print(f"\n{'='*70}")
        print(f"Sample {i+1}")
        print(f"{'='*70}")
        print(f"Image shape: {img.shape}")
        print(f"Heatmap shape: {heatmap.shape}")
        print(f"Classes shape: {classes.shape}")
        
        # Heatmap statistics
        print(f"\n📊 HEATMAP STATISTICS:")
        print(f"  Min: {heatmap.min():.6f}")
        print(f"  Max: {heatmap.max():.6f}")
        print(f"  Mean: {heatmap.mean():.6f}")
        print(f"  Std: {heatmap.std():.6f}")
        print(f"  Pixels > 0.1: {(heatmap > 0.1).sum()}")
        print(f"  Pixels > 0.3: {(heatmap > 0.3).sum()}")
        print(f"  Pixels > 0.5: {(heatmap > 0.5).sum()}")
        
        # Class predictions statistics
        max_class_probs = classes.max(axis=0)  # Max probability across classes at each location
        print(f"\n📊 CLASS PREDICTION STATISTICS:")
        print(f"  Min: {classes.min():.6f}")
        print(f"  Max: {classes.max():.6f}")
        print(f"  Mean: {classes.mean():.6f}")
        print(f"  Max per location - Min: {max_class_probs.min():.6f}")
        print(f"  Max per location - Max: {max_class_probs.max():.6f}")
        print(f"  Max per location - Mean: {max_class_probs.mean():.6f}")
        print(f"  Locations with max_cls > 0.5: {(max_class_probs > 0.5).sum()}")
        
        # Ground truth
        print(f"\n📌 GROUND TRUTH:")
        print(f"  Number of objects: {len(bbox)}")
        if len(bbox) > 0:
            for j, (box, cls) in enumerate(zip(bbox, label)):
                if box[0] < 0:  # padding
                    continue
                print(f"  Object {j+1}: class={cls}, box={box}")
        
        # Test decoder with different thresholds
        print(f"\n🔍 DECODER TESTS:")
        for thresh in [0.01, 0.05, 0.1, 0.2, 0.3]:
            decoder = AnchorFreeDecoder(conf_thresh=thresh, nms_thresh=0.45)
            boxes, scores, cls_ids = decoder.decode((pred_heatmap, pred_boxes, pred_classes))
            print(f"  Threshold {thresh:.2f}: {len(boxes)} detections")
            if len(boxes) > 0:
                print(f"    Score range: [{scores.min():.4f}, {scores.max():.4f}]")
                print(f"    Classes: {cls_ids.cpu().numpy()}")
        
        # Visualize heatmap
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        img_np = img[0].cpu().permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        axes[0].imshow(img_np)
        axes[0].set_title(f'Original Image')
        axes[0].axis('off')
        
        # Heatmap
        im1 = axes[1].imshow(heatmap, cmap='hot', vmin=0, vmax=1)
        axes[1].set_title(f'Predicted Heatmap\nMax={heatmap.max():.4f}')
        plt.colorbar(im1, ax=axes[1])
        
        # Max class probability
        im2 = axes[2].imshow(max_class_probs, cmap='hot', vmin=0, vmax=1)
        axes[2].set_title(f'Max Class Probability\nMax={max_class_probs.max():.4f}')
        plt.colorbar(im2, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(f'debug_sample_{i+1}.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved visualization: debug_sample_{i+1}.png")
        plt.close()


def main():
    # Load config
    config = load_config()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load dataset (test set)
    test_dataset = RoadSignDataset(
        img_dir='data/test/images',
        label_dir='data/test/labels',
        split='test',
        img_size=416,
        augment=False
    )
    
    print(f"Test dataset: {len(test_dataset)} samples\n")
    
    # Load model
    checkpoint_path = 'outputs/checkpoints/checkpoint_v2_epoch_100.pth'
    
    # Create model
    model = create_roadsignnet_v2(
        num_classes=43,
        enhanced=True
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"✓ Model loaded from: {checkpoint_path}")
    print(f"✓ Trained for {checkpoint['epoch']} epochs\n")
    
    # Visualize predictions
    visualize_predictions(model, test_dataset, device, num_samples=5)
    
    print(f"\n{'='*70}")
    print("Debugging complete! Check the debug_sample_*.png files.")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
