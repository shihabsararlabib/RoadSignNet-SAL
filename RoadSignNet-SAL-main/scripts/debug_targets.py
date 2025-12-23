"""
Debug target generation - check if the Gaussian heatmaps are being created correctly
"""

import torch
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roadsignnet_sal.loss_v2 import AnchorFreeLoss


def test_gaussian_generation():
    """Test if Gaussian heatmap generation is working"""
    
    # Create loss function
    loss_fn = AnchorFreeLoss(num_classes=43)
    
    # Create a simple test case: single object at center
    batch_size = 1
    
    # Bounding box: center of 416x416 image, size 100x100
    # In format [x1, y1, x2, y2] normalized to [0, 1]
    cx, cy = 208, 208  # center in pixels
    w, h = 100, 100  # size in pixels
    
    # Convert to normalized x1y1x2y2
    x1, y1 = (cx - w/2) / 416, (cy - h/2) / 416
    x2, y2 = (cx + w/2) / 416, (cy + h/2) / 416
    
    bboxes = torch.tensor([[[x1, y1, x2, y2]]])  # [B, 1, 4]
    labels = torch.tensor([[25]])  # [B, 1] - class 25
    
    # Generate targets
    output_size = (52, 52)  # 416/8 = 52
    heatmap, box_targets, class_targets, mask = loss_fn._generate_heatmap(
        bboxes, labels, output_size
    )
    
    print("="*70)
    print("GAUSSIAN HEATMAP TARGET GENERATION TEST")
    print("="*70)
    print(f"Output size: {output_size}")
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"\n📊 HEATMAP STATISTICS:")
    print(f"  Min: {heatmap.min():.6f}")
    print(f"  Max: {heatmap.max():.6f}")
    print(f"  Mean: {heatmap.mean():.6f}")
    print(f"  Std: {heatmap.std():.6f}")
    print(f"  Pixels > 0.0: {(heatmap > 0.0).sum()}")
    print(f"  Pixels > 0.1: {(heatmap > 0.1).sum()}")
    print(f"  Pixels > 0.5: {(heatmap > 0.5).sum()}")
    print(f"  Pixels > 0.9: {(heatmap > 0.9).sum()}")
    
    print(f"\n📊 MASK STATISTICS:")
    print(f"  Number of positive samples: {mask.sum()}")
    print(f"  Expected: 1 (one object)")
    
    # Find peak location
    max_val = heatmap.max()
    max_idx = (heatmap == max_val).nonzero(as_tuple=False)
    print(f"\n📍 PEAK LOCATION:")
    print(f"  Max value: {max_val:.6f}")
    print(f"  Location: {max_idx[0].tolist()}")
    expected_grid_x = int(cx / 8)
    expected_grid_y = int(cy / 8)
    print(f"  Expected grid location: [0, 0, {expected_grid_y}, {expected_grid_x}]")
    
    # Visualize the heatmap
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Heatmap
    im1 = axes[0].imshow(heatmap[0, 0].cpu().numpy(), cmap='hot', vmin=0, vmax=1)
    axes[0].set_title(f'Target Heatmap\nMax={heatmap.max():.4f}')
    axes[0].axhline(y=expected_grid_y, color='cyan', linestyle='--', linewidth=1, label='Expected center')
    axes[0].axvline(x=expected_grid_x, color='cyan', linestyle='--', linewidth=1)
    plt.colorbar(im1, ax=axes[0])
    
    # Box target mask
    im2 = axes[1].imshow(mask[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Positive Sample Mask\nCount={mask.sum()}')
    plt.colorbar(im2, ax=axes[1])
    
    # Class target
    im3 = axes[2].imshow(class_targets[0, 25].cpu().numpy(), cmap='hot', vmin=0, vmax=1)
    axes[2].set_title(f'Class Target (class 25)\nMax={class_targets[0, 25].max():.4f}')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('debug_target_generation.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization: debug_target_generation.png")
    
    # Test with multiple objects
    print(f"\n{'='*70}")
    print("TESTING MULTIPLE OBJECTS")
    print(f"{'='*70}")
    
    # Two objects: one at (100, 100) size 50x50, another at (300, 300) size 80x80
    obj1_cx, obj1_cy, obj1_w, obj1_h = 100, 100, 50, 50
    obj2_cx, obj2_cy, obj2_w, obj2_h = 300, 300, 80, 80
    
    obj1_x1 = (obj1_cx - obj1_w/2) / 416
    obj1_y1 = (obj1_cy - obj1_h/2) / 416
    obj1_x2 = (obj1_cx + obj1_w/2) / 416
    obj1_y2 = (obj1_cy + obj1_h/2) / 416
    
    obj2_x1 = (obj2_cx - obj2_w/2) / 416
    obj2_y1 = (obj2_cy - obj2_h/2) / 416
    obj2_x2 = (obj2_cx + obj2_w/2) / 416
    obj2_y2 = (obj2_cy + obj2_h/2) / 416
    
    bboxes_multi = torch.tensor([[[obj1_x1, obj1_y1, obj1_x2, obj1_y2],
                                   [obj2_x1, obj2_y1, obj2_x2, obj2_y2]]])
    labels_multi = torch.tensor([[10, 25]])
    
    heatmap_multi, box_multi, cls_multi, mask_multi = loss_fn._generate_heatmap(
        bboxes_multi, labels_multi, output_size
    )
    
    print(f"Heatmap - Max: {heatmap_multi.max():.6f}, Pixels > 0.5: {(heatmap_multi > 0.5).sum()}")
    print(f"Mask - Positive samples: {mask_multi.sum()} (expected: 2)")
    
    # Visualize multi-object
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(heatmap_multi[0, 0].cpu().numpy(), cmap='hot', vmin=0, vmax=1)
    ax.set_title(f'Multi-Object Heatmap\nMax={heatmap_multi.max():.4f}')
    
    # Mark expected centers
    exp1_gx, exp1_gy = int(obj1_cx / 8), int(obj1_cy / 8)
    exp2_gx, exp2_gy = int(obj2_cx / 8), int(obj2_cy / 8)
    ax.scatter([exp1_gx], [exp1_gy], c='cyan', marker='+', s=200, label='Obj1 expected')
    ax.scatter([exp2_gx], [exp2_gy], c='lime', marker='+', s=200, label='Obj2 expected')
    ax.legend()
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('debug_multi_object_targets.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization: debug_multi_object_targets.png")
    
    print(f"\n{'='*70}")
    print("Target generation test complete!")
    print(f"{'='*70}")


if __name__ == '__main__':
    test_gaussian_generation()
