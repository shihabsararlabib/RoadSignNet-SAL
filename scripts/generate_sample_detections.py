"""
Generate sample detection images for thesis defense slides.
Uses the proper DetectionDecoder from the loss module.
"""
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roadsignnet_sal.model_optimized import create_roadsignnet_optimized
from roadsignnet_sal.loss import DetectionDecoder
from PIL import Image
import torchvision.transforms as T
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
import numpy as np


def main():
    # Load class names
    with open('data/data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    class_names = data_config['names']
    
    # Create and load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_roadsignnet_optimized(num_classes=43, width_multiplier=1.35)
    checkpoint = torch.load('outputs/checkpoints/best_model_v5_finetuned.pth', 
                           map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f'Model loaded on {device}')
    
    # Create decoder (same as evaluation script)
    decoder = DetectionDecoder(
        num_classes=43,
        conf_thresh=0.3,  # Good balance for visualization
        iou_thresh=0.45,
        img_size=320
    )
    
    # Transform
    transform = T.Compose([
        T.Resize((320, 320)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get test images - select diverse ones
    test_images = glob.glob('data/test/images/*.jpg')
    # Pick 6 evenly spaced images
    if len(test_images) > 6:
        indices = np.linspace(0, len(test_images)-1, 6, dtype=int)
        test_images = [test_images[i] for i in indices]
    
    print(f'Processing {len(test_images)} test images')
    
    # Create output dir
    os.makedirs('outputs/sample_detections', exist_ok=True)
    
    # Process images
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, img_path in enumerate(test_images):
        print(f'Processing image {idx+1}/{len(test_images)}: {os.path.basename(img_path)}')
        
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        
        # Inference
        input_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
        
        # Decode predictions using proper decoder
        boxes, scores, labels = decoder.decode(outputs)
        
        # Move to CPU
        boxes = boxes.cpu()
        scores = scores.cpu()
        labels = labels.cpu()
        
        print(f'  Found {len(boxes)} detections')
        
        # Plot
        ax = axes[idx]
        ax.imshow(img)
        ax.set_title(f'{len(boxes)} detections', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Draw boxes (limit to top 10)
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        for i, (box, score, label) in enumerate(zip(boxes[:10], scores[:10], labels[:10])):
            x1, y1, x2, y2 = box.tolist()
            # Scale to original size (model uses 320x320)
            x1 = x1 * orig_w / 320
            y1 = y1 * orig_h / 320
            x2 = x2 * orig_w / 320
            y2 = y2 * orig_h / 320
            
            # Clamp to image bounds
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            
            color = colors[label.item() % 20]
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                      linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            class_name = class_names[label.item()] if label.item() < len(class_names) else f'cls_{label.item()}'
            # Shorten long names
            if len(class_name) > 15:
                class_name = class_name[:12] + '...'
            ax.text(x1, y1-3, f'{class_name}: {score:.2f}', 
                    color='white', fontsize=7, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.85))
    
    plt.suptitle('RoadSignNet-SAL: Sample Detections (mAP@0.5: 75.52%)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/sample_detections/sample_detections_grid.png', dpi=150, bbox_inches='tight')
    print('\n✓ Saved: outputs/sample_detections/sample_detections_grid.png')
    plt.close()
    
    print('\nDone!')


if __name__ == '__main__':
    main()
