"""
Generate sample detection images for thesis defense slides.
"""
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roadsignnet_sal.model_optimized import create_roadsignnet_optimized
from PIL import Image
import torchvision.transforms as T
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
import numpy as np


def decode_predictions(outputs, conf_thresh=0.3, nms_thresh=0.5):
    """Decode raw model outputs to boxes, scores, labels."""
    all_boxes = []
    all_scores = []
    all_labels = []
    
    strides = [8, 16, 32]
    
    for i, output in enumerate(outputs):
        stride = strides[i]
        batch_size, _, h, w = output.shape
        
        # Output format: [batch, 4+1+num_classes, h, w]
        output = output.permute(0, 2, 3, 1)  # [batch, h, w, channels]
        
        # Create grid
        yv, xv = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        grid = torch.stack([xv, yv], dim=-1).float().unsqueeze(0)
        
        # Decode boxes (first 4 channels)
        box_output = output[..., :4]
        xy = (torch.sigmoid(box_output[..., :2]) + grid) * stride
        wh = torch.exp(box_output[..., 2:4]) * stride
        
        x1 = xy[..., 0] - wh[..., 0] / 2
        y1 = xy[..., 1] - wh[..., 1] / 2
        x2 = xy[..., 0] + wh[..., 0] / 2
        y2 = xy[..., 1] + wh[..., 1] / 2
        
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        
        # Objectness (5th channel)
        obj = torch.sigmoid(output[..., 4:5])
        
        # Class probabilities
        cls = torch.sigmoid(output[..., 5:])
        
        # Combine objectness and class probs
        scores = obj * cls
        
        # Flatten
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1, cls.shape[-1])
        
        all_boxes.append(boxes)
        all_scores.append(scores)
    
    # Concatenate all scales
    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    
    # Get max class score and label for each box
    max_scores, labels = scores.max(dim=1)
    
    # Filter by confidence
    mask = max_scores > conf_thresh
    boxes = boxes[mask]
    max_scores = max_scores[mask]
    labels = labels[mask]
    
    # Simple NMS
    if len(boxes) > 0:
        keep = torchvision.ops.nms(boxes, max_scores, nms_thresh)
        boxes = boxes[keep]
        max_scores = max_scores[keep]
        labels = labels[keep]
    
    return boxes, max_scores, labels


def main():
    import torchvision.ops
    
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
    
    strides = [8, 16, 32]
    
    for idx, img_path in enumerate(test_images):
        print(f'Processing image {idx+1}/{len(test_images)}: {os.path.basename(img_path)}')
        
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        
        # Inference
        input_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
        
        # Decode predictions
        # Model output format: list of (cls, box, obj) tuples per scale
        # cls: [B, num_classes*3, H, W] -> reshape needed
        # box: [B, 12, H, W] -> 4 coords * 3 anchors  
        # obj: [B, 3, H, W] -> 3 anchors
        all_boxes = []
        all_scores = []
        
        for i, (cls_out, box_out, obj_out) in enumerate(outputs):
            stride = strides[i]
            batch_size, _, h, w = box_out.shape
            num_anchors = 3
            
            # Move to CPU
            cls_out = cls_out.cpu()  # [B, num_classes*3, H, W]
            box_out = box_out.cpu()  # [B, 12, H, W]
            obj_out = obj_out.cpu()  # [B, 3, H, W]
            
            # Reshape outputs
            cls_out = cls_out.view(batch_size, 43, num_anchors, h, w).permute(0, 2, 3, 4, 1)  # [B, 3, H, W, 43]
            box_out = box_out.view(batch_size, 4, num_anchors, h, w).permute(0, 2, 3, 4, 1)  # [B, 3, H, W, 4]
            obj_out = obj_out.view(batch_size, num_anchors, h, w).unsqueeze(-1)  # [B, 3, H, W, 1]
            
            # Create grid
            yv, xv = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            grid = torch.stack([xv, yv], dim=-1).float()  # [H, W, 2]
            
            # Decode boxes
            xy = (torch.sigmoid(box_out[..., :2]) + grid.unsqueeze(0).unsqueeze(0)) * stride
            wh = torch.exp(torch.clamp(box_out[..., 2:4], max=10)) * stride
            
            x1 = xy[..., 0] - wh[..., 0] / 2
            y1 = xy[..., 1] - wh[..., 1] / 2
            x2 = xy[..., 0] + wh[..., 0] / 2
            y2 = xy[..., 1] + wh[..., 1] / 2
            
            boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # [B, 3, H, W, 4]
            
            # Scores = obj * cls
            obj_prob = torch.sigmoid(obj_out)  # [B, 3, H, W, 1]
            cls_prob = torch.sigmoid(cls_out)  # [B, 3, H, W, 43]
            scores = obj_prob * cls_prob  # [B, 3, H, W, 43]
            
            # Flatten
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1, 43)
            
            all_boxes.append(boxes)
            all_scores.append(scores)
        
        boxes = torch.cat(all_boxes, dim=0)
        scores = torch.cat(all_scores, dim=0)
        
        # Get max class score
        max_scores, labels = scores.max(dim=1)
        
        # Filter by confidence
        conf_thresh = 0.25
        mask = max_scores > conf_thresh
        boxes = boxes[mask]
        max_scores = max_scores[mask]
        labels = labels[mask]
        
        # NMS
        if len(boxes) > 0:
            import torchvision.ops
            keep = torchvision.ops.nms(boxes, max_scores, 0.5)
            boxes = boxes[keep]
            max_scores = max_scores[keep]
            labels = labels[keep]
        
        print(f'  Found {len(boxes)} detections')
        
        # Plot
        ax = axes[idx]
        ax.imshow(img)
        ax.set_title(f'{len(boxes)} detections', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Draw boxes
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        for i, (box, score, label) in enumerate(zip(boxes[:10], max_scores[:10], labels[:10])):
            x1, y1, x2, y2 = box.tolist()
            # Scale to original size
            x1 = x1 * orig_w / 320
            y1 = y1 * orig_h / 320
            x2 = x2 * orig_w / 320
            y2 = y2 * orig_h / 320
            
            # Clamp
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
