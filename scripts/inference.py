#!/usr/bin/env python3
"""
RoadSignNet-SAL Inference Script
Run inference on single image or directory
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
import time

from roadsignnet_sal.model import create_roadsignnet_sal


def preprocess_image(image_path, img_size=640):
    """Preprocess image for inference"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (img_size, img_size))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)
    return img, img_tensor


def run_inference(model, image_path, device, img_size=640):
    """Run inference on single image"""
    original_img, img_tensor = preprocess_image(image_path, img_size)
    img_tensor = img_tensor.to(device)
    
    start_time = time.time()
    with torch.no_grad():
        predictions = model(img_tensor)
    inference_time = (time.time() - start_time) * 1000
    
    return original_img, predictions, inference_time


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    
    print("="*70)
    print("ROADSIGNNET-SAL INFERENCE")
    print("="*70)
    print(f"Device: {device}")
    
    # Load checkpoint first to get config
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Get num_classes from checkpoint config or default to 3
    if 'config' in checkpoint:
        num_classes = checkpoint['config'].get('model', {}).get('num_classes', 3)
    else:
        num_classes = 3
    
    # Load model with correct num_classes
    model = create_roadsignnet_sal(num_classes=num_classes, width_multiplier=1.0).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded: {args.checkpoint}")
    print(f"✓ Number of classes: {num_classes}")
    
    # Get image paths
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    else:
        # Fixed glob patterns for Windows
        image_paths = list(input_path.glob('*.jpg')) + \
                     list(input_path.glob('*.png')) + \
                     list(input_path.glob('*.jpeg')) + \
                     list(input_path.glob('*.JPG')) + \
                     list(input_path.glob('*.PNG'))
    
    if len(image_paths) == 0:
        print(f"❌ No images found in {input_path}")
        return
    
    print(f"✓ Found {len(image_paths)} images")
    
    # Run inference
    total_time = 0
    for img_path in image_paths:
        print(f"\nProcessing: {img_path.name}")
        try:
            original_img, predictions, inference_time = run_inference(model, img_path, device)
            total_time += inference_time
            
            print(f"  Inference time: {inference_time:.2f} ms")
            print(f"  Output shapes:")
            for i, (cls, box, obj) in enumerate(predictions):
                print(f"    Scale {i+1}: cls={cls.shape}, box={box.shape}, obj={obj.shape}")
        except Exception as e:
            print(f"  ❌ Error processing {img_path.name}: {e}")
    
    # Statistics
    if len(image_paths) > 0:
        avg_time = total_time / len(image_paths)
        fps = 1000 / avg_time if avg_time > 0 else 0
        
        print("\n" + "="*70)
        print("INFERENCE STATISTICS")
        print("="*70)
        print(f"Total images:      {len(image_paths)}")
        print(f"Average time:      {avg_time:.2f} ms")
        print(f"FPS:               {fps:.2f}")
        print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Image file or directory')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()
    
    main(args)