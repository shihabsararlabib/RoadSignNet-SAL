#!/usr/bin/env python3
"""
Advanced Machine Vision System for Road Sign Detection
Real-time video inference demonstrating "Moving Objects" capability

This script validates the thesis claim of lightweight model for moving objects.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from collections import deque

# Class names for 43 traffic signs
CLASS_NAMES = [
    'bicycle', 'bus_stop', 'children', 'crosswalk', 'do_not_stop',
    'do_not_turn_left', 'do_not_turn_right', 'do_not_u_turn', 'enter_left_lane',
    'give_way', 'green_light', 'left_lane_enter', 'left_turn', 'narrow_road',
    'no_entry', 'no_overtaking', 'no_parking', 'no_stop', 'no_waiting',
    'parking', 'railway_crossing', 'red_light', 'refueling', 'right_turn',
    'road_main', 'road_work', 'school_nearby', 'speed_bump', 'speed_limit_100',
    'speed_limit_120', 'speed_limit_20', 'speed_limit_30', 'speed_limit_40',
    'speed_limit_50', 'speed_limit_60', 'speed_limit_70', 'speed_limit_80',
    'stop', 't_intersection_l', 'truck', 'u_turn', 'warning', 'yellow_light'
]

# Colors for different sign categories
COLORS = {
    'speed': (0, 255, 0),       # Green
    'warning': (0, 165, 255),   # Orange
    'prohibition': (0, 0, 255), # Red
    'information': (255, 255, 0), # Cyan
    'traffic_light': (255, 0, 255), # Magenta
    'default': (255, 255, 255)  # White
}

def get_color(class_name):
    """Get color based on sign category"""
    if 'speed_limit' in class_name:
        return COLORS['speed']
    elif class_name in ['warning', 'children', 'road_work', 'narrow_road', 'speed_bump', 'railway_crossing', 'school_nearby']:
        return COLORS['warning']
    elif class_name in ['no_entry', 'no_parking', 'no_stop', 'no_waiting', 'no_overtaking', 'do_not_turn_left', 'do_not_turn_right', 'do_not_u_turn', 'do_not_stop']:
        return COLORS['prohibition']
    elif class_name in ['green_light', 'red_light', 'yellow_light']:
        return COLORS['traffic_light']
    else:
        return COLORS['information']


class RoadSignDetectionSystem:
    """
    Advanced Machine Vision System for Road Sign Detection
    Thesis: Lightweight Model for Moving Objects
    """
    
    def __init__(self, checkpoint_path, device='cuda', img_size=416, conf_thresh=0.25, iou_thresh=0.45):
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.detection_count = 0
        self.frame_count = 0
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        
        print(f"✓ System initialized on {self.device}")
        print(f"✓ Image size: {img_size}x{img_size}")
        print(f"✓ Confidence threshold: {conf_thresh}")
        
    def _load_model(self, checkpoint_path):
        """Load the lightweight RoadSignNet-SAL model"""
        from roadsignnet_sal.model_optimized import create_optimized_roadsignnet
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Get config from checkpoint
        if 'config' in checkpoint:
            num_classes = checkpoint['config'].get('model', {}).get('num_classes', 43)
            width_mult = checkpoint['config'].get('model', {}).get('width_multiplier', 1.35)
        else:
            num_classes = 43
            width_mult = 1.35
        
        model = create_optimized_roadsignnet(
            num_classes=num_classes,
            width_mult=width_mult
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"✓ Model loaded: {params:.2f}M parameters (Lightweight!)")
        
        return model
    
    def preprocess(self, frame):
        """Preprocess frame for inference"""
        # Resize
        img = cv2.resize(frame, (self.img_size, self.img_size))
        
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize (ImageNet stats)
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # HWC to CHW, add batch dimension
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        
        return img.to(self.device)
    
    def postprocess(self, outputs, orig_h, orig_w):
        """Post-process model outputs to bounding boxes"""
        detections = []
        
        # Process multi-scale outputs
        for scale_idx, (cls_out, box_out, obj_out) in enumerate(outputs):
            B, _, H, W = cls_out.shape
            
            # Get objectness scores
            obj_scores = torch.sigmoid(obj_out).squeeze(0)  # [num_anchors, H, W]
            
            # Get class predictions
            num_classes = cls_out.shape[1] // obj_out.shape[1]
            num_anchors = obj_out.shape[1]
            
            cls_out = cls_out.view(B, num_anchors, num_classes, H, W)
            box_out = box_out.view(B, num_anchors, 4, H, W)
            
            for anchor in range(num_anchors):
                obj_score = obj_scores[anchor]  # [H, W]
                
                # Find high confidence locations
                mask = obj_score > self.conf_thresh
                if not mask.any():
                    continue
                
                # Get indices
                y_indices, x_indices = torch.where(mask)
                
                for y, x in zip(y_indices, x_indices):
                    obj_conf = obj_score[y, x].item()
                    
                    # Get class scores
                    cls_scores = torch.sigmoid(cls_out[0, anchor, :, y, x])
                    cls_conf, cls_id = cls_scores.max(0)
                    cls_conf = cls_conf.item()
                    cls_id = cls_id.item()
                    
                    # Combined confidence
                    confidence = obj_conf * cls_conf
                    
                    if confidence < self.conf_thresh:
                        continue
                    
                    # Get box predictions
                    box = box_out[0, anchor, :, y, x]
                    
                    # Decode box (assuming center format)
                    stride = self.img_size // H
                    cx = (x.item() + torch.sigmoid(box[0]).item()) * stride
                    cy = (y.item() + torch.sigmoid(box[1]).item()) * stride
                    w = torch.exp(box[2]).item() * stride * 2
                    h = torch.exp(box[3]).item() * stride * 2
                    
                    # Convert to corner format and scale to original image
                    x1 = int((cx - w/2) * orig_w / self.img_size)
                    y1 = int((cy - h/2) * orig_h / self.img_size)
                    x2 = int((cx + w/2) * orig_w / self.img_size)
                    y2 = int((cy + h/2) * orig_h / self.img_size)
                    
                    # Clip to image bounds
                    x1 = max(0, min(orig_w, x1))
                    y1 = max(0, min(orig_h, y1))
                    x2 = max(0, min(orig_w, x2))
                    y2 = max(0, min(orig_h, y2))
                    
                    if x2 > x1 and y2 > y1:
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'class_id': cls_id,
                            'class_name': CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f'class_{cls_id}',
                            'confidence': confidence
                        })
        
        # Apply NMS
        if detections:
            detections = self._nms(detections)
        
        return detections
    
    def _nms(self, detections, iou_thresh=0.45):
        """Non-Maximum Suppression"""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            detections = [
                d for d in detections
                if self._iou(best['bbox'], d['bbox']) < iou_thresh
            ]
        
        return keep
    
    def _iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def detect(self, frame):
        """Run detection on a single frame"""
        orig_h, orig_w = frame.shape[:2]
        
        # Preprocess
        img_tensor = self.preprocess(frame)
        
        # Inference
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(img_tensor)
        inference_time = (time.time() - start_time) * 1000
        
        # Postprocess
        detections = self.postprocess(outputs, orig_h, orig_w)
        
        # Track FPS
        fps = 1000 / inference_time if inference_time > 0 else 0
        self.fps_history.append(fps)
        self.frame_count += 1
        self.detection_count += len(detections)
        
        return detections, inference_time
    
    def draw_detections(self, frame, detections, inference_time):
        """Draw bounding boxes and info on frame"""
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            color = get_color(class_name)
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{class_name}: {confidence:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw system info
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        info_lines = [
            f"RoadSignNet-SAL | Lightweight Model for Moving Objects",
            f"FPS: {avg_fps:.1f} | Latency: {inference_time:.1f}ms",
            f"Detections: {len(detections)} | Device: {self.device}",
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, 25 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw legend
        y_offset = frame.shape[0] - 100
        legend_items = [
            ("Speed Limit", COLORS['speed']),
            ("Warning", COLORS['warning']),
            ("Prohibition", COLORS['prohibition']),
            ("Traffic Light", COLORS['traffic_light']),
        ]
        
        for i, (name, color) in enumerate(legend_items):
            cv2.rectangle(frame, (10, y_offset + i*20), (25, y_offset + i*20 + 15), color, -1)
            cv2.putText(frame, name, (30, y_offset + i*20 + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def run_video(self, source, output_path=None, show=True):
        """
        Run detection on video source (webcam or file)
        
        Args:
            source: 0 for webcam, or path to video file
            output_path: Path to save output video
            show: Whether to display live
        """
        # Open video source
        if isinstance(source, int) or source.isdigit():
            cap = cv2.VideoCapture(int(source))
            source_name = f"Webcam {source}"
        else:
            cap = cv2.VideoCapture(source)
            source_name = Path(source).name
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\n{'='*60}")
        print(f"ADVANCED MACHINE VISION SYSTEM")
        print(f"Road Sign Detection for Moving Objects")
        print(f"{'='*60}")
        print(f"Source: {source_name}")
        print(f"Resolution: {width}x{height} @ {fps}fps")
        print(f"Press 'q' to quit")
        print(f"{'='*60}\n")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Recording to: {output_path}")
        
        # Process frames
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect
                detections, inference_time = self.detect(frame)
                
                # Draw results
                frame = self.draw_detections(frame, detections, inference_time)
                
                # Save frame
                if writer:
                    writer.write(frame)
                
                # Display
                if show:
                    cv2.imshow('RoadSignNet-SAL - Moving Object Detection', frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
        
        # Print statistics
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        print(f"\n{'='*60}")
        print(f"SESSION STATISTICS")
        print(f"{'='*60}")
        print(f"Total frames processed: {self.frame_count}")
        print(f"Total detections: {self.detection_count}")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Average detections/frame: {self.detection_count/max(1,self.frame_count):.2f}")
        print(f"{'='*60}")
        
        return {
            'frames': self.frame_count,
            'detections': self.detection_count,
            'avg_fps': avg_fps
        }
    
    def benchmark(self, num_iterations=100):
        """Benchmark inference speed with dummy input"""
        print(f"\n{'='*60}")
        print(f"BENCHMARK: Lightweight Model Performance")
        print(f"{'='*60}")
        
        # Create dummy input
        dummy = torch.randn(1, 3, self.img_size, self.img_size).to(self.device)
        
        # Warmup
        print("Warming up...")
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(dummy)
        
        # Benchmark
        print(f"Running {num_iterations} iterations...")
        times = []
        
        for _ in range(num_iterations):
            start = time.time()
            with torch.no_grad():
                _ = self.model(dummy)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.time() - start) * 1000)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1000 / avg_time
        
        print(f"\nResults ({self.device}):")
        print(f"  Average latency: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"  Min latency: {min_time:.2f} ms")
        print(f"  Max latency: {max_time:.2f} ms")
        print(f"  Average FPS: {fps:.2f}")
        print(f"  Real-time capable (>30 FPS): {'✓ YES' if fps > 30 else '✗ NO'}")
        print(f"{'='*60}")
        
        return {
            'device': str(self.device),
            'avg_latency_ms': avg_time,
            'std_latency_ms': std_time,
            'fps': fps,
            'real_time': fps > 30
        }


def main():
    parser = argparse.ArgumentParser(
        description='Advanced Machine Vision System for Road Sign Detection'
    )
    parser.add_argument('--checkpoint', type=str, 
                       default='outputs/checkpoints/best_model_v5_finetuned.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source: 0 for webcam, or path to video file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path (optional)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'])
    parser.add_argument('--img-size', type=int, default=416,
                       help='Input image size')
    parser.add_argument('--conf-thresh', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark instead of video')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable display (for headless systems)')
    
    args = parser.parse_args()
    
    # Initialize system
    system = RoadSignDetectionSystem(
        checkpoint_path=args.checkpoint,
        device=args.device,
        img_size=args.img_size,
        conf_thresh=args.conf_thresh
    )
    
    if args.benchmark:
        # Run benchmark
        system.benchmark()
    else:
        # Run video detection
        system.run_video(
            source=args.source,
            output_path=args.output,
            show=not args.no_display
        )


if __name__ == '__main__':
    main()
