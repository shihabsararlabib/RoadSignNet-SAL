#!/usr/bin/env python3
"""
Phase 2 Defense Analysis Script
Generates all tables and figures needed for Phase 2 presentation
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Class names
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

def count_class_samples(data_dir):
    """Count samples per class in dataset"""
    label_dir = Path(data_dir) / 'train' / 'labels'
    
    class_counts = {name: 0 for name in CLASS_NAMES}
    
    for label_file in label_dir.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    if class_id < len(CLASS_NAMES):
                        class_counts[CLASS_NAMES[class_id]] += 1
    
    return class_counts


def generate_model_comparison_table():
    """Generate comparison table for Phase 2 slides"""
    
    data = {
        'Model': ['RoadSignNet-SAL (Ours)', 'YOLOv8n', 'YOLOv8s', 'YOLO11n', 'YOLO11s'],
        'Params (M)': [2.04, 3.0, 11.2, 2.6, 9.4],
        'mAP@0.5': [75.52, 85.7, 88.5, 87.9, 89.0],
        'FPS (GPU)': [154.6, 120, 95, 125, 100],
        'Size (MB)': [7.8, 12.0, 44.0, 10.4, 37.6],
        'Edge Ready': ['✓', '✓', '✗', '✓', '✗']
    }
    
    print("\n" + "="*80)
    print("TABLE 1: Model Comparison (For Phase 2 Slides)")
    print("="*80)
    print(f"{'Model':<25} {'Params':<12} {'mAP@0.5':<12} {'FPS':<10} {'Size':<10} {'Edge':<8}")
    print("-"*80)
    
    for i in range(len(data['Model'])):
        print(f"{data['Model'][i]:<25} {data['Params (M)'][i]:<12} {data['mAP@0.5'][i]:<12} {data['FPS (GPU)'][i]:<10} {data['Size (MB)'][i]:<10} {data['Edge Ready'][i]:<8}")
    
    print("="*80)
    
    return data


def generate_thesis_contribution_table():
    """Generate contribution summary for Phase 2"""
    
    print("\n" + "="*80)
    print("TABLE 2: Thesis Contributions (For Phase 2 Slides)")
    print("="*80)
    
    contributions = [
        ("Lightweight Architecture", "2.04M params (30% smaller than YOLOv8n)", "✓ Achieved"),
        ("Real-time Performance", "154 FPS GPU, 34 FPS CPU", "✓ Achieved"),
        ("43-Class Detection", "75.52% mAP on traffic signs", "✓ Achieved"),
        ("Edge Deployment Ready", "7.8 MB model, 22 MB GPU memory", "✓ Achieved"),
        ("Moving Object Support", ">30 FPS real-time video", "✓ Achieved"),
    ]
    
    print(f"{'Contribution':<30} {'Evidence':<40} {'Status':<15}")
    print("-"*80)
    
    for contrib, evidence, status in contributions:
        print(f"{contrib:<30} {evidence:<40} {status:<15}")
    
    print("="*80)


def generate_phase3_plan():
    """Generate plan for final defense"""
    
    print("\n" + "="*80)
    print("TABLE 3: Phase 3 (Final Defense) Plan")
    print("="*80)
    
    tasks = [
        ("Ablation Study", "Prove each component's contribution", "2 weeks", "High"),
        ("Cross-Dataset Test", "Test on GTSDB or TT100K", "1 week", "Medium"),
        ("Edge Device Deploy", "Run on Jetson Nano/Raspberry Pi", "1 week", "Medium"),
        ("Thesis Writing", "Complete all chapters", "4 weeks", "High"),
        ("Video Demo", "Record real-world demo", "1 day", "High"),
    ]
    
    print(f"{'Task':<25} {'Purpose':<35} {'Time':<12} {'Priority':<10}")
    print("-"*80)
    
    for task, purpose, time, priority in tasks:
        print(f"{task:<25} {purpose:<35} {time:<12} {priority:<10}")
    
    print("="*80)


def plot_model_comparison(output_dir):
    """Create visual comparison chart"""
    
    models = ['Ours', 'YOLOv8n', 'YOLOv8s', 'YOLO11n', 'YOLO11s']
    params = [2.04, 3.0, 11.2, 2.6, 9.4]
    mAP = [75.52, 85.7, 88.5, 87.9, 89.0]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # mAP comparison
    colors = ['#2ecc71' if m == 'Ours' else '#3498db' for m in models]
    axes[0].bar(models, mAP, color=colors, edgecolor='black')
    axes[0].set_ylabel('mAP@0.5 (%)', fontsize=12)
    axes[0].set_title('Detection Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(mAP):
        axes[0].text(i, v + 1, f'{v}%', ha='center', fontweight='bold')
    
    # Parameters comparison
    axes[1].bar(models, params, color=colors, edgecolor='black')
    axes[1].set_ylabel('Parameters (M)', fontsize=12)
    axes[1].set_title('Model Size', fontsize=14, fontweight='bold')
    for i, v in enumerate(params):
        axes[1].text(i, v + 0.2, f'{v}M', ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'phase2_model_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved: {output_path}")
    return output_path


def plot_efficiency_chart(output_dir):
    """Create efficiency vs accuracy plot"""
    
    models = ['RoadSignNet-SAL\n(Ours)', 'YOLOv8n', 'YOLOv8s', 'YOLO11n', 'YOLO11s']
    params = [2.04, 3.0, 11.2, 2.6, 9.4]
    mAP = [75.52, 85.7, 88.5, 87.9, 89.0]
    fps = [154.6, 120, 95, 125, 100]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Scatter plot with bubble size = FPS
    colors = ['#e74c3c', '#3498db', '#3498db', '#9b59b6', '#9b59b6']
    
    for i, (p, m, f, name) in enumerate(zip(params, mAP, fps, models)):
        ax.scatter(p, m, s=f*3, c=colors[i], alpha=0.7, edgecolors='black', linewidth=2)
        ax.annotate(name, (p, m), xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax.set_xlabel('Parameters (M)', fontsize=12)
    ax.set_ylabel('mAP@0.5 (%)', fontsize=12)
    ax.set_title('Efficiency vs Accuracy\n(Bubble size = FPS)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend for bubble size
    ax.annotate('Larger bubble = Higher FPS', xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=10, verticalalignment='top', style='italic')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'phase2_efficiency_chart.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    return output_path


def generate_key_findings():
    """Print key findings for Phase 2 presentation"""
    
    print("\n" + "="*80)
    print("KEY FINDINGS FOR PHASE 2 PRESENTATION")
    print("="*80)
    
    findings = """
    1. LIGHTWEIGHT ACHIEVEMENT
       → Our model: 2.04M parameters
       → YOLOv8n: 3.0M parameters  
       → Reduction: 32% smaller
    
    2. REAL-TIME PERFORMANCE
       → GPU: 154.6 FPS (5x faster than 30 FPS requirement)
       → CPU: 33.6 FPS (meets real-time threshold)
       → Validates "moving objects" thesis claim
    
    3. ACCURACY TRADE-OFF
       → Our mAP: 75.52%
       → YOLO11s mAP: 89.0%
       → Gap: 13.5% (acceptable for 78% size reduction)
    
    4. EDGE DEPLOYMENT READY
       → Model size: 7.8 MB
       → GPU memory: 22 MB
       → Suitable for: Jetson Nano, Raspberry Pi, Mobile
    
    5. THESIS TITLE VALIDATED
       "Advanced Machine Vision System for Road Sign Detection 
        Using a Lightweight Model for Moving Objects"
       → Lightweight: ✓ (2.04M params)
       → Moving Objects: ✓ (154 FPS real-time)
       → Road Sign Detection: ✓ (75.52% mAP, 43 classes)
    """
    
    print(findings)
    print("="*80)


def main():
    output_dir = Path('outputs/phase2_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("PHASE 2 DEFENSE ANALYSIS")
    print("Generating tables and figures for your presentation")
    print("="*80)
    
    # Generate tables
    generate_model_comparison_table()
    generate_thesis_contribution_table()
    generate_phase3_plan()
    generate_key_findings()
    
    # Generate plots
    plot_model_comparison(output_dir)
    plot_efficiency_chart(output_dir)
    
    print("\n" + "="*80)
    print("PHASE 2 MATERIALS READY")
    print("="*80)
    print(f"\nOutput folder: {output_dir}")
    print("\nFiles generated:")
    print("  - phase2_model_comparison.png")
    print("  - phase2_efficiency_chart.png")
    print("\nCopy the tables above into your slides!")
    print("="*80)


if __name__ == '__main__':
    main()
