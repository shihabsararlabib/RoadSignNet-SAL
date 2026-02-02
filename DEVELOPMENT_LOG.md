# RoadSignNet-SAL Development Log

## Project Summary
- **Thesis Title**: "Advanced Machine Vision System for Road Sign Detection Using a Lightweight Model for Moving Objects"
- **Repository**: https://github.com/shihabsararlabib/RoadSignNet-SAL
- **Date**: February 2-3, 2026

---

## Model Results

### Best Model: V5 Finetuned
| Metric | Value |
|--------|-------|
| **mAP@0.5** | **75.52%** |
| **Parameters** | 2.04M |
| **GPU FPS** | 154.6 |
| **CPU FPS** | 33.6 |
| **Model Size** | 7.8 MB |
| **GPU Memory** | 22 MB |

### YOLO Comparison
| Model | Params | mAP@0.5 | FPS | Edge Ready |
|-------|--------|---------|-----|------------|
| **RoadSignNet-SAL (Ours)** | **2.04M** | 75.52% | **154.6** | ✓ |
| YOLOv8n | 3.0M | 85.7% | 120 | ✓ |
| YOLOv8s | 11.2M | 88.5% | 95 | ✗ |
| YOLO11n | 2.6M | 87.9% | 125 | ✓ |
| YOLO11s | 9.4M | 89.0% | 100 | ✗ |

---

## Dataset
- **Source**: Roboflow Universe Traffic Signs
- **Total Images**: 15,127
- **Classes**: 43 traffic sign types
- **Split**: Train (12,773), Val (1,490), Test (864)

---

## Key Scripts Created

| Script | Purpose |
|--------|---------|
| `scripts/fine_tune_v5.py` | Fine-tuning with class-balanced loss |
| `scripts/thesis_benchmark.py` | FPS/latency benchmark validation |
| `scripts/video_inference.py` | Real-time video demo |
| `scripts/generate_phase2_analysis.py` | Phase 2 defense materials |
| `scripts/evaluate.py` | Model evaluation |

---

## Training History

### V5 Original → V5 Finetuned
- Original: 58.69% mAP
- Finetuned: **75.52% mAP** (+16.83% improvement)
- Method: Class-balanced loss, differential learning rates
- Epochs: 50 fine-tuning epochs

---

## Thesis Novelty (Honest Assessment)

### What IS Novel:
1. **Spatial Position Bias** - Learnable prior for sign locations
2. **Specific combination** of lightweight techniques for traffic signs
3. **Edge deployment focus** with <2.5M parameters

### What is NOT Novel (existing techniques):
- Asymmetric Conv (from Inception, 2016)
- Attention modules (from CBAM, SE-Net, 2018)
- Feature Pyramid (from FPN, 2017)

### Thesis Framing:
> "A Systematic Study of Lightweight Detection Techniques for Real-Time Traffic Sign Recognition on Edge Devices"

---

## Phase 2 Defense Checklist

- [x] Working model (75.52% mAP)
- [x] YOLO comparison done
- [x] Benchmark results (154 FPS GPU)
- [x] Code on GitHub
- [x] Phase 2 analysis materials generated

---

## Phase 3 (Final Defense) TODO

| Task | Time | Priority |
|------|------|----------|
| Ablation Study | 2 weeks | HIGH |
| Per-Class Analysis | 1 week | HIGH |
| Cross-Dataset Test | 1 week | Medium |
| Edge Device Deploy | 1 week | Medium |
| Thesis Writing | 4 weeks | HIGH |
| Video Demo | 1 day | HIGH |

---

## Important Paths

```
outputs/checkpoints/best_model_v5_finetuned.pth  # Best model
outputs/benchmark_results.json                    # Benchmark data
outputs/phase2_analysis/                          # Phase 2 charts
```

---

## Commands Reference

```bash
# Run benchmark
python scripts/thesis_benchmark.py --checkpoint outputs/checkpoints/best_model_v5_finetuned.pth

# Evaluate model
python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model_v5_finetuned.pth

# Video inference (webcam)
python scripts/video_inference.py --checkpoint outputs/checkpoints/best_model_v5_finetuned.pth --source 0

# Generate Phase 2 materials
python scripts/generate_phase2_analysis.py
```

---

## Defense Statement

> "I have developed a lightweight road sign detection system with 2.04M parameters that achieves 154 FPS on GPU and 34 FPS on CPU. While the mAP is 75.52% compared to YOLO's 89%, my model is 32% smaller and suitable for edge deployment. The contribution is a systematic integration of lightweight techniques optimized for traffic sign detection on resource-constrained devices."

---

*Log generated: February 3, 2026*
