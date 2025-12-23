# V6 Training Results Analysis

## Summary

The V6 training (150 epochs with Mosaic augmentation, EMA, and enhanced hyperparameters) **failed to improve** on the V5 baseline due to severe overfitting.

## Training Results

### V6 (This Training)
| Metric | Value |
|--------|-------|
| **Parameters** | 2.46M (width_mult=1.5) |
| **Best Epoch** | 63 |
| **Best Val Loss** | 4.03 |
| **Final Val Loss** | 7.43 (150 epochs) |
| **mAP@0.5** | ~0% ❌ |
| **Issue** | Severe overfitting - objectness scores near zero |

### V5 (Previous Best)
| Metric | Value |
|--------|-------|
| **Parameters** | 2.1M (width_mult=1.35) |
| **Best Epoch** | 84 |
| **Best Val Loss** | 0.0364 |
| **mAP@0.5** | **47.87%** ✅ |
| **Precision** | 77.69% |
| **Recall** | 62.63% |
| **F1 Score** | 69.35% |

## Diagnostic Output

V6 model objectness scores (should be >0.1 for detections):
```
Scale 0: Obj range: 0.0000 to 0.0062, mean=0.0000
Scale 1: Obj range: 0.0000 to 0.0153, mean=0.0001
Scale 2: Obj range: 0.0000 to 0.0134, mean=0.0000
```

The model learned to predict "no objects" everywhere because the loss collapsed to near-zero on training data but didn't generalize.

## Root Cause Analysis

### Why V6 Failed:

1. **Wrong Loss Function**: V6 used `RoadSignNetLoss` which expects anchor-based targets, but the training data preparation may have mismatches.

2. **Capacity Mismatch**: `width_multiplier=1.5` (2.46M params) is significantly larger than V5's `width_multiplier=1.35` (2.1M params). More capacity led to memorizing training data.

3. **Loss Scale Issues**: V6 val_loss was ~4.0 while V5 val_loss was ~0.036 - a 100x difference indicating completely different loss computations.

4. **Mosaic Augmentation**: While beneficial for YOLO models, may have caused issues with the anchor assignment strategy in RoadSignNet-SAL.

5. **EMA Not Used During Evaluation**: The EMA model might be better but we evaluated the regular model.

## Comparison with YOLO

| Model | Parameters | mAP@0.5 | Notes |
|-------|-----------|---------|-------|
| RoadSignNet-SAL-v5 | 2.1M | 47.87% | Current best |
| RoadSignNet-SAL-v6 | 2.46M | ~0% | Failed (overfitting) |
| YOLOv8n | 3.0M | 85.7% | 1.4x params, 1.8x better |
| YOLO11n | 2.6M | 84.1% | Similar params, 1.7x better |
| YOLO11s | 9.4M | 89.0% | 4.5x params, 1.9x better |

## Recommendations for Fixing V6

### Option 1: Fix Loss Function (Recommended)
```python
# The V5 training used simpler loss that worked better
# Go back to V5's training approach with augmentation improvements
```

### Option 2: Use EMA Model for Evaluation
```python
# Check if EMA model performs better
checkpoint = torch.load('outputs/v6/.../best_model.pt')
if 'ema_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['ema_state_dict'])
```

### Option 3: Reduce Model Capacity
```python
# Match V5 configuration
width_multiplier = 1.35  # Not 1.5
```

### Option 4: Early Stopping
```python
# Stop training when val_loss starts increasing
# Best epoch was 63, should have stopped there
```

## Realistic Assessment

**RoadSignNet-SAL vs YOLO is not a fair comparison** because:

1. YOLO has 10+ years of optimization and engineering
2. YOLO uses sophisticated training recipes (SGD with warmup, mosaic, copy-paste, etc.)
3. YOLO uses pretrained COCO weights for feature learning
4. RoadSignNet-SAL is a research prototype for demonstrating novel ideas

**Novel contributions of RoadSignNet-SAL worth highlighting in thesis:**
1. Efficient asymmetric convolutions (reduce parameters 40-50%)
2. Lightweight channel-spatial attention
3. Multi-scale detection with anchor-based predictions
4. Custom loss function for traffic sign detection

**For thesis, report V5 results (47.87% mAP) as the achieved performance, and discuss the gap with YOLO honestly as future work.**

## Next Steps

1. ❌ Abandon V6 approach (doesn't work)
2. ✅ Use V5 as final model (47.87% mAP)
3. Focus thesis on architectural novelty, not beating YOLO
4. Discuss YOLO gap as limitation and future work
