# Critical Bug Report: Coordinate System Error in Gaussian Heatmap Generation

**Date:** January 14, 2025  
**Issue:** V2 model achieved 0% mAP despite training for 100 epochs  
**Root Cause:** Normalized bounding box coordinates were not converted to pixel coordinates before grid calculation

## Problem Summary

The RoadSignNet-SAL V2 model was trained for 100 epochs but achieved **0% mAP** (0 true positives, 172,800 false positives). Despite implementing Gaussian heatmap targets and optimizing the generation code, the model completely failed to detect any objects.

## Root Cause Analysis

### The Bug

In `roadsignnet_sal/loss_v2.py`, the `_generate_heatmap()` function received normalized bounding boxes in range `[0, 1]` but was calculating grid coordinates without first converting to pixel coordinates:

```python
# WRONG (before fix):
cx = (valid_bboxes[:, 0] + valid_bboxes[:, 2]) / 2  # cx in [0, 1]
cy = (valid_bboxes[:, 1] + valid_bboxes[:, 3]) / 2  # cy in [0, 1]
grid_x = (cx / stride).long().clamp(0, w - 1)  # Dividing [0,1] by 8.0 gives ~0!
grid_y = (cy / stride).long().clamp(0, h - 1)
```

For example, with an object at center of image:
- Normalized: `cx = 0.5, cy = 0.5`
- Divided by stride (8): `0.5 / 8 = 0.0625`
- Converted to int: `int(0.0625) = 0`
- **Result:** Gaussian peak generated at grid position `[0, 0]` instead of `[26, 26]`

### The Fix

```python
# CORRECT (after fix):
cx = ((valid_bboxes[:, 0] + valid_bboxes[:, 2]) / 2) * img_size  # Convert to pixels
cy = ((valid_bboxes[:, 1] + valid_bboxes[:, 3]) / 2) * img_size  # Convert to pixels
box_w = (valid_bboxes[:, 2] - valid_bboxes[:, 0]) * img_size
box_h = (valid_bboxes[:, 3] - valid_bboxes[:, 1]) * img_size
grid_x = (cx / stride).long().clamp(0, w - 1)  # Now divides pixels by stride
grid_y = (cy / stride).long().clamp(0, h - 1)
```

Now with an object at center:
- Normalized: `cx = 0.5, cy = 0.5`
- Converted to pixels: `cx = 0.5 * 640 = 320`
- Divided by stride: `320 / 8 = 40`
- **Result:** Gaussian peak at grid `[26, 26]` (for 416×416 images) ✅

## Diagnostic Evidence

### Test Results (Before Fix)
```
📍 PEAK LOCATION:
  Max value: 1.000000
  Location: [0, 0, 0, 0]  ← WRONG!
  Expected grid location: [0, 0, 26, 26]

TESTING MULTIPLE OBJECTS
Mask - Positive samples: 1.0 (expected: 2)  ← WRONG COUNT!
```

### Test Results (After Fix)
```
📍 PEAK LOCATION:
  Max value: 1.000000
  Location: [0, 0, 26, 26]  ← CORRECT!
  Expected grid location: [0, 0, 26, 26]

TESTING MULTIPLE OBJECTS
Mask - Positive samples: 2.0 (expected: 2)  ← CORRECT COUNT!
```

### Model Prediction Analysis

Trained model with buggy targets produced:
- Heatmap predictions: ~0.5 everywhere (completely saturated)
- 1598 out of 1600 pixels above 0.5 threshold
- No localization ability whatsoever

The model learned to predict high confidence uniformly across the entire image because:
1. All training targets had Gaussian peaks at `[0, 0]`
2. The model minimized loss by predicting ~0.5 everywhere
3. No meaningful object localization was learned

## Impact

- **100 epochs of V2 training completely wasted** (~8 hours of GPU time)
- Model parameters: 1.66M (within target)
- Training loss decreased (7.4 → 0.11) indicating learning occurred
- But learned completely wrong pattern due to broken targets
- Evaluation results: 0% mAP, 0 TP, 172,800 FP

## Action Required

**IMMEDIATE:** Retrain V2 model with corrected target generation
- Same architecture (1.66M params)
- Same hyperparameters (AdamW, lr=0.001, 100 epochs, batch_size=32)
- Fixed heatmap generation will enable actual object localization

## Lessons Learned

1. **Always visualize targets during debugging** - If we had checked target generation earlier, this bug would have been caught before wasting 100 epochs of training
2. **Unit test coordinate transformations** - Normalized vs. pixel coordinate confusion is a common source of bugs
3. **Sanity check predictions** - The saturated heatmap (0.5 everywhere) was a red flag that should have triggered immediate investigation of target generation
4. **Test with simple cases first** - Single object at known location makes bugs obvious

## Files Modified

- `roadsignnet_sal/loss_v2.py`: Fixed coordinate conversion in `_generate_heatmap()`
- `scripts/debug_targets.py`: Created to test target generation
- `scripts/debug_predictions.py`: Created to analyze model outputs

## Next Steps

1. ✅ Bug identified and fixed
2. ⏳ Retrain V2 model (100 epochs, ~5 hours)
3. ⏳ Evaluate with fixed model
4. ⏳ Compare against V1 baseline (50.71% mAP) and transfer learning (58.22% mAP)
5. ⏳ Conduct ablation studies for thesis
