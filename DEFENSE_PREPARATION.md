# FINAL YEAR THESIS DEFENSE PREPARATION
## RoadSignNet-SAL - Road Sign Detection

---

# 🎯 10-SLIDE PRESENTATION OUTLINE

## Slide 1: Title
```
"Advanced Machine Vision System for Road Sign Detection 
Using a Lightweight Model for Moving Objects"

Your Name
Supervisor: [Name]
Department of [Department]
February 2026
```

---

## Slide 2: Problem Statement
```
PROBLEM:
• Self-driving cars need to detect traffic signs in REAL-TIME
• Existing models (YOLO) are too LARGE for cheap devices
• Need: Fast + Small + Accurate model

RESEARCH QUESTION:
"Can we build a lightweight model that detects traffic signs 
in real-time on resource-constrained devices?"
```

---

## Slide 3: Dataset
```
DATASET: Traffic Sign Detection (Roboflow)
• Total Images: 15,127
• Classes: 43 traffic sign types
• Split: Train (12,773) / Val (1,490) / Test (864)

SIGN CATEGORIES:
• Speed Limits (9 types): 20, 30, 40, 50, 60, 70, 80, 100, 120
• Warnings (7 types): children, road_work, speed_bump, etc.
• Prohibitions (9 types): no_entry, no_parking, etc.
• Traffic Lights (3 types): red, yellow, green
• Others (15 types): stop, crosswalk, etc.
```

---

## Slide 4: Proposed Architecture
```
[Include architecture diagram]

THREE MAIN COMPONENTS:
1. BACKBONE: MobileNetV3 (lightweight feature extractor)
2. NECK: Feature Pyramid + Attention (multi-scale detection)
3. HEAD: Detection Head (outputs bounding boxes + classes)

KEY FEATURES:
• Asymmetric Convolutions (33% fewer parameters)
• Attention Module (focuses on important regions)
• Only 2.04 Million parameters
```

---

## Slide 5: Training Process
```
TRAINING CONFIGURATION:
• Optimizer: AdamW
• Learning Rate: 1e-4 (backbone), 1e-3 (head)
• Batch Size: 16
• Epochs: 150 + 50 fine-tuning
• Hardware: NVIDIA RTX 4070 Ti

FINE-TUNING STRATEGY:
• Class-balanced loss for hard classes
• Differential learning rates
• Improved from 58.69% → 75.52% mAP
```

---

## Slide 6: Results - Accuracy
```
DETECTION ACCURACY:

| Metric      | Value    |
|-------------|----------|
| mAP@0.5     | 75.52%   |
| Precision   | 81.01%   |
| Recall      | 87.69%   |
| F1-Score    | 84.22%   |

[Include precision-recall curve or confusion matrix image]
```

---

## Slide 7: Results - Comparison with YOLO
```
COMPARISON WITH STATE-OF-THE-ART:

| Model           | Params | mAP    | FPS   | Edge? |
|-----------------|--------|--------|-------|-------|
| Ours            | 2.04M  | 75.52% | 154.6 | ✓     |
| YOLOv8n         | 3.0M   | 85.7%  | 120   | ✓     |
| YOLOv8s         | 11.2M  | 88.5%  | 95    | ✗     |
| YOLO11n         | 2.6M   | 87.9%  | 125   | ✓     |

KEY FINDING:
• 32% smaller than YOLOv8n
• 29% faster than YOLOv8n
• 10% accuracy trade-off (acceptable for edge devices)
```

---

## Slide 8: Results - Speed/Efficiency
```
REAL-TIME PERFORMANCE:

| Hardware        | FPS    | Latency | Real-time? |
|-----------------|--------|---------|------------|
| GPU (RTX 4070)  | 154.6  | 6.5ms   | ✓ YES      |
| CPU (i7)        | 33.6   | 29.7ms  | ✓ YES      |

MEMORY USAGE:
• Model Size: 7.8 MB
• GPU Memory: 22 MB

THESIS CLAIM VALIDATED:
✓ Lightweight: 2.04M params
✓ Moving Objects: 154 FPS (real-time)
✓ Road Sign Detection: 43 classes, 75.52% mAP
```

---

## Slide 9: Demo (Optional but IMPRESSIVE)
```
[Show video or live demo]

DEMONSTRATION:
• Real-time detection on test video
• Shows bounding boxes + class labels
• Displays FPS counter

[If no video: show sample detection images]
```

---

## Slide 10: Conclusion & Future Work
```
CONCLUSION:
• Successfully built lightweight traffic sign detector
• Achieved 75.52% mAP with only 2.04M parameters
• Real-time capable: 154 FPS on GPU, 34 FPS on CPU
• Suitable for edge deployment (phones, Raspberry Pi)

LIMITATIONS:
• 10% lower accuracy than YOLO
• Some classes (speed_limit_100) have low detection rate

FUTURE WORK:
• Test on edge devices (Jetson Nano, Raspberry Pi)
• Improve hard class detection
• Collect more training data for weak classes
```

---

# 📝 EXPECTED QUESTIONS & ANSWERS

## Q1: "Why not just use YOLO?"
**A:** "YOLO achieves higher accuracy but requires more memory and compute. 
My model is designed for edge devices like Raspberry Pi where YOLO is too large. 
I trade 10% accuracy for 32% size reduction and 29% speed increase."

## Q2: "Why is your accuracy lower?"
**A:** "Two reasons: (1) My model is intentionally smaller with fewer parameters, 
(2) YOLO was pretrained on COCO (118k images) while mine was trained from scratch. 
The trade-off is acceptable for real-time edge deployment."

## Q3: "What's novel in your approach?"
**A:** "I systematically combined lightweight techniques (MobileNet, attention, 
asymmetric convolutions) specifically optimized for traffic sign detection. 
The contribution is the integration and optimization for edge deployment."

## Q4: "Which classes perform poorly and why?"
**A:** "Speed limit signs (100, 70, 80) have lower accuracy because they look 
visually similar - only the number differs. The model sometimes confuses them. 
This could be improved with more training data."

## Q5: "Can this run on a phone?"
**A:** "Yes, the model is only 7.8 MB and achieves 34 FPS on CPU. 
With ONNX optimization, it can run on mobile devices."

## Q6: "What would you do differently?"
**A:** "I would: (1) Collect more data for hard classes, (2) Use knowledge 
distillation from YOLO to improve accuracy, (3) Test on actual edge devices."

---

# ✅ DEFENSE DAY CHECKLIST

## Before Defense:
- [ ] Test all demos work
- [ ] Backup presentation on USB
- [ ] Print slides (backup)
- [ ] Arrive 15 mins early
- [ ] Bring laptop charger

## During Defense:
- [ ] Speak slowly and clearly
- [ ] Make eye contact with panel
- [ ] Use "We" not "I" 
- [ ] Admit limitations honestly
- [ ] Say "I don't know" if unsure

## Key Numbers to Memorize:
- 75.52% mAP
- 2.04M parameters
- 154 FPS (GPU)
- 34 FPS (CPU)
- 43 classes
- 15,127 images
- 32% smaller than YOLO

---

# 🗣️ OPENING STATEMENT (Memorize This)

"Good morning. My thesis presents a lightweight machine vision system for 
real-time traffic sign detection. The key contribution is a 2.04 million 
parameter model that achieves 75% accuracy while running at 154 frames 
per second. This makes it suitable for deployment on resource-constrained 
edge devices like Raspberry Pi or mobile phones."

(30 seconds, confident, clear)

---

# 🏁 CLOSING STATEMENT (Memorize This)

"In conclusion, I have successfully developed a lightweight traffic sign 
detection system that balances accuracy and efficiency. While YOLO achieves 
higher accuracy, my model is 32% smaller and 29% faster, making it practical 
for edge deployment. Future work includes testing on actual edge hardware 
and improving detection of visually similar signs. Thank you."

(30 seconds, confident, summarizes everything)
