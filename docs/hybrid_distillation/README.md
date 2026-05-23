# Hybrid Distillation (Multi-CNN + TinyViT)

This prototype trains a lightweight student using a multi-CNN + ViT teacher.

## What it adds
- Triple-backbone fusion: DenseNet-121 + EfficientNet-B0 + TinyViT (timm features_only)
- Distillation training script (teacher -> SAL or transfer student)
- Smoke test harnesses for triple backbone and KAN heads

## Quick smoke test (no dataset)
```powershell
# From repo root
python scripts/smoke_hybrid_distill.py
```

## Distillation training
Edit the distillation section in config_v6.yaml if needed, then run:
```powershell
python scripts/train_distill_hybrid.py --config config/config_v6.yaml
```

## Train a hybrid teacher
```powershell
python scripts/train.py --config config/config_v6.yaml --transfer --backbone densenet121+efficientnet_b0+vit_tiny_patch16_224
```

After training, set:
- `distillation.teacher_checkpoint` to the saved teacher checkpoint path.
- `distillation.student_use_kan_cls` and `distillation.student_kan_grid` if you want a KAN student.

## Compare baseline vs distilled
```powershell
python scripts/compare_distilled.py --config config/config.yaml --baseline path/to/baseline.pth --distilled path/to/best_model_distill.pth
```

## KAN classifier head (detection)
Train with a KAN-based classification head on transfer backbones:
```powershell
python scripts/train.py --config config/config_v6.yaml --transfer --backbone densenet121+efficientnet_b0+vit_tiny_patch16_224 --kan_cls --kan_grid 8
```

Train with a KAN-based classification head on the base RoadSignNet-SAL:
```powershell
python scripts/train.py --config config/config_v6.yaml --kan_cls --kan_grid 8
```

Smoke tests:
```powershell
python scripts/smoke_kan_classifier.py
```

## Notes
- Teacher checkpoint is optional; if not provided, distillation uses untrained teacher outputs.
- For real gains, train the hybrid teacher first and set distillation.teacher_checkpoint.
- Use small batch sizes and mixed precision on RTX 3070 Ti.
