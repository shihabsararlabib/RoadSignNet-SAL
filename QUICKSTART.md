# RoadSignNet-SAL - Quick Reference

## 🚀 Start Here (Windows PowerShell)

### 1. Setup (First Time Only)
```powershell
cd C:\RoadSignNet-SAL
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Prepare Your Data
- Download dataset (e.g., from Roboflow in YOLO format)
- Extract to `data/` folder
- Structure should be:
  ```
  data/
  ├── train/
  │   ├── images/  (*.jpg, *.png)
  │   └── labels/  (*.txt - YOLO format)
  └── val/
      ├── images/
      └── labels/
  ```

### 3. Update Config (Optional)
Edit `config/config.yaml` if needed:
- Change `num_classes` to match your dataset
- Adjust `batch_size` if GPU memory issues
- Modify `epochs` for training duration

### 4. Train
```powershell
python scripts/train.py
```
- Best model saved to: `outputs/checkpoints/best_model.pth`
- View progress: `tensorboard --logdir=outputs/logs`

### 5. Inference
```powershell
python scripts/inference.py --checkpoint outputs/checkpoints/best_model.pth --input C:\path\to\image.jpg
```

### 6. Evaluate
```powershell
python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pth
```

---

## 📂 Important Paths

| Purpose | Path |
|---------|------|
| Training Images | `C:\RoadSignNet-SAL\data\train\images\` |
| Training Labels | `C:\RoadSignNet-SAL\data\train\labels\` |
| Validation Images | `C:\RoadSignNet-SAL\data\val\images\` |
| Validation Labels | `C:\RoadSignNet-SAL\data\val\labels\` |
| Config File | `C:\RoadSignNet-SAL\config\config.yaml` |
| Best Model | `C:\RoadSignNet-SAL\outputs\checkpoints\best_model.pth` |
| TensorBoard Logs | `C:\RoadSignNet-SAL\outputs\logs\` |

---

## ⚙️ Common Configuration Changes

### Edit `config/config.yaml`:

**Change number of classes:**
```yaml
model:
  num_classes: 50  # Change to your number
```

**Reduce memory usage:**
```yaml
training:
  batch_size: 8    # Reduce from 16
  num_workers: 4   # Reduce from 8
```

**Use CPU instead of GPU:**
```yaml
hardware:
  device: "cpu"    # Change from "cuda"
```

**Train longer:**
```yaml
training:
  epochs: 200      # Increase from 100
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: roadsignnet_sal` | Run `pip install -e .` from project root |
| `CUDA out of memory` | Reduce `batch_size` in config.yaml |
| `No images found` | Check paths use `C:\` format, not `./` relative paths |
| `ModuleNotFoundError: cv2` | Run `pip install opencv-python` |
| `ModuleNotFoundError: torch` | Run `pip install torch torchvision` |

---

## 📊 Label Format (YOLO)

Each label file (`.txt`) should have one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```

**Example:** (Image is 1280×720)
```
0 0.5 0.5 0.3 0.4
2 0.75 0.25 0.2 0.3
```

Means:
- Object 0 (class 0) at center, 30% width, 40% height
- Object 1 (class 2) at 75% right, 25% down, 20% width, 30% height

All coordinates are **normalized** (0-1 range).

---

## 🎯 Model Specs

- **Architecture**: RoadSignNet-SAL (lightweight)
- **Input Size**: 640×640 RGB images
- **Output**: 3 scales (P3, P4, P5)
- **Parameters**: ~2.1M (vs 3.2M for YOLOv8n)
- **Speed**: 50+ FPS on RTX 3090, ~20 FPS on Raspberry Pi

---

## 📈 Monitor Training

While training, open another PowerShell window:
```powershell
cd C:\RoadSignNet-SAL
tensorboard --logdir=outputs/logs
```

Then open browser: `http://localhost:6006`

Metrics tracked:
- Training/Validation Loss
- Class Loss, Box Loss, Object Loss
- Learning Rate Schedule

---

## 💾 Save & Export

After training:

**Save model:**
```powershell
# Automatically saved during training as best_model.pth
```

**Export for deployment:**
```powershell
python scripts/export.py --checkpoint outputs/checkpoints/best_model.pth
```
Creates:
- `outputs/exports/roadsignnet_sal.onnx` (for inference servers)
- `outputs/exports/roadsignnet_sal_scripted.pt` (for PyTorch apps)

---

## ⚡ Performance Tips

1. **Faster training**: Reduce image size in config (e.g., 480 instead of 640)
2. **Better accuracy**: Increase epochs and use larger batch size (if GPU allows)
3. **Faster inference**: Reduce `img_size` in config
4. **Memory**: Use smaller `width_multiplier` (e.g., 0.75)

---

## 🆘 Get More Help

- **Setup**: See `SETUP_GUIDE.md`
- **Fixes**: See `FIXES_SUMMARY.md`
- **Config**: See `config/config.yaml` (well-commented)
- **Code**: Check `roadsignnet_sal/` package files

---

## ✅ Verification

Test if everything works:
```powershell
# Activate venv first
.\venv\Scripts\Activate.ps1

# Test imports
python -c "from roadsignnet_sal import create_roadsignnet_sal; print('✓ Model imports OK')"

# Test dataset
python -c "from roadsignnet_sal.dataset import create_dataloader; print('✓ Dataset loads OK')"

# Test GPU
python -c "import torch; print(f'✓ GPU Available: {torch.cuda.is_available()}')"
```

---

**Ready to train!** 🚀
