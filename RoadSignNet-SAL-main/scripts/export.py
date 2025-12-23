#!/usr/bin/env python3
"""
RoadSignNet-SAL Model Export Script
Export to ONNX, TorchScript, TFLite
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import yaml
import argparse
from pathlib import Path

from roadsignnet_sal.model import create_roadsignnet_sal


def export_model(config, checkpoint_path):
    """Export model to multiple formats"""
    
    device = torch.device('cpu')  # Export on CPU
    
    print("="*70)
    print("ROADSIGNNET-SAL MODEL EXPORT")
    print("="*70)
    
    # Load model
    model = create_roadsignnet_sal(
        num_classes=config['model']['num_classes'],
        width_multiplier=config['model']['width_multiplier']
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded from: {checkpoint_path}")
    
    # Create export directory
    export_dir = Path(config['export']['save_dir'])
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Dummy input
    dummy_input = torch.randn(1, 3, 640, 640).to(device)
    
    # 1. ONNX Export
    if 'onnx' in config['export']['formats']:
        print("\n[1/3] Exporting to ONNX...")
        onnx_path = export_dir / 'roadsignnet_sal.onnx'
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"✓ ONNX saved: {onnx_path}")
        print(f"  Size: {onnx_path.stat().st_size / (1024**2):.2f} MB")
    
    # 2. TorchScript Export
    if 'torchscript' in config['export']['formats']:
        print("\n[2/3] Exporting to TorchScript...")
        try:
            scripted_model = torch.jit.script(model)
            script_path = export_dir / 'roadsignnet_sal_scripted.pt'
            scripted_model.save(str(script_path))
            print(f"✓ TorchScript saved: {script_path}")
            print(f"  Size: {script_path.stat().st_size / (1024**2):.2f} MB")
        except Exception as e:
            print(f"⚠ TorchScript export failed: {e}")
            print("  Saving standard PyTorch model instead...")
            script_path = export_dir / 'roadsignnet_sal.pt'
            torch.save(model.state_dict(), script_path)
            print(f"✓ PyTorch model saved: {script_path}")
    
    # 3. Quantized Model
    if config['export']['quantize']:
        print("\n[3/3] Applying INT8 quantization...")
        model_quantized = torch.quantization.quantize_dynamic(
            model.cpu(),
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )
        quant_path = export_dir / 'roadsignnet_sal_int8.pth'
        torch.save(model_quantized.state_dict(), quant_path)
        print(f"✓ Quantized model saved: {quant_path}")
        print(f"  Size: {quant_path.stat().st_size / (1024**2):.2f} MB")
    
    print("\n" + "="*70)
    print("✅ EXPORT COMPLETE")
    print("="*70)
    print(f"\nExported models:")
    for file in export_dir.iterdir():
        if file.is_file():
            print(f"  {file.name} ({file.stat().st_size / (1024**2):.2f} MB)")
    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    export_model(config, args.checkpoint)