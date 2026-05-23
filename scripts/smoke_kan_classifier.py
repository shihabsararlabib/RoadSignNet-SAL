#!/usr/bin/env python3
"""Smoke test for KAN classification head in detection model."""

import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from roadsignnet_sal.model import create_roadsignnet_transfer


def main():
    device = torch.device('cpu')
    model = create_roadsignnet_transfer(
        num_classes=43,
        backbone='densenet121+efficientnet_b0+vit_tiny_patch16_224',
        pretrained=False,
        use_kan_cls=True,
        kan_grid=8
    ).to(device)
    model.eval()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)

    print('KAN smoke OK')
    for i, (cls, box, obj) in enumerate(out):
        print(i, cls.shape, box.shape, obj.shape)


if __name__ == '__main__':
    main()
