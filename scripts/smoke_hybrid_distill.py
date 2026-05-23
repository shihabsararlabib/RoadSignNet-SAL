#!/usr/bin/env python3
"""Quick smoke test for hybrid distillation components."""

import sys
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from roadsignnet_sal.model import create_roadsignnet_transfer, create_roadsignnet_sal


def distill_sigmoid_mse(student_logits, teacher_logits, temperature=1.0):
    s = torch.sigmoid(student_logits / temperature)
    t = torch.sigmoid(teacher_logits / temperature)
    return F.mse_loss(s, t)


def main():
    device = torch.device('cpu')
    teacher = create_roadsignnet_transfer(
        num_classes=43,
        backbone='densenet121+efficientnet_b0+vit_tiny_patch16_224',
        pretrained=False
    ).to(device).eval()

    student = create_roadsignnet_sal(
        num_classes=43,
        width_multiplier=1.0,
        use_kan_cls=True,
        kan_grid=8
    ).to(device).eval()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        t_out = teacher(x)
        s_out = student(x)

    distill_loss = 0.0
    for (s_cls, s_box, s_obj), (t_cls, t_box, t_obj) in zip(s_out, t_out):
        if t_cls.shape[-2:] != s_cls.shape[-2:]:
            t_cls = F.interpolate(t_cls, size=s_cls.shape[-2:], mode='bilinear', align_corners=False)
            t_box = F.interpolate(t_box, size=s_box.shape[-2:], mode='bilinear', align_corners=False)
            t_obj = F.interpolate(t_obj, size=s_obj.shape[-2:], mode='bilinear', align_corners=False)
        distill_loss += distill_sigmoid_mse(s_cls, t_cls, 2.0)

    print("Smoke OK")
    print(f"Distill loss (cls only): {distill_loss.item():.6f}")


if __name__ == '__main__':
    main()
