
# Class weights for handling imbalanced dataset
# Generated automatically - add to loss.py

import torch

CLASS_WEIGHTS = torch.tensor([
    0.013448,
    0.006164,
    0.026789,
    0.003093,
    0.015163,
    0.006436,
    0.007133,
    0.009601,
    0.006112,
    0.012433,
    0.001017,
    0.027132,
    0.009956,
    21.081363,
    0.005137,
    0.002593,
    0.009732,
    0.263609,
    0.012291,
    0.008606,
    0.009820,
    0.000764,
    0.033044,
    0.008640,
    0.022295,
    0.030220,
    21.081363,
    0.013794,
    0.040645,
    0.033044,
    0.045932,
    0.007942,
    0.019097,
    0.019445,
    0.009644,
    0.016704,
    0.024618,
    0.004749,
    0.010490,
    0.024906,
    0.004054,
    0.008121,
    0.002859
], dtype=torch.float32)

# Use in loss function:
# cls_loss = F.binary_cross_entropy_with_logits(
#     cls_pred, cls_target, 
#     weight=CLASS_WEIGHTS[target_classes].to(device)
# )
