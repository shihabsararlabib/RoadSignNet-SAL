from roadsignnet_sal.model_optimized import create_roadsignnet_optimized
import torch

for w in [1.0, 1.1, 1.2, 1.25, 1.3, 1.35, 1.4]:
    model = create_roadsignnet_optimized(43, w)
    params = sum(p.numel() for p in model.parameters())
    print(f'Width {w:.2f}: {params/1e6:.2f}M params')
    del model
    torch.cuda.empty_cache()
