import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

def get_optimizer_and_scheduler(model, base_lr=0.001, warmup_steps=1000, total_steps=10000):
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    return optimizer, warmup_scheduler, cosine_scheduler