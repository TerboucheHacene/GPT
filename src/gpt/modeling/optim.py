import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class GPTLearningRateScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        min_lr: float,
        warmup_steps: int,
        max_steps: int,
    ):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.step_count = 0

    def get_lr(self) -> float:
        if self.step_count < self.warmup_steps:
            lr = self.max_lr * (self.step_count + 1) / self.warmup_steps
        elif self.step_count > self.max_steps:
            lr = self.min_lr
        else:
            decay_ratio = (self.step_count - self.warmup_steps) / (
                self.max_steps - self.warmup_steps
            )
            coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
            lr = self.min_lr + coeff * (self.max_lr - self.min_lr)
        return lr

    def step(self):
        self.step_count += 1
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.get_lr()
        return self.get_lr()
