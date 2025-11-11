import torch
import torch.nn as nn
import numpy as np


class PriorityLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(PriorityLoss, self).__init__()
        self.reduction = reduction  # 'mean' ili 'sum'

    def forward(self, pred, target, priority):
        """
        Args:
            pred (Tensor): model prediction, shape (batch_size, num_actions)
            target (Tensor): target Q-values, shape (batch_size, num_actions)
            priority (Tensor): weights / priorities, shape (batch_size,)
        Returns:
            Tensor: scalar loss (weighted MSE)
        """
        # 1️⃣ Per-sample MSE po redu
        loss_per_sample = (target - pred).pow(2).mean(dim=1)

        # 2️⃣ Uvjeri se da su veličine kompatibilne
        if priority.dim() == 1:
            priority = priority.view(-1)  # flatten ako treba
        assert loss_per_sample.shape == priority.shape, \
            f"Shape mismatch: {loss_per_sample.shape} vs {priority.shape}"

        # 3️⃣ Primijeni težine (prioritete)
        weighted_loss = loss_per_sample * priority

        # 4️⃣ Kombiniraj rezultate
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss  # bez redukcije


class huberLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(huberLoss, self).__init__()
        self.reduction = reduction  # 'mean' ili 'sum'

    def forward(self, pred, target, weights = None):
        """
        Args:
            pred (Tensor): model prediction, shape (batch_size, num_actions)
            target (Tensor): target Q-values, shape (batch_size, num_actions)
            priority (Tensor): weights / priorities, shape (batch_size,)
        Returns:
            Tensor: scalar loss (weighted MSE)
        """

        if weights != None:
            diff = (target - pred) * weights
        else:
            diff = (target - pred)
            
        loss = torch.where(diff.abs() < 1.0, 0.5 * diff.pow(2), diff.abs() - 0.5).mean()


        return loss



import math
from torch.optim.lr_scheduler import _LRScheduler

class CustomScheduler(_LRScheduler):
    def __init__(self, optimizer, total_steps, base_lr = 5e-4, min_lr = 1e-5, step_size=None, lr_mult = 3, step_size_multipl = None, last_epoch=-1, verbose=False):
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch, verbose,)
        self.min_lr = min_lr
        
        if step_size:
            self.step_size = step_size
            self.step_size_multipl = step_size_multipl
        else:
            self.step_size = total_steps/math.log(base_lr/min_lr, 3)
            self.step_size_multipl = 1

        self.step = self.step_size

        self.lr_mult = lr_mult
        self.lr_pot = lr_mult

    def get_lr(self):
        """Vrati listu LR-ova za svaku param_group u optimizeru."""
        # primjer: jednostavan cosine decay
        if(num_games > self.step):
            self.step_size *= self.step_size_multipl
            self.step += self.step_size
            
            self.lr_pot *= self.lr_mult          

        return [
            base_lr/self.lr_pot * ((1+self.lr_mult)/2 + ((1+self.lr_mult)/2-1) * math.cos(3*math.pi * (num_games - (self.step - self.step_size)) / self.step_size))
            for base_lr in self.base_lrs
        ]


class PiecewiseScheduler(_LRScheduler):
    def __init__(self, optimizer, milestones, values, last_epoch=-1):
        self.milestones = milestones
        self.values = values
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        for i, m in enumerate(self.milestones):
            if step < m:
                return [self.values[i] for _ in self.base_lrs]
        return [self.values[-1] for _ in self.base_lrs]