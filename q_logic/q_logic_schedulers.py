'''
file conataining different schedulers, easy to add new ones, 
for now only warmup scheduler showed promise, only problem is you need to monitor it a bit to see if learning platoes
RL is more sensitive to learning rate and lr should decay but not too fast, you need to find a sweet spot 
'''
import numpy as np
from collections.abc import Iterable
import math 

class CustomLRScheduler:
    def __init__(self, optimizer, initial_lr, min_lr, max_lr, cooldown_bool=True, cooldown_factor = 0.5):
        self.optimizer = optimizer
        self.initial_lr = np.array(initial_lr)
        self.min_lr = np.array(min_lr)
        self.max_lr = np.array(max_lr)
        self.current_lr = np.array(initial_lr)

        self.is_list = isinstance(initial_lr, Iterable)

        

        self.cooldown_factor = cooldown_factor
        self.cooldown_remaining = 0  # counter
        self.cooldown_bool = cooldown_bool

        if self.is_list:
            for pg, lr in zip(self.optimizer.param_groups, initial_lr):
                pg["lr"] = lr
        else:
            for pg in self.optimizer.param_groups:
                pg["lr"] = initial_lr

    def notify_target_update(self, target_cycle):
        """Call this whenever the target network is updated."""
        self.cooldown_remaining = target_cycle*0.1

    def _apply_cooldown(self, lr):
        """If in cooldown, scale LR down."""
        if self.cooldown_remaining > 0 and self.cooldown_bool:
            lr = lr * self.cooldown_factor
            self.cooldown_remaining -= 1
        return lr

    def set_lr(self, lr):
        new_lr = self._apply_cooldown(lr)
        lr = np.maximum(self.min_lr, np.minimum(self.max_lr, new_lr))
        

        self.current_lr = lr

        if self.is_list:
            for pg, lr in zip(self.optimizer.param_groups, lr):
                pg["lr"] = lr
        else:
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

    def get_lr(self):
        return self.current_lr

    def step(self, *args, **kwargs):
        """Override in subclasses."""
        pass




class LinearDecayScheduler(CustomLRScheduler):
    def __init__(self, optimizer, warmup_steps = 2500, peak_steps = 2000,decay_steps = 30_000, initial_lr = 1e-4, max_lr = 5e-4, final_lr = 1e-6):
        super().__init__(optimizer, initial_lr, final_lr, max_lr)

        self.warmup_steps = warmup_steps
        self.peak_steps = peak_steps
        self.decay_steps = decay_steps


        self.final_lr = final_lr
        self.max_lr = max_lr
        self.global_step = 0

    def step(self):
        self.global_step +=1
        if self.global_step < self.warmup_steps:
            # Warmup linearly
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * (self.global_step / self.warmup_steps)

        elif self.global_step < self.warmup_steps + self.peak_steps:
            # Stay at peak
            lr = self.max_lr

        else:
            t = self.global_step - (self.warmup_steps + self.peak_steps)
            if (self.max_lr - t * (self.max_lr - self.final_lr) / self.decay_steps < self.max_lr/10):
                self.max_lr /= 10
                self.global_step = (self.warmup_steps + self.peak_steps) # ovo sam novo dodao cini se zanimljivo
            # Linear decay
            
            lr = max(self.final_lr , self.max_lr - t * (self.max_lr - self.final_lr) / self.decay_steps) # decay length

        self.set_lr(lr)


class CosineAnealSchedulerWarmReset(CustomLRScheduler):
    def __init__(self, optimizer, warmup_steps = 2500, peak_steps = 2000,decay_steps = 100_000, initial_lr = 1e-4, max_lr = 5e-4, reset_multiplier = 0.5, decay_step_multiplier = 1.3, final_lr = 1e-6):
        super().__init__(optimizer, initial_lr, final_lr, max_lr)

        """
        schduler which every decay_steps number of steps gets reset to max_lr, but max_lr also decays
        """

        self.warmup_steps = warmup_steps
        self.peak_steps = peak_steps
        self.decay_steps = decay_steps #steps untill it resets
        self.reset_multiplier = reset_multiplier # after reset max_lr gets multiplied by this
        self.decay_step_multiplier = decay_step_multiplier

        self.final_lr = final_lr
        self.max_lr = max_lr
        self.global_step = 0

    def step(self):
        self.global_step +=1
        if self.global_step < self.warmup_steps:
            # Warmup linearly
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * (self.global_step / self.warmup_steps)

        elif self.global_step < self.warmup_steps + self.peak_steps:
            # Stay at peak
            lr = self.max_lr


        else:
            progress = self.global_step / self.decay_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            if ( self.global_step == self.decay_steps):
                self.max_lr *= self.reset_multiplier
                self.global_step *= self.decay_step_multiplier 
                self.global_step = (self.warmup_steps + self.peak_steps) # ovo sam novo dodao cini se zanimljivo
            # Linear decay
            
            lr = self.final_lr + (self.max_lr - self.final_lr) * cosine_decay # decay length

        self.set_lr(lr)

