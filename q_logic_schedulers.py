class CustomLRScheduler:
    def __init__(self, optimizer, initial_lr, min_lr, max_lr, cooldown_bool=True, cooldown_factor = 0.5):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.current_lr = initial_lr

        

        self.cooldown_factor = cooldown_factor
        self.cooldown_remaining = 0  # counter
        self.cooldown_bool = cooldown_bool

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
        lr = max(self.min_lr, min(self.max_lr, new_lr))
        

        self.current_lr = lr
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def get_lr(self):
        return self.current_lr

    def step(self, *args, **kwargs):
        """Override in subclasses."""
        pass




class WarmupPeakDecayScheduler(CustomLRScheduler):
    def __init__(self, optimizer, warmup_steps = 2500, peak_steps = 2000,decay_steps = 100_000, initial_lr = 1e-4, max_lr = 5e-4, final_lr = 5e-6):
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
            # Linear decay
            t = self.global_step - (self.warmup_steps + self.peak_steps)
            lr = max(self.final_lr , self.max_lr - t * (self.max_lr - self.final_lr) / self.decay_steps) # decay length

        self.set_lr(lr)


class TDAdaptiveScheduler(CustomLRScheduler):
    def __init__(self, optimizer, initial_lr = 1e-4, min_lr = 5e-6, max_lr = 5e-4,
                 low_err=0.1, high_err=1.0, up_factor=1.05, down_factor=0.995):
        super().__init__(optimizer, initial_lr, min_lr, max_lr)
        self.low_err = low_err
        self.high_err = high_err
        self.up_factor = up_factor
        self.down_factor = down_factor

    def step(self, td_error):
        td = abs(td_error)

        if td > self.high_err:
            lr = self.current_lr * self.up_factor
        elif td < self.low_err:
            lr = self.current_lr * self.down_factor
        else:
            lr = self.current_lr  # stable region

        self.set_lr(lr)



class LossAdaptiveLRScheduler(CustomLRScheduler):
    def __init__(self,
                optimizer,
                initial_lr=1e-4,
                min_lr=5e-6,
                max_lr=5e-4,
                ema_alpha=0.05,            
                up_threshold=0.98,         
                down_threshold=1.05,       
                lr_up_factor=1.03,         
                lr_down_factor=0.9,   

            ):
        super().__init__(optimizer, initial_lr, min_lr, max_lr)

        # EMA tracking
        self.ema_loss = None
        self.alpha = ema_alpha

        # thresholds & scale factors
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        self.lr_up_factor = lr_up_factor
        self.lr_down_factor = lr_down_factor


    def step(self, loss_value):
        """Update LR based on loss trend + optional cooldown."""
        
        # Initialize EMA on first call
        if self.ema_loss is None:
            self.ema_loss = loss_value
            return self.current_lr

        prev_ema = self.ema_loss
        self.ema_loss = (1 - self.alpha) * self.ema_loss + self.alpha * loss_value

        ratio = loss_value / prev_ema

        # Loss spike → reduce lr sharply
        if ratio > self.down_threshold:
            new_lr = self.current_lr * self.lr_down_factor

        # Loss stagnant → gently increase lr
        elif ratio > self.up_threshold:
            new_lr = self.current_lr * self.lr_up_factor

        # Loss decreasing → keep lr
        else:
            new_lr = self.current_lr

        # Apply to optimizer
        self.set_lr(new_lr)

        return self.current_lr