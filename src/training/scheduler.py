"""Learning rate schedulers for osu! replay training."""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import Optional, List, Union


class WarmupScheduler(_LRScheduler):
    """Learning rate scheduler with warmup period."""
    
    def __init__(self, optimizer: optim.Optimizer, warmup_steps: int, 
                 total_steps: int, min_lr: float = 0.0, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_steps:
            # Warmup phase: linear increase
            warmup_factor = self.last_epoch / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Decay phase: cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """Cosine annealing with warm restarts and warmup."""
    
    def __init__(self, optimizer: optim.Optimizer, first_cycle_steps: int,
                 cycle_mult: float = 1.0, max_lr: float = 0.1, min_lr: float = 0.001,
                 warmup_steps: int = 0, gamma: float = 1.0, last_epoch: int = -1):
        
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        
        super().__init__(optimizer, last_epoch)
        
        # Initialize learning rates
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self) -> List[float]:
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            # Warmup phase
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing phase
            return [
                base_lr + (self.max_lr - base_lr) * 
                (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) / 
                             (self.cur_cycle_steps - self.warmup_steps))) / 2
                for base_lr in self.base_lrs
            ]
    
    def step(self, epoch: Optional[int] = None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_lrs[0] * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Linear warmup followed by cosine annealing."""
    
    def __init__(self, optimizer: optim.Optimizer, warmup_epochs: int, max_epochs: int,
                 warmup_start_lr: float = 1e-8, eta_min: float = 1e-8, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * self.last_epoch / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            return [
                self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr in self.base_lrs
            ]


class PolynomialLR(_LRScheduler):
    """Polynomial learning rate decay."""
    
    def __init__(self, optimizer: optim.Optimizer, total_iters: int, power: float = 1.0,
                 min_lr: float = 0.0, last_epoch: int = -1):
        self.total_iters = total_iters
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group['lr'] for group in self.optimizer.param_groups]
        
        decay_factor = (1 - self.last_epoch / self.total_iters) ** self.power
        return [
            self.min_lr + (base_lr - self.min_lr) * decay_factor
            for base_lr in self.base_lrs
        ]


class OneCycleLR(_LRScheduler):
    """One cycle learning rate policy."""
    
    def __init__(self, optimizer: optim.Optimizer, max_lr: Union[float, List[float]],
                 total_steps: int, pct_start: float = 0.3, anneal_strategy: str = 'cos',
                 cycle_momentum: bool = True, base_momentum: float = 0.85,
                 max_momentum: float = 0.95, div_factor: float = 25.0,
                 final_div_factor: float = 1e4, last_epoch: int = -1):
        
        if not isinstance(max_lr, (list, tuple)):
            self.max_lrs = [max_lr] * len(optimizer.param_groups)
        else:
            self.max_lrs = list(max_lr)
        
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.cycle_momentum = cycle_momentum
        self.base_momentum = base_momentum
        self.max_momentum = max_momentum
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        self.step_up_size = int(self.total_steps * self.pct_start)
        self.step_down_size = self.total_steps - self.step_up_size
        
        # Calculate base learning rates
        self.base_lrs = [max_lr / self.div_factor for max_lr in self.max_lrs]
        self.min_lrs = [max_lr / self.final_div_factor for max_lr in self.max_lrs]
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch <= self.step_up_size:
            # Increasing phase
            pct = self.last_epoch / self.step_up_size
            return [
                base_lr + pct * (max_lr - base_lr)
                for base_lr, max_lr in zip(self.base_lrs, self.max_lrs)
            ]
        else:
            # Decreasing phase
            pct = (self.last_epoch - self.step_up_size) / self.step_down_size
            
            if self.anneal_strategy == 'cos':
                return [
                    min_lr + (max_lr - min_lr) * (1 + math.cos(math.pi * pct)) / 2
                    for min_lr, max_lr in zip(self.min_lrs, self.max_lrs)
                ]
            else:  # linear
                return [
                    max_lr - pct * (max_lr - min_lr)
                    for min_lr, max_lr in zip(self.min_lrs, self.max_lrs)
                ]
    
    def step(self, epoch: Optional[int] = None):
        super().step(epoch)
        
        # Update momentum if cycle_momentum is True
        if self.cycle_momentum:
            if self.last_epoch <= self.step_up_size:
                # Decreasing momentum phase
                pct = self.last_epoch / self.step_up_size
                momentum = self.max_momentum - pct * (self.max_momentum - self.base_momentum)
            else:
                # Increasing momentum phase
                pct = (self.last_epoch - self.step_up_size) / self.step_down_size
                momentum = self.base_momentum + pct * (self.max_momentum - self.base_momentum)
            
            for param_group in self.optimizer.param_groups:
                if 'momentum' in param_group:
                    param_group['momentum'] = momentum
                elif 'betas' in param_group:
                    param_group['betas'] = (momentum, param_group['betas'][1])


def get_scheduler(optimizer: optim.Optimizer, scheduler_config: dict) -> _LRScheduler:
    """Factory function to create learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_config: Configuration dictionary
        
    Returns:
        Learning rate scheduler
    """
    scheduler_type = scheduler_config.get('type', 'cosine')
    
    if scheduler_type == 'warmup':
        return WarmupScheduler(
            optimizer,
            warmup_steps=scheduler_config.get('warmup_steps', 1000),
            total_steps=scheduler_config.get('total_steps', 10000),
            min_lr=scheduler_config.get('min_lr', 0.0)
        )
    
    elif scheduler_type == 'cosine_restarts':
        return CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=scheduler_config.get('first_cycle_steps', 1000),
            cycle_mult=scheduler_config.get('cycle_mult', 1.0),
            max_lr=scheduler_config.get('max_lr', 0.1),
            min_lr=scheduler_config.get('min_lr', 0.001),
            warmup_steps=scheduler_config.get('warmup_steps', 100),
            gamma=scheduler_config.get('gamma', 1.0)
        )
    
    elif scheduler_type == 'linear_warmup_cosine':
        return LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=scheduler_config.get('warmup_epochs', 10),
            max_epochs=scheduler_config.get('max_epochs', 100),
            warmup_start_lr=scheduler_config.get('warmup_start_lr', 1e-8),
            eta_min=scheduler_config.get('eta_min', 1e-8)
        )
    
    elif scheduler_type == 'polynomial':
        return PolynomialLR(
            optimizer,
            total_iters=scheduler_config.get('total_iters', 10000),
            power=scheduler_config.get('power', 1.0),
            min_lr=scheduler_config.get('min_lr', 0.0)
        )
    
    elif scheduler_type == 'onecycle':
        return OneCycleLR(
            optimizer,
            max_lr=scheduler_config.get('max_lr', 0.1),
            total_steps=scheduler_config.get('total_steps', 10000),
            pct_start=scheduler_config.get('pct_start', 0.3),
            anneal_strategy=scheduler_config.get('anneal_strategy', 'cos'),
            cycle_momentum=scheduler_config.get('cycle_momentum', True),
            base_momentum=scheduler_config.get('base_momentum', 0.85),
            max_momentum=scheduler_config.get('max_momentum', 0.95),
            div_factor=scheduler_config.get('div_factor', 25.0),
            final_div_factor=scheduler_config.get('final_div_factor', 1e4)
        )
    
    elif scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 1000),
            gamma=scheduler_config.get('gamma', 0.1)
        )
    
    elif scheduler_type == 'exponential':
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=scheduler_config.get('gamma', 0.95)
        )
    
    elif scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get('T_max', 1000),
            eta_min=scheduler_config.get('eta_min', 0.0)
        )
    
    elif scheduler_type == 'reduce_on_plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get('mode', 'min'),
            factor=scheduler_config.get('factor', 0.1),
            patience=scheduler_config.get('patience', 10),
            threshold=scheduler_config.get('threshold', 1e-4),
            min_lr=scheduler_config.get('min_lr', 0.0)
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")