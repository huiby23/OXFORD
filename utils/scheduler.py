import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler

class WarmupReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(self, optimizer, warmup_iters, mode='min', factor=0.5, patience=5, 
                 threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, 
                 eps=1e-8, verbose=False):
        super().__init__(optimizer, mode, factor, patience, threshold, 
                         threshold_mode, cooldown, min_lr, eps, verbose)    # 注意参数顺序
        
        self.warmup_iters = warmup_iters
        self.current_iter = 0
    
    def step(self, metrics, epoch=None):
        # 更新迭代计数器
        self.current_iter += 1
        
        # 在预热阶段保持学习率不变
        if self.current_iter <= self.warmup_iters:
            return
        
        # 预热结束后，使用正常的ReduceLROnPlateau逻辑
        super().step(metrics, epoch)



# 使用示例
# warmup_iters = 100000  # 前100,000次迭代保持学习率不变
# patience = 25000 // 5

# scheduler = WarmupReduceLROnPlateau(
#     optimizer,
#     warmup_iters=warmup_iters,
#     mode='min',
#     patience=patience,
#     factor=0.5
# )



class LrTempScheduler(_LRScheduler):
    def __init__(self, optimizer, lr_schedule, t_schedule, last_step=-1):
        """
        lr_schedule: List of tuples [(step, lr), ...]
        t_schedule: List of tuples [(step, T), ...]
        last_step: current training step (int)
        """
        self.lr_schedule = sorted(lr_schedule)
        self.t_schedule = sorted(t_schedule)
        self.T = self._get_temperature(last_step)
        self.current_step = last_step
        super().__init__(optimizer, last_epoch=last_step)

    def _get_lr(self, step):
        current_lr = self.lr_schedule[0][1]
        for s, lr in self.lr_schedule:
            if step >= s:
                current_lr = lr
            else:
                break
        return current_lr

    def _get_temperature(self, step):
        current_T = self.t_schedule[0][1]
        for s, T in self.t_schedule:
            if step >= s:
                current_T = T
            else:
                break
        return current_T

    def step(self, counter=None):
        """ Update lr and T using external counter (e.g. global_step) """
        if counter is None:
            counter = self.current_step

        self.current_step = counter
        self.T = self._get_temperature(counter)
        lr = self._get_lr(counter)

        # Update learning rate for all param groups
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {
            'lr_schedule': self.lr_schedule,
            't_schedule': self.t_schedule,
            'current_step': self.current_step,
            'T': self.T
        }

    def load_state_dict(self, state_dict):
        self.lr_schedule = state_dict['lr_schedule']
        self.t_schedule = state_dict['t_schedule']
        self.current_step = state_dict['current_step']
        self.T = state_dict['T']
        self.step(self.current_step)  # ensure optimizer.lr is updated