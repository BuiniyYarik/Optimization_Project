from typing import Any, Dict
from functools import reduce

import numpy as np
import torch
from torch.optim import Optimizer


def _flatten_shape(shape: list):
    return reduce(lambda x, y: x*y, shape)

def proximal_l1(alpha, x):
    # soft threshold
    r = torch.abs(x) - alpha
    r[r < 0] = 0
    l = torch.sign(x)
    return l * r
    

class ProxFinito(Optimizer):
    def __init__(self, params, n_batches: int, lr=0.01, theta=0.5) -> None:
        super(ProxFinito, self).__init__(params, defaults={"lr": lr, "theta": theta})
        self.lr = lr
        self.theta = theta
        self.n_batches = n_batches        
        self._iter = 0
        self._init_weights()
        
    def _init_weights(self):
        self.state = dict()
        for group in self.param_groups:
            for p in group['params']:
                # initalising weights, should be improved in the future
                weights = torch.rand(self.n_batches, *p.data.shape) - 0.5
                z_p = weights.mean(dim=0)
                self.state[p] = dict(z=weights, z_p=z_p, old_epoch_z_p=z_p)
                p.data = proximal_l1(self.lr, z_p)
                
    def _dump(self, p):            
        # epoch passed, dumping process
    
        # TODO: в статье здесь вместо знака равно стрелка влево, может это значит что то другое
        c_z_p = self.state[p]["z_p"]
        self.state[p]["z_p"] = (1 - self.theta) * self.state[p]["old_epoch_z_p"] + self.theta * c_z_p
        self.state[p]["old_epoch_z_p"] = c_z_p
                
    def step(self, batch_idx: int):
        for group in self.param_groups:
            for p in group["params"]:
                z = self.state[p]["z"]
                grad = p.grad.data
                d = p.data - self.lr * grad - z[batch_idx]
                
                self.state[p]["z_p"] = self.state[p]["z_p"] + d / self.n_batches
                self.state[p]["z"][batch_idx] = self.state[p]["z"][batch_idx] + self.theta * d
                
                p.data = proximal_l1(self.lr, self.state[p]["z_p"])

                if self._iter == self.n_batches - 1:
                    self._dump(p)
                    
        self._iter += 1