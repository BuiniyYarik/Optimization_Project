import numpy as np
import math
import time
from loss import safe_sparse_norm
from opt_trace import Trace, StochasticTrace
from utils import set_seed
from tqdm import tqdm


class Optimizer:
    def __init__(self, loss, t_max=np.inf, it_max=100, tolerance=0):
        self.loss = loss
        self.t_max = t_max
        self.it_max = it_max
        self.tolerance = tolerance
        self.initialized = False
        self.x_old = None

    def run(self, x0):
        self.init_run(x0)
        for _ in tqdm(range(self.it_max), desc="Running Optimizer"):
            if self.check_convergence():
                break
            self.step()
        return self.x

    def check_convergence(self):
        time_exceeded = time.time() - self.t_start >= self.t_max
        iterations_exceeded = self.it >= self.it_max
        tolerance_met = self.tolerance > 0 and safe_sparse_norm(self.x - self.x_old) < self.tolerance
        return time_exceeded or iterations_exceeded or tolerance_met

    def step(self):
        raise NotImplementedError("Subclasses should implement this!")

    def init_run(self, x0):
        self.x = x0.copy()
        self.it = 0
        self.t_start = time.time()
        self.initialized = True


class StochasticOptimizer(Optimizer):
    def __init__(self, loss, n_seeds=1, seeds=None, *args, **kwargs):
        super().__init__(loss=loss, *args, **kwargs)
        self.seeds = seeds if seeds else [np.random.randint(100000) for _ in range(n_seeds)]

    def run(self, x0):
        results = []
        for seed in self.seeds:
            set_seed(seed)
            self.init_run(x0)
            for _ in tqdm(range(self.it_max), desc=f"Running Stochastic Optimizer (Seed: {seed})"):
                if self.check_convergence():
                    break
                self.step()
            results.append(self.x.copy())
        return results


class Ig(Optimizer):
    def __init__(self, lr0=None, lr_max=np.inf, lr_decay_coef=0, lr_decay_power=1, it_start_decay=None, batch_size=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr0 = lr0 if lr0 is not None else 1 / self.loss.batch_smoothness(batch_size)
        self.lr_max = lr_max
        self.lr_decay_coef = lr_decay_coef
        self.lr_decay_power = lr_decay_power
        self.it_start_decay = it_start_decay or self.it_max // 40
        self.batch_size = batch_size
        self.i = 0

    def step(self):
        idx = np.arange(self.i, self.i + self.batch_size) % self.loss.n
        self.i = (self.i + self.batch_size) % self.loss.n
        self.grad = self.loss.stochastic_gradient(self.x, idx=idx)
        lr = 1 / (1 / self.lr0 + self.lr_decay_coef * max(0, self.it - self.it_start_decay)**self.lr_decay_power)
        self.lr = min(lr, self.lr_max)
        # Compute the mean gradient and reshape it to match the shape of self.x
        if self.grad.ndim == 2:
            mean_grad = np.mean(self.grad, axis=1).reshape(-1, 1)
        else:
            mean_grad = self.grad.reshape(-1, 1)
        self.x -= self.lr * mean_grad


class Nesterov(Optimizer):
    def __init__(self, lr=None, strongly_convex=False, mu=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr if lr is not None else 1 / self.loss.smoothness()
        self.strongly_convex = strongly_convex
        self.mu = mu
        self.alpha = 1.0
        self.x_nest = None
        self.momentum = 0

    def init_run(self, x0):
        super().init_run(x0)
        self.x_nest = self.x.copy()
        if self.strongly_convex:
            kappa = (1 / self.lr) / self.mu
            self.momentum = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)

    def step(self):
        self.x_nest_old = self.x_nest.copy()
        self.grad = self.loss.gradient(self.x)
        self.x_nest = self.x - self.lr * self.grad

        if not self.strongly_convex:
            alpha_new = 0.5 * (1 + np.sqrt(1 + 4 * self.alpha ** 2))
            self.momentum = (self.alpha - 1) / alpha_new
            self.alpha = alpha_new

        self.x = self.x_nest + self.momentum * (self.x_nest - self.x_nest_old)


class Sgd(StochasticOptimizer):
    def __init__(self, lr0=None, lr_max=np.inf, lr_decay_coef=0, lr_decay_power=1, it_start_decay=None, batch_size=1, avoid_cache_miss=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr0 = lr0 if lr0 is not None else 1 / self.loss.batch_smoothness(batch_size)
        self.lr_max = lr_max
        self.lr_decay_coef = lr_decay_coef
        self.lr_decay_power = lr_decay_power
        self.it_start_decay = it_start_decay or self.it_max // 40
        self.batch_size = batch_size
        self.avoid_cache_miss = avoid_cache_miss

    def step(self):
        idx = np.random.choice(self.loss.n, self.batch_size, replace=self.avoid_cache_miss)
        self.grad = self.loss.stochastic_gradient(self.x, idx=idx)
        lr = 1 / (1 / self.lr0 + self.lr_decay_coef * max(0, self.it - self.it_start_decay)**self.lr_decay_power)
        self.lr = min(lr, self.lr_max)
        # Compute the mean gradient and reshape it to match the shape of self.x
        if self.grad.ndim == 2:
            mean_grad = np.mean(self.grad, axis=1).reshape(-1, 1)
        else:
            mean_grad = self.grad.reshape(-1, 1)
        self.x -= self.lr * mean_grad


class Shuffling(StochasticOptimizer):
    def __init__(self, steps_per_permutation=None, lr0=None, lr_max=np.inf, lr_decay_coef=0, lr_decay_power=1, it_start_decay=None, batch_size=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.steps_per_permutation = steps_per_permutation or math.ceil(self.loss.n / batch_size)
        self.lr0 = lr0 if lr0 is not None else 1 / self.loss.batch_smoothness(batch_size)
        self.lr_max = lr_max
        self.lr_decay_coef = lr_decay_coef
        self.lr_decay_power = lr_decay_power
        self.it_start_decay = it_start_decay or self.it_max // 40
        self.batch_size = batch_size
        self.permutation = None

    def step(self):
        if self.it % self.steps_per_permutation == 0 or self.permutation is None:
            self.permutation = np.random.permutation(self.loss.n)
            self.i = 0

        idx_perm = np.arange(self.i, self.i + self.batch_size) % self.loss.n
        self.i = (self.i + self.batch_size) % self.loss.n
        idx = self.permutation[idx_perm]
        self.grad = self.loss.stochastic_gradient(self.x, idx=idx)

        lr = 1 / (1 / self.lr0 + self.lr_decay_coef * max(0, self.it - self.it_start_decay)**self.lr_decay_power)
        self.lr = min(lr, self.lr_max)

        # Compute the mean gradient and reshape it to match the shape of self.x
        if self.grad.ndim == 2:
            mean_grad = np.mean(self.grad, axis=1).reshape(-1, 1)
        else:
            mean_grad = self.grad.reshape(-1, 1)
        self.x -= self.lr * mean_grad

