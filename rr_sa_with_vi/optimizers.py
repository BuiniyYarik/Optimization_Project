import numpy as np
import scipy
from utils import set_seed, safe_sparse_norm


class Optimizer:
    """
    Base class for optimization algorithms.
    """
    def __init__(self, loss, t_max=np.inf, it_max=np.inf, trace_len=200, tolerance=0):
        self.loss = loss
        self.t_max = t_max
        self.it_max = it_max
        self.trace_len = trace_len
        self.tolerance = tolerance
        self.x = None
        self.it = 0
        self.trace = []

    def step(self):
        raise NotImplementedError

    def run(self, x0):
        self.x = x0.copy()
        while self.it < self.it_max:
            self.step()
            self.trace.append(self.loss.value(self.x))
            self.it += 1


class StochasticOptimizer(Optimizer):
    """
    Base class for stochastic optimization algorithms.
    """
    def __init__(self, loss, n_seeds=1, *args, **kwargs):
        super().__init__(loss, *args, **kwargs)
        self.n_seeds = n_seeds
        self.seeds = [np.random.randint(100000) for _ in range(n_seeds)]

    def run(self, x0):
        traces = []
        for seed in self.seeds:
            set_seed(seed)
            super().run(x0)
            traces.append(self.trace.copy())
            self.trace = []
            self.it = 0
        return traces


class Ig(StochasticOptimizer):
    """
    Incremental gradient descent (IG) optimizer.
    """
    def __init__(self, loss, lr, *args, **kwargs):
        super().__init__(loss, *args, **kwargs)
        self.lr = lr

    def step(self):
        idx = np.random.randint(self.loss.n)
        grad = self.loss.stochastic_gradient(self.x, idx=idx)
        self.x -= self.lr * grad


class Nesterov(Optimizer):
    def __init__(self, loss, lr=None, mu=0, it_max=np.inf, trace_len=200):
        super().__init__(loss, it_max, trace_len)
        self.lr = lr if lr is not None else 1 / loss.smoothness()
        self.mu = mu
        self.y = None
        self.momentum = 0
        self.alpha = 1

    def run(self, x0):
        self.x = x0.copy()
        self.y = self.x.copy()
        if self.mu > 0:
            self.momentum = (1 - np.sqrt(self.lr * self.mu)) / (1 + np.sqrt(self.lr * self.mu))
        while self.it < self.it_max:
            self.step()
            self.trace.append(self.loss.value(self.x))
            self.it += 1

    def step(self):
        grad = self.loss.gradient(self.y)
        x_new = self.y - self.lr * grad
        if self.mu > 0:
            self.x = x_new + self.momentum * (x_new - self.x)
        else:
            alpha_new = 0.5 * (1 + np.sqrt(1 + 4 * self.alpha**2))
            momentum = (self.alpha - 1) / alpha_new
            self.x = x_new + momentum * (x_new - self.x)
            self.alpha = alpha_new
        self.y = x_new


class Sgd(StochasticOptimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.
    """
    def __init__(self, loss, lr, *args, **kwargs):
        super().__init__(loss, *args, **kwargs)
        self.lr = lr

    def step(self):
        grad = self.loss.stochastic_gradient(self.x)
        self.x -= self.lr * grad


class Shuffling(StochasticOptimizer):
    """
    Shuffling-based stochastic gradient descent optimizer.
    """
    def __init__(self, loss, lr, *args, **kwargs):
        super().__init__(loss, *args, **kwargs)
        self.lr = lr
        self.permutation = None

    def step(self):
        if self.permutation is None or self.it % self.loss.n == 0:
            self.permutation = np.random.permutation(self.loss.n)
        idx = self.permutation[self.it % self.loss.n]
        grad = self.loss.stochastic_gradient(self.x, idx=idx)
        self.x -= self.lr * grad

