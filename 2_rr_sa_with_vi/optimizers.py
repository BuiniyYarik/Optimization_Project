import numpy as np
import math
import numpy.linalg as la
import scipy
import time

from loss import safe_sparse_norm
from opt_trace import Trace, StochasticTrace
from utils import set_seed


class Optimizer:
    """
    Base class for optimization algorithms. Provides methods to run them,
    save the trace and plot the results.
    """
    def __init__(self, loss, t_max=np.inf, it_max=np.inf, trace_len=200, tolerance=0):
        if t_max is np.inf and it_max is np.inf:
            it_max = 100
            print('The number of iterations is set to 100.')
        self.loss = loss
        self.t_max = t_max
        self.it_max = it_max
        self.trace_len = trace_len
        self.tolerance = tolerance
        self.initialized = False
        self.x_old = None
        self.trace = Trace(loss=loss)

    def run(self, x0):
        if not self.initialized:
            self.init_run(x0)
            self.initialized = True

        while not self.check_convergence():
            if self.tolerance > 0:
                self.x_old = self.x.copy()
            self.step()
            self.save_checkpoint()
            assert scipy.sparse.issparse(self.x) or np.isfinite(self.x).all()

        return self.trace

    def check_convergence(self):
        no_it_left = self.it >= self.it_max
        no_time_left = time.time()-self.t_start >= self.t_max
        if self.tolerance > 0:
            tolerance_met = self.x_old is not None and safe_sparse_norm(self.x-self.x_old) < self.tolerance
        else:
            tolerance_met = False
        return no_it_left or no_time_left or tolerance_met

    def step(self):
        pass

    def init_run(self, x0):
        self.dim = x0.shape[0]
        self.x = x0.copy()
        self.trace.xs = [x0.copy()]
        self.trace.its = [0]
        self.trace.ts = [0]
        self.it = 0
        self.t = 0
        self.t_start = time.time()
        self.time_progress = 0
        self.iterations_progress = 0
        self.max_progress = 0

    def save_checkpoint(self, first_iterations=10):
        self.it += 1
        self.t = time.time() - self.t_start
        self.time_progress = int((self.trace_len-first_iterations) * self.t / self.t_max)
        self.iterations_progress = int((self.trace_len-first_iterations) * (self.it / self.it_max))
        if (max(self.time_progress, self.iterations_progress) > self.max_progress) or (self.it <= first_iterations):
            self.update_trace()
        self.max_progress = max(self.time_progress, self.iterations_progress)

    def update_trace(self):
        self.trace.xs.append(self.x.copy())
        self.trace.ts.append(self.t)
        self.trace.its.append(self.it)


class StochasticOptimizer(Optimizer):
    """
    Base class for stochastic optimization algorithms.
    The class has the same methods as Optimizer and, in addition, uses
    multiple seeds to run the experiments.
    """
    def __init__(self, loss, n_seeds=1, seeds=None, *args, **kwargs):
        super(StochasticOptimizer, self).__init__(loss=loss, *args, **kwargs)
        self.seeds = seeds
        if not seeds:
            np.random.seed(42)
            self.seeds = [np.random.randint(100000) for _ in range(n_seeds)]
        self.finished_seeds = []
        self.trace = StochasticTrace(loss=loss)

    def run(self, *args, **kwargs):
        for seed in self.seeds:
            if seed in self.finished_seeds:
                continue
            set_seed(seed)
            self.trace.init_seed()
            super(StochasticOptimizer, self).run(*args, **kwargs)
            self.trace.append_seed_results(seed)
            self.finished_seeds.append(seed)
            self.initialized = False
        return self.trace


class Ig(Optimizer):
    """
    Incremental gradient descent (IG) with decreasing or constant learning rate.

    Arguments:
        lr0 (float, optional): an estimate of the inverse maximal smoothness constant
    """
    def __init__(self, lr0=None, lr_max=np.inf, lr_decay_coef=0, lr_decay_power=1,
                 it_start_decay=None, batch_size=1, *args, **kwargs):
        super(Ig, self).__init__(*args, **kwargs)
        self.lr0 = lr0
        self.lr_max = lr_max
        self.lr_decay_coef = lr_decay_coef
        self.lr_decay_power = lr_decay_power
        self.it_start_decay = it_start_decay
        if it_start_decay is None and np.isfinite(self.it_max):
            self.it_start_decay = self.it_max // 40 if np.isfinite(self.it_max) else 0
        self.batch_size = batch_size

    def step(self):
        idx = np.arange(self.i, self.i + self.batch_size)
        idx %= self.loss.n
        self.i += self.batch_size
        self.i %= self.loss.n
        self.grad = self.loss.stochastic_gradient(self.x, idx=idx)
        denom_const = 1 / self.lr0
        lr_decayed = 1 / (denom_const + self.lr_decay_coef*max(0, self.it-self.it_start_decay)**self.lr_decay_power)
        if lr_decayed < 0:
            lr_decayed = np.inf
        self.lr = min(lr_decayed, self.lr_max)
        self.x -= self.lr * self.grad

    def init_run(self, *args, **kwargs):
        super(Ig, self).init_run(*args, **kwargs)
        if self.lr0 is None:
            self.lr0 = 1 / self.loss.batch_smoothness(batch_size)
        self.i = 0


class Nesterov(Optimizer):
    """
    Accelerated gradient descent with constant learning rate.
    For details, see, e.g., http://mpawankumar.info/teaching/cdt-big-data/nesterov83.pdf

    Arguments:
        lr (float, optional): an estimate of the inverse smoothness constant
    """
    def __init__(self, lr=None, strongly_convex=False, mu=0, *args, **kwargs):
        super(Nesterov, self).__init__(*args, **kwargs)
        self.lr = lr
        if mu < 0:
            raise ValueError("Invalid mu: {}".format(mu))
        if strongly_convex and mu == 0:
            raise ValueError("""Mu must be larger than 0 for strongly_convex=True,
                             invalid value: {}""".format(mu))
        self.strongly_convex = strongly_convex
        if self.strongly_convex:
            self.mu = mu

    def step(self):
        if not self.strongly_convex:
            alpha_new = 0.5 * (1 + np.sqrt(1 + 4*self.alpha**2))
            self.momentum = (self.alpha - 1) / alpha_new
            self.alpha = alpha_new
        self.x_nest_old = self.x_nest.copy()
        self.grad = self.loss.gradient(self.x)
        self.x_nest = self.x - self.lr*self.grad
        self.x = self.x_nest + self.momentum*(self.x_nest-self.x_nest_old)

    def init_run(self, *args, **kwargs):
        super(Nesterov, self).init_run(*args, **kwargs)
        if self.lr is None:
            self.lr = 1 / self.loss.smoothness()
        self.x_nest = self.x.copy()
        if self.strongly_convex:
            kappa = (1/self.lr)/self.mu
            self.momentum = (np.sqrt(kappa)-1) / (np.sqrt(kappa)+1)
        else:
            self.alpha = 1.0


class Sgd(StochasticOptimizer):
    """
    Stochastic gradient descent with decreasing or constant learning rate.

    Arguments:
        lr (float, optional): an estimate of the inverse smoothness constant
    """
    def __init__(self, lr0=None, lr_max=np.inf, lr_decay_coef=0, lr_decay_power=1, it_start_decay=None,
                 batch_size=1, avoid_cache_miss=True, *args, **kwargs):
        super(Sgd, self).__init__(*args, **kwargs)
        self.lr0 = lr0
        self.lr_max = lr_max
        self.lr_decay_coef = lr_decay_coef
        self.lr_decay_power = lr_decay_power
        self.it_start_decay = it_start_decay
        if it_start_decay is None and np.isfinite(self.it_max):
            self.it_start_decay = self.it_max // 40 if np.isfinite(self.it_max) else 0
        self.batch_size = batch_size
        self.avoid_cache_miss = avoid_cache_miss

    def step(self):
        if self.avoid_cache_miss:
            i = np.random.choice(self.loss.n)
            idx = np.arange(i, i + self.batch_size)
            idx %= self.loss.n
            self.grad = self.loss.stochastic_gradient(self.x, idx=idx)
        else:
            self.grad = self.loss.stochastic_gradient(self.x, batch_size=self.batch_size)
        denom_const = 1 / self.lr0
        lr_decayed = 1 / (denom_const + self.lr_decay_coef*max(0, self.it-self.it_start_decay)**self.lr_decay_power)
        if lr_decayed < 0:
            lr_decayed = np.inf
        self.lr = min(lr_decayed, self.lr_max)
        self.x -= self.lr * self.grad

    def init_run(self, *args, **kwargs):
        super(Sgd, self).init_run(*args, **kwargs)
        if self.lr0 is None:
            self.lr0 = 1 / self.loss.batch_smoothness(batch_size)


class Shuffling(StochasticOptimizer):
    """
    Shuffling-based stochastic gradient descent with decreasing or constant learning rate.

    Arguments:
        lr (float, optional): an estimate of the inverse smoothness constant
    """
    def __init__(self, steps_per_permutation=None, lr0=None, lr_max=np.inf, lr_decay_coef=0,
                 lr_decay_power=1, it_start_decay=None, batch_size=1, *args, **kwargs):
        super(Shuffling, self).__init__(*args, **kwargs)
        self.steps_per_permutation = steps_per_permutation if steps_per_permutation else math.ceil(self.loss.n/batch_size)
        self.lr0 = lr0
        self.lr_max = lr_max
        self.lr_decay_coef = lr_decay_coef
        self.lr_decay_power = lr_decay_power
        self.it_start_decay = it_start_decay
        if it_start_decay is None and np.isfinite(self.it_max):
            self.it_start_decay = self.it_max // 40 if np.isfinite(self.it_max) else 0
        self.batch_size = batch_size
        self.sampled_permutations = 0

    def step(self):
        if self.it%self.steps_per_permutation == 0:
            self.permutation = np.random.permutation(self.loss.n)
            self.i = 0
            self.sampled_permutations += 1
        if self.steps_per_permutation is np.inf:
            idx_perm = np.arange(self.i, self.i + self.batch_size)
            idx_perm %= self.loss.n
            normalization = self.batch_size
        else:
            idx_perm = np.arange(self.i, min(self.loss.n, self.i+self.batch_size))
            normalization = self.loss.n / self.steps_per_permutation #works only for RR
        idx = self.permutation[idx_perm]
        self.i += self.batch_size
        self.i %= self.loss.n
        # since the objective is 1/n sum_{i=1}^n f_i(x) + l2/2*||x||^2
        # any incomplete minibatch should be normalized by batch_size
        self.grad = self.loss.stochastic_gradient(self.x, idx=idx, normalization=normalization)
        denom_const = 1 / self.lr0
        lr_decayed = 1 / (denom_const + self.lr_decay_coef*max(0, self.it-self.it_start_decay)**self.lr_decay_power)
        self.lr = min(lr_decayed, self.lr_max)
        self.x -= self.lr * self.grad

    def init_run(self, *args, **kwargs):
        super(Shuffling, self).init_run(*args, **kwargs)
        if self.lr0 is None:
            self.lr0 = 1 / self.loss.batch_smoothness(batch_size)