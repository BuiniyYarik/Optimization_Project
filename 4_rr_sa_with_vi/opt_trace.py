import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from loss import safe_sparse_norm


class Trace:
    """
    Class that stores the logs of running an optimization method
    and plots the trajectory.
    """
    def __init__(self, loss):
        self.loss = loss
        self.xs = []
        self.ts = []
        self.its = []
        self.loss_vals = None
        self.its_converted_to_epochs = False

    def compute_loss_of_iterates(self):
        if self.loss_vals is None:
            self.loss_vals = np.array([self.loss.value(x) for x in self.xs])

    def convert_its_to_epochs(self, batch_size=1):
        if not self.its_converted_to_epochs:
            self.its = np.array(self.its) / (self.loss.n / batch_size)
            self.its_converted_to_epochs = True

    def plot_losses(self, f_opt=None, markevery=None, *args, **kwargs):
        self.compute_loss_of_iterates()
        f_opt = f_opt if f_opt is not None else np.min(self.loss_vals)
        markevery = markevery or max(1, len(self.loss_vals) // 20)
        plt.plot(self.its, self.loss_vals - f_opt, markevery=markevery, *args, **kwargs)
        plt.ylabel(r'$f(x)-f^*$')

    def plot_distances(self, x_opt=None, markevery=None, *args, **kwargs):
        x_opt = x_opt if x_opt is not None else self.xs[np.argmin(self.loss_vals)]
        markevery = markevery or max(1, len(self.xs) // 20)
        dists = [safe_sparse_norm(x - x_opt) ** 2 for x in self.xs]
        plt.plot(self.its, dists, markevery=markevery, *args, **kwargs)
        plt.ylabel(r'$\Vert x-x^*\Vert^2$')

    def best_loss_value(self):
        self.compute_loss_of_iterates()
        return np.min(self.loss_vals)

    def save(self, file_name, path='./results/'):
        self.loss = None
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(Path(path) / file_name, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, path, loss):
        if Path(path).is_file():
            with open(path, 'rb') as f:
                trace = pickle.load(f)
                trace.loss = loss
                return trace
        return None


class StochasticTrace(Trace):
    """
    Class that stores the logs of running a stochastic
    optimization method and plots the trajectory.
    """
    def __init__(self, loss):
        super().__init__(loss)
        self.xs_all = {}
        self.ts_all = {}
        self.its_all = {}
        self.loss_vals_all = {}

    def append_seed_results(self, seed):
        self.xs_all[seed] = self.xs[:]
        self.ts_all[seed] = self.ts[:]
        self.its_all[seed] = self.its[:]
        self.loss_vals_all[seed] = self.loss_vals[:]

    def compute_loss_of_iterates(self):
        for seed, loss_vals in self.loss_vals_all.items():
            if loss_vals is None:
                self.loss_vals_all[seed] = np.array([self.loss.value(x) for x in self.xs_all[seed]])

    def best_loss_value(self):
        self.compute_loss_of_iterates()
        return min(np.min(vals) for vals in self.loss_vals_all.values())

    def plot_losses(self, f_opt=None, markevery=None, alpha=0.25, *args, **kwargs):
        self.compute_loss_of_iterates()
        f_opt = f_opt if f_opt is not None else self.best_loss_value()
        it_ave = np.mean([np.array(its) for its in self.its_all.values()], axis=0)
        y = [vals - f_opt for vals in self.loss_vals_all.values()]
        y_ave, y_std = np.mean(y, axis=0), np.std(y, axis=0)
        upper, lower = y_ave + y_std, y_ave - y_std
        markevery = markevery or max(1, len(y_ave) // 20)
        plot = plt.plot(it_ave, y_ave, markevery=markevery, *args, **kwargs)
        if len(self.loss_vals_all) > 1:
            plt.fill_between(it_ave, upper, lower, alpha=alpha, color=plot[0].get_color())
        plt.ylabel(r'$f(x)-f^*$')
