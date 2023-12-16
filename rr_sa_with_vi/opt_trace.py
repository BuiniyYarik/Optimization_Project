import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import os

from loss import safe_sparse_norm


class Trace:
    """
    Class that stores the logs of running an optimization method
    and plots the trajectory.
    """
    def __init__(self, loss):
        """
        Initializes the Trace object with a given loss function.
        Args:
            loss: The loss function to be used.
        """
        self.loss = loss
        self.xs = []
        self.ts = []
        self.its = []
        self.loss_vals = None
        self.its_converted_to_epochs = False
        self.loss_is_computed = False

    def compute_loss_of_iterates(self):
        """
        Computes the loss values for all iterates if not already computed.
        """
        if self.loss_vals is None:
            self.loss_vals = np.asarray([self.loss.value(x) for x in self.xs])
        else:
            print('Loss values have already been computed. Set .loss_vals = None to recompute')

    def convert_its_to_epochs(self, batch_size=1):
        """
        Converts iteration counts to epochs.
        Args:
            batch_size (int, optional): The batch size used in optimization. Defaults to 1.
        """
        its_per_epoch = self.loss.n / batch_size
        if self.its_converted_to_epochs:
            return
        self.its = np.asarray(self.its) / its_per_epoch
        self.its_converted_to_epochs = True

    def plot_losses(self, f_opt=None, *args, **kwargs):
        """
        Plots the losses over iterations or epochs.
        Args:
            f_opt (float, optional): Optimal loss value for reference. Defaults to None.
        """
        if self.loss_vals is None:
            self.compute_loss_of_iterates()
        if f_opt is None:
            f_opt = np.min(self.loss_vals)

        plt.plot(self.its, self.loss_vals - f_opt, *args, **kwargs)
        plt.ylabel(r'$f(x)-f^*$')

    def plot_distances(self, x_opt=None, *args, **kwargs):
        """
        Plots the distances of iterates from the optimal solution.
        Args:
            x_opt (array-like, optional): Optimal solution for reference. Defaults to None.
        """
        if x_opt is None:
            if self.loss_vals is None:
                x_opt = self.xs[-1]
            else:
                i_min = np.argmin(self.loss_vals)
                x_opt = self.xs[i_min]

        dists = [safe_sparse_norm(x - x_opt) ** 2 for x in self.xs]
        plt.plot(self.its, dists, *args, **kwargs)
        plt.ylabel(r'$\Vert x-x^*\Vert^2$')

    def best_loss_value(self):
        """
        Returns the best loss value achieved.
        Returns:
            float: The minimum loss value.
        """
        if not self.loss_is_computed:
            self.compute_loss_of_iterates()
        return np.min(self.loss_vals)

    def save(self, file_name, path='./results/'):
        """
        Saves the Trace object to a file.
        Args:
            file_name (str): The name of the file to save.
            path (str, optional): The directory path to save the file. Defaults to './results/'.
        """
        self.loss = None
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(path + file_name, 'wb') as f:
            pickle.dump(self, f)

    def from_pickle(cls, path, loss):
        """
        Loads a Trace object from a pickle file.
        Args:
            path (str): The path to the pickle file.
            loss: The loss function to be associated with the loaded Trace.
        Returns:
            Trace: The loaded Trace object.
        """
        if not os.path.isfile(path):
            return None
        with open(path, 'rb') as f:
            trace = pickle.load(f)
            trace.loss = loss
        return trace


class StochasticTrace(Trace):
    """
    Class that extends Trace for storing logs of a stochastic optimization method.
    """
    def __init__(self, loss):
        """
        Initializes the StochasticTrace object with a given loss function.
        Args:
            loss: The loss function to be used.
        """
        super().__init__(loss)
        self.xs_all = {}
        self.ts_all = {}
        self.its_all = {}
        self.loss_vals_all = {}

    def init_seed(self):
        """
        Initializes the seed-dependent properties for a new run.
        """
        self.xs = []
        self.ts = []
        self.its = []
        self.loss_vals = None

    def append_seed_results(self, seed):
        """
        Stores the results of the current seed.
        Args:
            seed: The seed value used for the stochastic process.
        """
        self.xs_all[seed] = self.xs.copy()
        self.ts_all[seed] = self.ts.copy()
        self.its_all[seed] = self.its.copy()
        self.loss_vals_all[seed] = self.loss_vals.copy() if self.loss_vals else None

    def compute_loss_of_iterates(self):
        """
        Computes the loss values for all iterates for each seed.
        """
        for seed, loss_vals in self.loss_vals_all.items():
            if loss_vals is None:
                self.loss_vals_all[seed] = np.asarray([self.loss.value(x) for x in self.xs_all[seed]])
            else:
                print(f"Loss values for seed {seed} have already been computed. Set .loss_vals_all[{seed}] = None to recompute")
        self.loss_is_computed = True

    def plot_losses(self, f_opt=None, log_std=True, alpha=0.25, *args, **kwargs):
        """
        Plots the average losses over iterations or epochs for stochastic optimization.
        Args:
            f_opt (float, optional): Optimal loss value for reference. Defaults to None.
            log_std (bool, optional): Whether to plot the logarithm of the standard deviation. Defaults to True.
            alpha (float, optional): Transparency level for the fill. Defaults to 0.25.
        """
        if not self.loss_is_computed:
            self.compute_loss_of_iterates()
        if f_opt is None:
            f_opt = self.best_loss_value()

        it_ave = np.mean([np.asarray(its) for its in self.its_all.values()], axis=0)
        if log_std:
            y_log = [np.log(loss_vals - f_opt) for loss_vals in self.loss_vals_all.values()]
            y_log_ave = np.mean(y_log, axis=0)
            y_log_std = np.std(y_log, axis=0)
            upper, lower = np.exp(y_log_ave + y_log_std), np.exp(y_log_ave - y_log_std)
            y_ave = np.exp(y_log_ave)
        else:
            y = [loss_vals - f_opt for loss_vals in self.loss_vals_all.values()]
            y_ave = np.mean(y, axis=0)
            y_std = np.std(y, axis=0)
            upper, lower = y_ave + y_std, y_ave - y_std

        plot = plt.plot(it_ave, y_ave, *args, **kwargs)
        if len(self.loss_vals_all.keys()) > 1:
            plt.fill_between(it_ave, upper, lower, alpha=alpha, color=plot[0].get_color())
        plt.ylabel(r'$f(x)-f^*$')

    def plot_distances(self, x_opt=None, log_std=True, alpha=0.25, *args, **kwargs):
        """
        Plots the average distances of iterates from the optimal solution for stochastic optimization.
        Args:
            x_opt (array-like, optional): Optimal solution for reference. Defaults to None.
            log_std (bool, optional): Whether to plot the logarithm of the standard deviation. Defaults to True.
            alpha (float, optional): Transparency level for the fill. Defaults to 0.25.
        """
        if x_opt is None:
            if self.loss_is_computed:
                f_opt = np.inf
                for seed, loss_vals in self.loss_vals_all.items():
                    i_min = np.argmin(loss_vals)
                    if loss_vals[i_min] < f_opt:
                        f_opt = loss_vals[i_min]
                        x_opt = self.xs_all[seed][i_min]
                else:
                    x_opt = self.xs[-1]

        it_ave = np.mean([np.asarray(its) for its in self.its_all.values()], axis=0)
        dists = [np.asarray([safe_sparse_norm(x - x_opt) ** 2 for x in xs]) for xs in self.xs_all.values()]
        if log_std:
            y_log = [np.log(dist) for dist in dists]
            y_log_ave = np.mean(y_log, axis=0)
            y_log_std = np.std(y_log, axis=0)
            upper, lower = np.exp(y_log_ave + y_log_std), np.exp(y_log_ave - y_log_std)
            y_ave = np.exp(y_log_ave)
        else:
            y = dists
            y_ave = np.mean(y, axis=0)
            y_std = np.std(y, axis=0)
            upper, lower = y_ave + y_std, y_ave - y_std

        plot = plt.plot(it_ave, y_ave, *args, **kwargs)
        if len(self.loss_vals_all.keys()) > 1:
            plt.fill_between(it_ave, upper, lower, alpha=alpha, color=plot[0].get_color())
        plt.ylabel(r'$\Vert x-x^*\Vert^2$')
