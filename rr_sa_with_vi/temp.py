# Import necessary libraries
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
from scipy.sparse import csc_matrix
from model import create_logistic_regression_model
from train import train_model
from get_dataset import get_dataset
from utils import get_trace, relative_round

# Set up visualization styles
sns.set(style="whitegrid", font_scale=1.2, context="talk", palette=sns.color_palette("bright"), color_codes=False)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.figsize'] = (8, 6)

# Load dataset
dataset = 'w8a.txt'
A, b = get_dataset(dataset)

# Create logistic regression model
model = create_logistic_regression_model(A, b)

# Set training parameters
n, dim = A.shape
L = model.smoothness()
l2 = L / np.sqrt(n)
model.l2 = l2
x0 = csc_matrix((dim, 1))
n_epoch = 600
batch_size = 512
n_seeds = 2  # was set to 20 in the paper
stoch_it = 250 * n // batch_size
trace_len = 300
trace_path = f'results/log_reg_{dataset}_l2_{relative_round(l2)}/'

# Train with different optimizers
optimizers = ['Nesterov', 'Sgd', 'Ig', 'Shuffling']
traces = []
labels = ['Nesterov', 'SGD', 'IG', 'Shuffling']
markers = [',', 'o', 'D', '*']

for opt in optimizers:
    trace = get_trace(f'{trace_path}{opt}')
    if not trace:
        trained_model, trace = train_model(model, opt, x0, n_epoch, 1/l2, trace_path=f'{trace_path}{opt}')
    traces.append(trace)

# Plotting the results
f_opt = np.min([np.min(trace.loss_vals) for trace in traces])
x_opt = trace.xs[-1]

for trace, label, marker in zip(traces, labels, markers):
    trace.plot_losses(f_opt=f_opt, label=label, marker=marker)
plt.yscale('log')
plt.legend()
plt.xlabel('Data passes')
plt.tight_layout()
plt.savefig(f'./plots/{dataset}_func.pdf', dpi=300)

for trace, label, marker in zip(traces, labels, markers):
    trace.plot_distances(x_opt=x_opt, label=label, marker=marker)
plt.yscale('log')
plt.legend()
plt.xlabel('Data passes')
plt.tight_layout()
plt.savefig(f'./plots/{dataset}_dist.pdf', dpi=300)

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

    def init_run(self, x0):
        super().init_run(x0)
        self.y = self.x.copy()

        # Initialize momentum for strongly convex case
        if self.mu > 0:
            self.momentum = (1 - np.sqrt(self.lr * self.mu)) / (1 + np.sqrt(self.lr * self.mu))

    def step(self):
        grad = self.loss.gradient(self.y)

        # Ensure grad is a column vector before subtraction
        if grad.ndim == 1:
            grad = grad.reshape(-1, 1)

        x_new = self.y - self.lr * grad

        if self.mu > 0:
            # Update with momentum for strongly convex case
            self.x = x_new + self.momentum * (x_new - self.x)
        else:
            # Update without momentum for non-strongly convex case
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

import numpy as np
import scipy
from sklearn.utils.extmath import safe_sparse_dot


def safe_sparse_add(a, b):
    if scipy.sparse.issparse(a) and scipy.sparse.issparse(b):
        return a + b
    else:
        if scipy.sparse.issparse(a):
            a = a.toarray()
        elif scipy.sparse.issparse(b):
            b = b.toarray()
        return a + b


def safe_sparse_multiply(a, b):
    if scipy.sparse.issparse(a) and scipy.sparse.issparse(b):
        return a.multiply(b)
    if scipy.sparse.issparse(a):
        a = a.toarray()
    elif scipy.sparse.issparse(b):
        b = b.toarray()
    return np.multiply(a, b)


def safe_sparse_norm(a, ord=None):
    if scipy.sparse.issparse(a):
        return scipy.sparse.linalg.norm(a, ord=ord)
    return np.linalg.norm(a, ord=ord)


def logsig(x):
    out = np.zeros_like(x)
    idx0 = x < -33
    out[idx0] = x[idx0]
    idx1 = (x >= -33) & (x < -18)
    out[idx1] = x[idx1] - np.exp(x[idx1])
    idx2 = (x >= -18) & (x < 37)
    out[idx2] = -np.log1p(np.exp(-x[idx2]))
    idx3 = x >= 37
    out[idx3] = -np.exp(-x[idx3])
    return out


class Oracle:
    def __init__(self, l1=0, l2=0):
        if l1 < 0.0 or l2 < 0.0:
            raise ValueError("Invalid values for l1 or l2 regularization.")
        self.l1 = l1
        self.l2 = l2

    def value(self, x):
        raise NotImplementedError

    def gradient(self, x):
        raise NotImplementedError

    def hessian(self, x):
        raise NotImplementedError

    def norm(self, x):
        raise NotImplementedError

    def smoothness(self):
        raise NotImplementedError

    def max_smoothness(self):
        raise NotImplementedError

    def average_smoothness(self):
        raise NotImplementedError


class LogisticRegression(Oracle):
    def __init__(self, A, b, store_mat_vec_prod=True, l1=0, l2=0):
        super().__init__(l1, l2)
        self.A = A
        b = np.asarray(b)
        if (np.unique(b) == [1, 2]).all():
            self.b = b - 1
        elif (np.unique(b) == [-1, 1]).all():
            self.b = (b + 1) / 2
        else:
            assert (np.unique(b) == [0, 1]).all()
            self.b = b
        self.n, self.dim = A.shape
        self.store_mat_vec_prod = store_mat_vec_prod
        self.x_last = None
        self.mat_vec_prod = np.zeros(self.n)

    def mat_vec_product(self, x):
        if not self.store_mat_vec_prod or self.x_last is None or np.any(x != self.x_last):
            result = self.A.dot(x)
            if scipy.sparse.issparse(result):
                result = result.toarray()  # Convert to dense array if result is sparse
            self.mat_vec_prod = result.ravel()
            self.x_last = x.copy()
        return self.mat_vec_prod

    def value(self, x):
        z = self.mat_vec_product(x)
        regularization = self.l1 * safe_sparse_norm(x, ord=1) + self.l2 / 2 * safe_sparse_norm(x) ** 2
        return np.mean(safe_sparse_multiply(1 - self.b, z) - logsig(z)) + regularization

    def gradient(self, x):
        z = self.mat_vec_product(x)
        activation = scipy.special.expit(z)

        # Ensure that 'activation' and 'self.b' are both 1D arrays before subtraction
        if activation.ndim > 1:
            activation = activation.ravel()
        if self.b.ndim > 1:
            self.b = self.b.ravel()

        # Perform the subtraction and dot product
        grad = self.A.T.dot(activation - self.b) / self.n + self.l2 * x

        # Flatten the gradient to a 1D array if it's not already
        if grad.ndim > 1:
            grad = grad.ravel()

        return grad

    def hessian(self, x):
        z = self.mat_vec_product(x)
        activation = scipy.special.expit(z)
        D = np.diag(activation * (1 - activation))
        return self.A.T.dot(D).dot(self.A) / self.n + self.l2 * np.eye(self.dim)

    def stochastic_gradient(self, x, idx=None, batch_size=1, replace=False):
        if idx is None:
            idx = np.random.choice(self.n, size=batch_size, replace=replace)
        z = self.A[idx].dot(x)
        activation = scipy.special.expit(z)
        stoch_grad = self.A[idx].T.dot(activation - self.b[idx]) / batch_size
        return stoch_grad + self.l2 * x

    def smoothness(self):
        """
        Compute the smoothness constant for logistic regression.
        For logistic regression, the smoothness constant is typically
        calculated as 0.25 * max eigenvalue of (A.T @ A) / n + l2, where
        A is the feature matrix.
        """
        if scipy.sparse.issparse(self.A):
            # If A is sparse, use sparse matrix operations for efficiency
            eigenvalues = scipy.sparse.linalg.eigs(self.A.T.dot(self.A) / self.n, k=1, return_eigenvectors=False)
            max_eigenvalue = np.abs(eigenvalues[0])
        else:
            # If A is dense, use standard numpy operations
            covariance_matrix = self.A.T.dot(self.A) / self.n
            max_eigenvalue = np.max(np.linalg.eigvalsh(covariance_matrix))

        return 0.25 * max_eigenvalue + self.l2

from loss import LogisticRegression


def create_logistic_regression_model(A, b, l1=0, l2=0):
    """
    Create a logistic regression model with given parameters.

    :param A: Feature matrix
    :param b: Label vector
    :param l1: L1 regularization parameter
    :param l2: L2 regularization parameter
    :return: Instance of LogisticRegression
    """
    return LogisticRegression(A, b, l1=l1, l2=l2)

import numpy as np
import os
import pickle
import random
import scipy.sparse


def set_seed(seed=42):
    """
    Set the seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def safe_sparse_add(a, b):
    """
    Safely add two matrices, supporting sparse matrices.
    """
    if scipy.sparse.issparse(a) and scipy.sparse.issparse(b):
        return a + b
    else:
        if scipy.sparse.issparse(a):
            a = a.toarray()
        elif scipy.sparse.issparse(b):
            b = b.toarray()
        return a + b


def safe_sparse_dot(a, b):
    """
    Safely compute dot product, supporting sparse matrices.
    """
    if scipy.sparse.issparse(a):
        a = a.toarray()
    if scipy.sparse.issparse(b):
        b = b.toarray()
    return np.dot(a, b)


def safe_sparse_norm(a, ord=None):
    """
    Compute the norm of an array, supporting sparse matrices.
    """
    if scipy.sparse.issparse(a):
        return scipy.sparse.linalg.norm(a, ord=ord)
    return np.linalg.norm(a, ord=ord)


def relative_round(x, precision=3):
    """
    Round a number to a relative precision.
    """
    return round(x, precision - int(np.floor(np.log10(abs(x)))) - 1)


def get_trace(file_path):
    """
    Load a trace from a file.
    """
    if not os.path.isfile(file_path):
        return None
    with open(file_path, 'rb') as file:
        trace = pickle.load(file)
    return trace


def save_trace(trace, file_path):
    """
    Save a trace to a file.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(trace, file)


def variance_at_opt(x_opt, loss, batch_size=1, n_perms=1, lr=None):
    """
    Calculate variance at the optimum.
    """
    if lr is None:
        lr = 1 / loss.smoothness()

    perms = [np.random.permutation(loss.n) for _ in range(n_perms)]
    variances = []

    for perm in perms:
        grad_sum = 0
        variance = 0

        for i in range(0, loss.n, batch_size):
            idx = perm[i:i + batch_size]
            stoch_grad = loss.stochastic_gradient(x_opt, idx=idx)
            grad_sum += stoch_grad

            x = x_opt - lr * grad_sum
            variance += np.sum((loss.partial_value(x, idx) - loss.partial_value(x_opt, idx)) ** 2)

        variances.append(variance / loss.n)

    return np.mean(variances)

from optimizers import Ig, Nesterov, Sgd, Shuffling
from utils import get_trace, save_trace


def train_model(model, optimizer_type, x0, it_max, lr, mu=None, trace_path=None):
    """
    Train the model using the specified optimizer.

    :param model: The model to be trained
    :param optimizer_type: The type of optimizer to use ('Ig', 'Nesterov', 'Sgd', 'Shuffling')
    :param x0: Initial point for optimization
    :param it_max: Maximum number of iterations
    :param lr: Learning rate
    :param mu: Momentum parameter (used only for Nesterov's optimizer)
    :param trace_path: Path to save the training trace
    :return: Trained model and trace
    """
    optimizer = None

    if optimizer_type == 'Ig':
        optimizer = Ig(loss=model, lr=lr, it_max=it_max)
    elif optimizer_type == 'Nesterov':
        optimizer = Nesterov(loss=model, lr=lr, mu=mu, it_max=it_max)
    elif optimizer_type == 'Sgd':
        optimizer = Sgd(loss=model, lr=lr, it_max=it_max)
    elif optimizer_type == 'Shuffling':
        optimizer = Shuffling(loss=model, lr=lr, it_max=it_max)

    if optimizer is not None:
        trace = optimizer.run(x0)
        if trace_path:
            save_trace(trace, trace_path)
        return model, trace
    else:
        raise ValueError("Invalid optimizer type")