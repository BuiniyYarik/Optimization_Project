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
