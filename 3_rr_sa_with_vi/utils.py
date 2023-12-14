import numpy as np
import scipy


def safe_sparse_add(a, b):
    if scipy.sparse.issparse(a) and scipy.sparse.issparse(b):
        # both are sparse, keep the result sparse
        return a + b
    else:
        # one of them is non-sparse, convert
        # everything to dense.
        if scipy.sparse.issparse(a):
            a = a.toarray()
            if a.ndim == 2 and b.ndim == 1:
                b.ravel()
        elif scipy.sparse.issparse(b):
            b = b.toarray()
            if b.ndim == 2 and a.ndim == 1:
                b = b.ravel()
        return a + b


def safe_sparse_dot(a, b):
    if scipy.sparse.issparse(a) and scipy.sparse.issparse(b):

        if a.shape[1] == b.shape[0]:
            return (a @ b)[0, 0]
        if a.shape[0] == b.shape[0]:
            return (a.T @ b)[0, 0]
        return (a @ b.T)[0, 0]
    if scipy.sparse.issparse(a):
        a = a.toarray()
    elif scipy.sparse.issparse(b):
        b = b.toarray()
    return a @ b


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


import math
import numpy as np
import os
import pickle
import random



def relative_round(x):
    """
    A util that rounds the input to the most significant digits.
    Useful for storing the results as rounding float
    numbers may cause file name ambiguity.
    """
    mantissa, exponent = math.frexp(x)
    return round(mantissa, 3) * 2**exponent


def get_trace(path, loss):
    if not os.path.isfile(path):
        return None
    f = open(path, 'rb')
    trace = pickle.load(f)
    trace.loss = loss
    f.close()
    return trace


def set_seed(seed=42):
    """
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def variance_at_opt(x_opt, loss, batch_size=1, perms=None, n_perms=1, L_batch=None, lr=None):
    if L_batch is None:
        # for simplicity, we use max smoothness, but one might want to use batch smoothness instead
        L_batch = loss.max_smoothness()
    if lr is None:
        lr = 1 / L_batch
    if perms is None:
        perms = [np.random.permutation(loss.n) for _ in range(n_perms)]
    else:
        n_perms = len(perms)
    variance_sgd = 0
    variances = None
    for permutation in perms:
        grad_sum = 0
        start_idx = range(0, loss.n, batch_size)
        n_grads = len(start_idx)
        if variances is None:
            variances = np.zeros(n_grads)
        for e, i in enumerate(start_idx):
            idx = permutation[np.arange(i, min(loss.n, i + batch_size))]
            stoch_grad = loss.stochastic_gradient(x_opt, idx=idx)
            variance_sgd += safe_sparse_norm(stoch_grad)**2 / n_grads / n_perms

            x = x_opt - lr * grad_sum
            loss_x = loss.partial_value(x, idx)
            loss_x_opt = loss.partial_value(x_opt, idx)
            linear_dif = safe_sparse_dot(stoch_grad, x - x_opt)
            bregman_div = loss_x - loss_x_opt - linear_dif
            variances[e] += bregman_div / n_perms / lr
            grad_sum += stoch_grad
    variance_rr = np.max(variances)
    variance_rr_upper = variance_sgd * n_grads * lr * L_batch / 4
    variance_rr_lower = variance_sgd * n_grads * lr * loss.l2 / 8
    return variance_sgd, variance_rr, variance_rr_upper, variance_rr_lower
