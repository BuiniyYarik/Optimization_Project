import scipy
import math
import numpy as np
import os
import pickle
import random


def safe_sparse_add(a, b):
    """
    Adds two arrays or sparse matrices, ensuring the result remains sparse if both inputs are sparse.

    Args:
        a (numpy.ndarray or scipy.sparse matrix): The first addend.
        b (numpy.ndarray or scipy.sparse matrix): The second addend.

    Returns:
        numpy.ndarray or scipy.sparse matrix: The sum of a and b.
    """
    if scipy.sparse.issparse(a) and scipy.sparse.issparse(b):
        # both are sparse, keep the result sparse
        return a + b
    else:
        # one of them is non-sparse, convert everything to dense.
        if scipy.sparse.issparse(a):
            a = a.toarray()
            if a.ndim == 2 and b.ndim == 1:
                b = b.ravel()
        elif scipy.sparse.issparse(b):
            b = b.toarray()
            if b.ndim == 2 and a.ndim == 1:
                b = b.ravel()
        return a + b


def safe_sparse_dot(a, b):
    """
    Computes the dot product of two arrays or sparse matrices, ensuring the result remains dense.

    Args:
        a (numpy.ndarray or scipy.sparse matrix): The first operand.
        b (numpy.ndarray or scipy.sparse matrix): The second operand.

    Returns:
        numpy.ndarray: The dot product of a and b.
    """
    if scipy.sparse.issparse(a) and scipy.sparse.issparse(b):
        # special cases for sparse dot product
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
    """
    Multiplies two arrays or sparse matrices element-wise, ensuring the result remains sparse if both inputs are sparse.

    Args:
        a (numpy.ndarray or scipy.sparse matrix): The first multiplicand.
        b (numpy.ndarray or scipy.sparse matrix): The second multiplicand.

    Returns:
        numpy.ndarray or scipy.sparse matrix: The element-wise product of a and b.
    """
    if scipy.sparse.issparse(a) and scipy.sparse.issparse(b):
        return a.multiply(b)
    if scipy.sparse.issparse(a):
        a = a.toarray()
    elif scipy.sparse.issparse(b):
        b = b.toarray()
    return np.multiply(a, b)


def safe_sparse_norm(a, ord=None):
    """
    Computes the norm of an array or sparse matrix.

    Args:
        a (numpy.ndarray or scipy.sparse matrix): The input array or sparse matrix.
        ord (int, float, inf, -inf, 'fro', 'nuc', optional): The order of the norm.

    Returns:
        float: The norm of the input array or sparse matrix.
    """
    if scipy.sparse.issparse(a):
        return scipy.sparse.linalg.norm(a, ord=ord)
    return np.linalg.norm(a, ord=ord)


def relative_round(x):
    """
    Rounds a number to three significant figures.

    Args:
        x (float): The number to be rounded.

    Returns:
        float: The rounded number.
    """
    mantissa, exponent = math.frexp(x)
    return round(mantissa, 3) * 2**exponent


def get_trace(path, loss):
    """
    Loads a trace object from a file and sets its loss attribute.

    Args:
        path (str): The path to the file containing the trace object.
        loss: The loss object to be set for the trace.

    Returns:
        object: The loaded trace object with the loss set, or None if the file does not exist.
    """
    if not os.path.isfile(path):
        return None
    with open(path, 'rb') as f:
        trace = pickle.load(f)
        trace.loss = loss
    return trace


def set_seed(seed=42):
    """
    Sets the random seed for the random, numpy, and os modules to ensure reproducibility.

    Args:
        seed (int): The seed value to set. Defaults to 42.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
