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
