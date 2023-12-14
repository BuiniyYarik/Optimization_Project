import scipy
import numpy as np
import numpy.linalg as la
from sklearn.utils.extmath import row_norms
from utils import safe_sparse_add, safe_sparse_multiply, safe_sparse_norm

def logsig(x):
    """
    Compute the log-sigmoid function component-wise.
    """
    return np.where(x < -33, x,
                    np.where(x < -18, x - np.exp(x),
                             np.where(x < 37, -np.log1p(np.exp(-x)), -np.exp(-x))))

class LogisticRegression:
    """
    Logistic regression class that returns loss values, gradients, and Hessians.
    """
    def __init__(self, A, b, l1=0, l2=0):
        self.A = A
        self.l1 = l1
        self.l2 = l2
        self.n, self.dim = A.shape
        self.b = self.transform_labels(b)
        self.x_last = None
        self.mat_vec_prod = None

    @staticmethod
    def transform_labels(b):
        """
        Transform labels to binary format.
        """
        unique_b = np.unique(b)
        if set(unique_b) == {1, 2}:
            return b - 1
        elif set(unique_b) == {-1, 1}:
            return (b + 1) / 2
        elif set(unique_b) == {0, 1}:
            return b
        else:
            raise ValueError("Invalid label values in b.")

    def value(self, x):
        """
        Compute the value of the logistic regression loss function.
        """
        z = self.mat_vec_product(x)
        regularization = self.l1 * safe_sparse_norm(x, ord=1) + self.l2 / 2 * safe_sparse_norm(x) ** 2
        return np.mean(safe_sparse_multiply(1 - self.b, z) - logsig(z)) + regularization

    def gradient(self, x):
        """
        Compute the gradient of the logistic regression loss function.
        """
        z = self.mat_vec_product(x)
        activation = scipy.special.expit(z)
        grad = safe_sparse_add(self.A.T @ (activation - self.b) / self.n, self.l2 * x)
        return scipy.sparse.csr_matrix(grad).T if scipy.sparse.issparse(x) else grad

    def mat_vec_product(self, x):
        """
        Efficient matrix-vector product computation.
        """
        if self.x_last is None or not np.allclose(x.toarray(), self.x_last.toarray()):
            z = self.A @ x
            self.mat_vec_prod = z.toarray().ravel() if scipy.sparse.issparse(z) else z
            self.x_last = x.copy()
        return self.mat_vec_prod

    def smoothness(self):
        """
        Compute the smoothness constant.
        """
        if scipy.sparse.issparse(self.A):
            sing_val_max = scipy.sparse.linalg.svds(self.A, k=1, return_singular_vectors=False)[0]
            return 0.25 * sing_val_max ** 2 / self.n + self.l2
        else:
            covariance = self.A.T @ self.A / self.n
            return 0.25 * np.max(la.eigvalsh(covariance)) + self.l2

    def max_smoothness(self):
        """
        Compute the maximum smoothness constant.
        """
        max_squared_sum = row_norms(self.A, squared=True).max()
        return 0.25 * max_squared_sum + self.l2

    def stochastic_gradient(self, x, idx=None, batch_size=1, replace=False, normalization=None):
        """
        Compute the stochastic gradient.
        """
        if idx is None:
            idx = np.random.choice(self.n, size=batch_size, replace=replace)
        normalization = batch_size if normalization is None else normalization
        z = self.A[idx] @ x
        activation = scipy.special.expit(z.toarray().ravel() if scipy.sparse.issparse(z) else z)
        return safe_sparse_add(self.A[idx].T @ (activation - self.b[idx]) / normalization, self.l2 * x)

    def batch_smoothness(self, batch_size):
        """
        Smoothness constant of stochastic gradients sampled in minibatches.
        """
        L = self.smoothness()
        L_max = self.max_smoothness()
        return self.n / (self.n - 1) * (1 - 1 / batch_size) * L + (self.n / batch_size - 1) / (self.n - 1) * L_max
