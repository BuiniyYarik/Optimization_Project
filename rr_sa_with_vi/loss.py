import scipy
import numpy as np
import numpy.linalg as la
from sklearn.utils.extmath import row_norms
from utils import safe_sparse_add, safe_sparse_multiply, safe_sparse_norm


def logsig(x):
    """
    Compute the log-sigmoid function component-wise.

    Args:
        x (numpy.ndarray): The input array.

    Returns:
        numpy.ndarray: The log-sigmoid of the input array.
    """
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


class LogisticRegressionLoss:
    """
    Logistic regression loss class that returns loss values, gradients, and Hessians.

    Args:
        A (numpy.ndarray or scipy.sparse matrix): The feature matrix.
        b (numpy.ndarray): The response vector.
        store_mat_vec_prod (bool): Whether to store the matrix-vector product. Defaults to True.
        l1 (float): L1 regularization parameter. Defaults to 0.
        l2 (float): L2 regularization parameter. Defaults to 0.
    """
    def __init__(self, A, b, store_mat_vec_prod=True, l1=0, l2=0):
        self.A = A
        self.l1 = l1
        self.l2 = l2
        b = np.asarray(b)
        # Handling different label conventions
        if (np.unique(b) == [1, 2]).all():
            self.b = b - 1
        elif (np.unique(b) == [-1, 1]).all():
            self.b = (b + 1) / 2
        else:
            assert (np.unique(b) == [0, 1]).all()
            self.b = b
        self.n, self.dim = A.shape
        self.store_mat_vec_prod = store_mat_vec_prod
        self.x_last = 0.
        self.mat_vec_prod = np.zeros(self.n)

    def value(self, x):
        """
        Computes the value of the logistic regression loss function.

        Args:
            x (numpy.ndarray): The point at which to evaluate the loss.

        Returns:
            float: The value of the loss function.
        """
        z = self.mat_vec_product(x)
        regularization = self.l1 * safe_sparse_norm(x, ord=1) + self.l2 / 2 * safe_sparse_norm(x) ** 2
        return np.mean(safe_sparse_multiply(1 - self.b, z) - logsig(z)) + regularization

    def gradient(self, x):
        """
        Computes the gradient of the logistic regression loss function.

        Args:
            x (numpy.ndarray): The point at which to compute the gradient.

        Returns:
            numpy.ndarray: The gradient of the loss function.
        """
        z = self.mat_vec_product(x)
        activation = scipy.special.expit(z)
        grad = safe_sparse_add(self.A.T @ (activation - self.b) / self.n, self.l2 * x)
        if scipy.sparse.issparse(x):
            grad = scipy.sparse.csr_matrix(grad).T
        return grad

        def stochastic_gradient(self, x, idx=None, batch_size=1, replace=False, normalization=None):
            """
            Computes the stochastic gradient of the logistic regression loss function for a subset or random batch.

            Args:
                x (numpy.ndarray): The point at which to compute the gradient.
                idx (array-like, optional): Indices for the subset to be used. Defaults to None.
                batch_size (int, optional): The size of the random batch if idx is None. Defaults to 1.
                replace (bool, optional): Whether to sample with replacement. Defaults to False.
                normalization (float, optional): Normalization factor for the gradient. Defaults to None.

            Returns:
                numpy.ndarray: The stochastic gradient of the loss function.
            """
        if idx is None:
            idx = np.random.choice(self.n, size=batch_size, replace=replace)
        else:
            batch_size = 1 if np.isscalar(idx) else len(idx)
        if normalization is None:
            normalization = batch_size
        z = self.A[idx] @ x
        if scipy.sparse.issparse(z):
            z = z.toarray().ravel()
        activation = scipy.special.expit(z)
        stoch_grad = safe_sparse_add(self.A[idx].T @ (activation - self.b[idx]) / normalization, self.l2 * x)
        if scipy.sparse.issparse(x):
            stoch_grad = scipy.sparse.csr_matrix(stoch_grad).T
        return stoch_grad

    def hessian(self, x):
        """
        Computes the Hessian of the logistic regression loss function.

        Args:
            x (numpy.ndarray): The point at which to compute the Hessian.

        Returns:
            numpy.ndarray: The Hessian matrix of the loss function.
        """
        z = self.mat_vec_product(x)
        activation = scipy.special.expit(z)
        weights = activation * (1 - activation)
        A_weighted = safe_sparse_multiply(self.A.T, weights)
        return A_weighted @ self.A / self.n + self.l2 * np.eye(self.dim)

    def stochastic_hessian(self, x, idx=None, batch_size=1, replace=False, normalization=None):
        """
        Computes the stochastic Hessian of the logistic regression loss function for a subset or random batch.

        Args:
            x (numpy.ndarray): The point at which to compute the stochastic Hessian.
            idx (array-like, optional): Indices for the subset to be used. Defaults to None.
            batch_size (int, optional): The size of the random batch if idx is None. Defaults to 1.
            replace (bool, optional): Whether to sample with replacement. Defaults to False.
            normalization (float, optional): Normalization factor for the Hessian. Defaults to None.

        Returns:
            numpy.ndarray: The stochastic Hessian matrix of the loss function.
        """
        if idx is None:
            idx = np.random.choice(self.n, size=batch_size, replace=replace)
        else:
            batch_size = 1 if np.isscalar(idx) else len(idx)
        if normalization is None:
            normalization = batch_size
        z = self.A[idx] @ x
        if scipy.sparse.issparse(z):
            z = z.toarray().ravel()
        activation = scipy.special.expit(z)
        weights = activation * (1 - activation)
        A_weighted = safe_sparse_multiply(self.A[idx].T, weights)
        return A_weighted @ self.A[idx] / normalization + self.l2 * np.eye(self.dim)

    def mat_vec_product(self, x):
        """
        Computes the matrix-vector product of A and x, with optional caching.

        Args:
            x (numpy.ndarray): The vector to be multiplied with the matrix A.

        Returns:
            numpy.ndarray: The product of matrix A and vector x.
        """
        if not self.store_mat_vec_prod or safe_sparse_norm(x - self.x_last) != 0:
            z = self.A @ x
            if scipy.sparse.issparse(z):
                z = z.toarray().ravel()
            if self.store_mat_vec_prod:
                self.mat_vec_prod = z
                self.x_last = x.copy()

        return self.mat_vec_prod

    def norm(self, x):
        """
        Computes the norm of a vector x.

        Args:
            x (numpy.ndarray): The vector whose norm is to be computed.

        Returns:
            float: The norm of the vector.
        """
        return safe_sparse_norm(x)

    def smoothness(self):
        """
        Computes the smoothness constant of the logistic regression function.

        Returns:
            float: The smoothness constant.
        """
        if scipy.sparse.issparse(self.A):
            sing_val_max = scipy.sparse.linalg.svds(self.A, k=1, return_singular_vectors=False)[0]
            return 0.25 * sing_val_max**2 / self.n + self.l2
        else:
            covariance = self.A.T @ self.A / self.n
            return 0.25 * np.max(la.eigvalsh(covariance)) + self.l2

    def max_smoothness(self):
        """
        Computes the maximum smoothness constant across all data points.

        Returns:
            float: The maximum smoothness constant.
        """
        max_squared_sum = row_norms(self.A, squared=True).max()
        return 0.25 * max_squared_sum + self.l2

    def average_smoothness(self):
        """
        Computes the average smoothness constant across all data points.

        Returns:
            float: The average smoothness constant.
        """
        ave_squared_sum = row_norms(self.A, squared=True).mean()
        return 0.25 * ave_squared_sum + self.l2

    def batch_smoothness(self, batch_size):
        """
        Computes the smoothness constant of stochastic gradients sampled in minibatches.

        Args:
            batch_size (int): The size of the minibatch.

        Returns:
            float: The smoothness constant for the given batch size.
        """
        L = self.smoothness()
        L_max = self.max_smoothness()
        L_batch = self.n / (self.n - 1) * (1 - 1 / batch_size) * L + (self.n / batch_size - 1) / (self.n - 1) * L_max
        return L_batch

    def density(self, x):
        """
        Computes the density of a sparse matrix or vector x.

        Args:
            x (numpy.ndarray or scipy.sparse matrix): The matrix or vector whose density is to be computed.

        Returns:
            float: The density of the matrix or vector.
        """
        if hasattr(x, "toarray"):
            dty = float(x.nnz) / (x.shape[0] * x.shape[1])
        else:
            dty = 0 if x is None else float((x != 0).sum()) / x.size
        return dty

    def stochastic_gradient(self, x, idx=None, batch_size=1, replace=False, normalization=None):
        """
        Computes the stochastic gradient of the logistic regression loss function for a subset or random batch.

        Args:
            x (numpy.ndarray): The point at which to compute the gradient.
            idx (array-like, optional): Indices for the subset to be used. Defaults to None.
            batch_size (int, optional): The size of the random batch if idx is None. Defaults to 1.
            replace (bool, optional): Whether to sample with replacement. Defaults to False.
            normalization (float, optional): Normalization factor for the gradient. Defaults to None.

        Returns:
            numpy.ndarray: The stochastic gradient of the loss function.
        """
        if idx is None:
            idx = np.random.choice(self.n, size=batch_size, replace=replace)
        else:
            batch_size = 1 if np.isscalar(idx) else len(idx)
        if normalization is None:
            normalization = batch_size
        z = self.A[idx] @ x
        if scipy.sparse.issparse(z):
            z = z.toarray().ravel()
        activation = scipy.special.expit(z)
        stoch_grad = safe_sparse_add(self.A[idx].T @ (activation - self.b[idx]) / normalization, self.l2 * x)
        if scipy.sparse.issparse(x):
            stoch_grad = scipy.sparse.csr_matrix(stoch_grad).T
        return stoch_grad

    def hessian(self, x):
        """
        Computes the Hessian of the logistic regression loss function.

        Args:
            x (numpy.ndarray): The point at which to compute the Hessian.

        Returns:
            numpy.ndarray: The Hessian matrix of the loss function.
        """
        z = self.mat_vec_product(x)
        activation = scipy.special.expit(z)
        weights = activation * (1 - activation)
        A_weighted = safe_sparse_multiply(self.A.T, weights)
        return A_weighted @ self.A / self.n + self.l2 * np.eye(self.dim)

    def stochastic_hessian(self, x, idx=None, batch_size=1, replace=False, normalization=None):
        """
        Computes the stochastic Hessian of the logistic regression loss function for a subset or random batch.

        Args:
            x (numpy.ndarray): The point at which to compute the stochastic Hessian.
            idx (array-like, optional): Indices for the subset to be used. Defaults to None.
            batch_size (int, optional): The size of the random batch if idx is None. Defaults to 1.
            replace (bool, optional): Whether to sample with replacement. Defaults to False.
            normalization (float, optional): Normalization factor for the Hessian. Defaults to None.

        Returns:
            numpy.ndarray: The stochastic Hessian matrix of the loss function.
        """
        if idx is None:
            idx = np.random.choice(self.n, size=batch_size, replace=replace)
        else:
            batch_size = 1 if np.isscalar(idx) else len(idx)
        if normalization is None:
            normalization = batch_size
        z = self.A[idx] @ x
        if scipy.sparse.issparse(z):
            z = z.toarray().ravel()
        activation = scipy.special.expit(z)
        weights = activation * (1 - activation)
        A_weighted = safe_sparse_multiply(self.A[idx].T, weights)
        return A_weighted @ self.A[idx] / normalization + self.l2 * np.eye(self.dim)

    def mat_vec_product(self, x):
        """
        Computes the matrix-vector product of A and x, with optional caching.

        Args:
            x (numpy.ndarray): The vector to be multiplied with the matrix A.

        Returns:
            numpy.ndarray: The product of matrix A and vector x.
        """
        if not self.store_mat_vec_prod or safe_sparse_norm(x - self.x_last) != 0:
            z = self.A @ x
            if scipy.sparse.issparse(z):
                z = z.toarray().ravel()
            if self.store_mat_vec_prod:
                self.mat_vec_prod = z
                self.x_last = x.copy()

        return self.mat_vec_prod

    def norm(self, x):
        """
        Computes the norm of a vector x.

        Args:
            x (numpy.ndarray): The vector whose norm is to be computed.

        Returns:
            float: The norm of the vector.
        """
        return safe_sparse_norm(x)

    def smoothness(self):
        """
        Computes the smoothness constant of the logistic regression function.

        Returns:
            float: The smoothness constant.
        """
        if scipy.sparse.issparse(self.A):
            sing_val_max = scipy.sparse.linalg.svds(self.A, k=1, return_singular_vectors=False)[0]
            return 0.25 * sing_val_max**2 / self.n + self.l2
        else:
            covariance = self.A.T @ self.A / self.n
            return 0.25 * np.max(la.eigvalsh(covariance)) + self.l2

    def max_smoothness(self):
        """
        Computes the maximum smoothness constant across all data points.

        Returns:
            float: The maximum smoothness constant.
        """
        max_squared_sum = row_norms(self.A, squared=True).max()
        return 0.25 * max_squared_sum + self.l2

    def average_smoothness(self):
        """
        Computes the average smoothness constant across all data points.

        Returns:
            float: The average smoothness constant.
        """
        ave_squared_sum = row_norms(self.A, squared=True).mean()
        return 0.25 * ave_squared_sum + self.l2

    def batch_smoothness(self, batch_size):
        """
        Computes the smoothness constant of stochastic gradients sampled in minibatches.

        Args:
            batch_size (int): The size of the minibatch.

        Returns:
            float: The smoothness constant for the given batch size.
        """
        L = self.smoothness()
        L_max = self.max_smoothness()
        L_batch = self.n / (self.n - 1) * (1 - 1 / batch_size) * L + (self.n / batch_size - 1) / (self.n - 1) * L_max
        return L_batch

    def density(self, x):
        """
        Computes the density of a sparse matrix or vector x.

        Args:
            x (numpy.ndarray or scipy.sparse matrix): The matrix or vector whose density is to be computed.

        Returns:
            float: The density of the matrix or vector.
        """
        if hasattr(x, "toarray"):
            dty = float(x.nnz) / (x.shape[0] * x.shape[1])
        else:
            dty = 0 if x is None else float((x != 0).sum()) / x.size
        return dty
