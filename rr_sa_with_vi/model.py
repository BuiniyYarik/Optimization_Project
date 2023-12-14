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
