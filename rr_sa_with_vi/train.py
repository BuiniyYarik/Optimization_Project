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
