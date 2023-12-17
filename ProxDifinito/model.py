import torch

class LogisticRegression(torch.nn.Module):
    
    def __init__(self, n_inputs: int, n_outputs: int):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs)
    
    def forward(self, x: torch.tensor):
        out = self.linear.forward(x)
        return torch.sigmoid(out)
