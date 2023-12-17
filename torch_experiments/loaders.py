import torch
import math
from torch.utils.data import Dataset


def standart_collate_fn(x, y):
    x = torch.stack(x)
    y = torch.tensor(y)
    return x, y


class CustomDataLoader:
    def __init__(
        self,
        dataset: Dataset,
        n_iterations: int = None,
        batch_size: int = 1,
        collate_fn=None,
    ) -> None:
        self.batch_size = batch_size
        self.n_batches = math.ceil(len(dataset) / batch_size)
        self.order = torch.arange(start=0, end=self.n_batches)
        self.dataset = dataset
        if collate_fn is None:
            self.collate_fn = standart_collate_fn
        else:
            self.collate_fn = collate_fn

        if n_iterations is None:
            self.n_iterations = self.n_batches
        else:
            self.n_iterations = n_iterations

    def __iter__(self):
        self.cnt = 0
        return self
    
    def __len__(self):
        return self.n_iterations

    def __next__(self):
        if self.cnt == self.n_iterations:
            raise StopIteration
        i = torch.randint(low=0, high=self.n_batches - 1, size=(1,))
        x = []
        y = []
        for j in range(
            i * self.batch_size, min((i + 1) * self.batch_size, len(self.dataset))
        ):
            c_x, c_y = self.dataset[j]

            x.append(c_x)
            y.append(c_y)

        self.cnt += 1
        
        return self.collate_fn(x, y)
