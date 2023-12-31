import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from optimizers.IARVR import ProxFinito


def train_model(
    dataloader: DataLoader, model, loss_fn, optimizer: Optimizer, device="cpu"
):
    device = torch.device(device)
    model.to(device)
    bar = tqdm(total=len(dataloader))
    loss_data = []
    model.train()
    
    is_finito = isinstance(optimizer, ProxFinito)

    for i, (data, label) in enumerate(dataloader):
        optimizer.zero_grad()
        data = data.flatten(start_dim=1).to(device)
        label = label.to(device)

        output = model(data)
        loss = loss_fn(output, label)
        loss.backward()
        if is_finito:
            optimizer.step(i)
        else:
            optimizer.step()

        loss_data.append(loss.detach().cpu())

        bar.update(1)
        bar.set_postfix({"loss": loss.detach().cpu().numpy()})
    return loss_data



def test_model(dataloader: DataLoader, model, loss_fn, device="cpu"):
    bar = tqdm(total=len(dataloader))
    loss_data = []
    device = torch.device(device)
    model.to(device)
    model.eval()

    tp = 0
    total = 0
    with torch.no_grad():
        for i, (data, label) in enumerate(dataloader):
            data = data.flatten(start_dim=1).to(device)
            label = label.to(device)

            output = model(data)
            loss = loss_fn(output, label)

            pred = output.argmax(dim=1)
            tp += (pred == label).sum()
            total += label.shape[0]

            loss_data.append(loss.detach().cpu())
            bar.update(1)
            bar.set_postfix({"loss": loss.detach().cpu().numpy(), "acc": {tp / total}})

    return loss_data, tp / total