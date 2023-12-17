import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Optimizer


def train_model(
    dataloader: DataLoader, model, loss_fn, optimizer: Optimizer, device="cpu", verbose=True
):
    device = torch.device(device)
    model.to(device)
    if verbose:
        bar = tqdm(total=len(dataloader))
    loss_data = []
    acc_data = []
    model.train()
    
    tp = 0
    total = 0
    for i, (data, label) in enumerate(dataloader):
        optimizer.zero_grad()
        data = data.to(device)
        # data = data.flatten(start_dim=1).to(device)
        label = label.to(device)

        output = model(data)
        
        pred = output.argmax(dim=1)
        tp += (pred == label).sum()
        total += label.shape[0]
        acc_data.append((tp/total).detach().cpu())
        
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

        loss_data.append(loss.detach().cpu())
        if verbose:
            bar.update(1)
            bar.set_postfix({"loss": loss.detach().cpu().numpy()})
    return loss_data, acc_data



def test_model(dataloader: DataLoader, model, loss_fn, device="cpu", verbose=True):
    if verbose:
        bar = tqdm(total=len(dataloader))
    loss_data = []
    acc_data = []
    device = torch.device(device)
    model.to(device)
    model.eval()

    tp = 0
    total = 0
    with torch.no_grad():
        for i, (data, label) in enumerate(dataloader):
            data = data.to(device)
            # data = data.flatten(start_dim=1).to(device)
            label = label.to(device)

            output = model(data)
            loss = loss_fn(output, label)

            pred = output.argmax(dim=1)
            tp += (pred == label).sum()
            total += label.shape[0]
            acc_data.append((tp/total).detach().cpu())

            loss_data.append(loss.detach().cpu())
            if verbose:
                bar.update(1)
                bar.set_postfix({"loss": loss.detach().cpu().numpy(), "acc": {acc_data[-1]}})

    return loss_data, tp / total, acc_data