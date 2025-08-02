import torch
import torch.nn as nn


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    # revisit
    loss = nn.functional.cross_entropy(logits, target_batch)
    return loss
    

def calc_loss_loader(data_loader, model, device, n_val_batches=200):
    model.eval()

    val_loss = 0.0
    with torch.no_grad():
        for i, (X, y) in enumerate(data_loader):
            if i >= n_val_batches:
                break
            loss = calc_loss_batch(X, y, model, device)
            val_loss += loss.item()
    model.train()
    return val_loss / n_val_batches