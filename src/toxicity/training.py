import os
from typing import Optional, Callable

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from sklearn.metrics import (
    f1_score, fbeta_score, accuracy_score, recall_score, precision_score)

from .utils.tqdm import tqdm

def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: optim.AdamW,
    device: str,
):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Load the model state
    model.load_state_dict(checkpoint["model_state_dict"])
    # Load the optimizer state
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # Load the epoch and loss
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    # Send the model to the device
    model.to(device)

    # Return the model, optimizer, epoch, and loss
    return model, optimizer, epoch, loss

def checkpoint_handler(
    current_epoch: int,
    current_loss: float,
    checkpoint_path: str,
    model: nn.Module,
    optimizer: optim.AdamW,
):
    os.makedirs(checkpoint_path, exist_ok=True)
    torch.save(
        {
            "epoch": current_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": current_loss,
        },
        os.path.join(checkpoint_path, f"checkpoint.pt"),
    )

def train_epochs(
    trainer: Callable,
    epoch_count: int,
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.BCEWithLogitsLoss,
    optimizer: optim.AdamW,
    device: str,
    start_epoch: int = 0,
    use_tqdm: bool = True,
    log_progress=True,
    checkpoint_path: Optional[str] = None,
    autocast: bool = False,
    autocast_dtype: Optional[torch.dtype] = torch.float16,
    epoch_callback: Optional[Callable[[int, float], None]] = None,
):
    for epoch in range(start_epoch, epoch_count):
        if log_progress:
            print(f"Running training epoch {epoch + 1}/{epoch_count}")
        avg_loss = trainer(
            model, data_loader, loss_fn, optimizer, device, use_tqdm, autocast,
            autocast_dtype=autocast_dtype
        )
        if checkpoint_path:
            checkpoint_handler(epoch, avg_loss, checkpoint_path, model, optimizer)
        if epoch_callback:
            epoch_callback(epoch, avg_loss)
        if log_progress:
            print(
                f"Finished training epoch {epoch + 1}/{epoch_count}; Average Loss: {avg_loss:.4f}"
            )


def model_metrics(targets, results, print_metrics: bool = False):
    fixed_weighted_f1 = f1_score(targets, results, average='weighted')
    fixed_macro_f1 = f1_score(targets, results, average='macro')
    fixed_weighted_f2 = fbeta_score(targets, results, beta=2, average='weighted')
    fixed_macro_f2 = fbeta_score(targets, results, beta=2, average='macro')
    fixed_accuracy = accuracy_score(targets, results)
    fixed_recall = recall_score(targets, results, average='weighted')
    fixed_precision = precision_score(targets, results, average='weighted')

    metrics = {
        "weighted_f1": fixed_weighted_f1,
        "macro_f1": fixed_macro_f1,
        "weighted_f2": fixed_weighted_f2,
        "macro_f2": fixed_macro_f2,
        "accuracy": fixed_accuracy,
        "recall": fixed_recall,
        "precision": fixed_precision,
    }

    if print_metrics:
        print(f"Weighted F1: {fixed_weighted_f1}")
        print(f"Macro F1: {fixed_macro_f1}")
        print(f"Weighted F2: {fixed_weighted_f2}")
        print(f"Macro F2: {fixed_macro_f2}")
        print(f"Accuracy: {fixed_accuracy}")
        print(f"Recall: {fixed_recall}")
        print(f"Precision: {fixed_precision}")

    return metrics