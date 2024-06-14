import os
from typing import Optional, Callable

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from ..utils.tqdm import tqdm

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
        os.path.join(checkpoint_path, f"checkpoint_{current_epoch}.pt"),
    )


def _train(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.BCEWithLogitsLoss,
    optimizer: optim.AdamW,
    device: str,
    use_tqdm: bool = True,
    autocast: bool = False,
    autocast_dtype: Optional[torch.dtype] = torch.float16,
):
    # Setup model to training mode
    model.train()

    # Track the model as it progresses
    epoch_loss = 0
    epoch_steps = 0

    # Allow TQDM to be disabled
    it = tqdm(data_loader, total=len(data_loader)) if use_tqdm else data_loader

    # Iterate over the training batches
    for batch_data in it:
        # Update the parameters
        optimizer.zero_grad()

        # Send the params to the device
        ids = batch_data["ids"].to(device)
        mask = batch_data["mask"].to(device)
        targets = batch_data["targets"].to(device)

        if autocast:
            with torch.amp.autocast(device_type=device, dtype=autocast_dtype):
                # Run the model
                outputs = model(ids, mask)
                # Compute the results loss
                loss = loss_fn(outputs, targets)
        else:
            # Run the model
            outputs = model(ids, mask)
            # Compute the results loss
            loss = loss_fn(outputs, targets)


        loss.backward()
        optimizer.step()

        # Store Stats
        epoch_steps += 1
        epoch_loss += loss.item()

    return epoch_loss / epoch_steps


def train_epochs(
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
):
    for epoch in range(start_epoch, epoch_count):
        if log_progress:
            print(f"Running training epoch {epoch + 1}/{epoch_count}")
        avg_loss = _train(
            model, data_loader, loss_fn, optimizer, device, use_tqdm, autocast,
            autocast_dtype=autocast_dtype
        )
        if checkpoint_path:
            checkpoint_handler(epoch, avg_loss, checkpoint_path, model, optimizer)
        if log_progress:
            print(
                f"Finished training epoch {epoch + 1}/{epoch_count}; Average Loss: {avg_loss:.4f}"
            )


def validate(
    model: nn.Module, data_loader: DataLoader, 
    device: str, use_tqdm: bool = True,
    autocast: bool = False,
    autocast_dtype: Optional[torch.dtype] = torch.float16
):
    # Final Data
    actual_outputs = []
    target_outputs = []

    # Put the model in evaluation mode
    model.eval()

    # Allow TQDM to be disabled
    it = tqdm(data_loader, total=len(data_loader)) if use_tqdm else data_loader

    # Disable gradient calculations, as it's not needed for inference
    with torch.no_grad():
        # Iterate over the test batches
        for batch_data in it:
            # Send the params to the device
            ids = batch_data["ids"].to(device)
            mask = batch_data["mask"].to(device)

            # Get the model outputs
            if autocast:
                with torch.amp.autocast(device_type=device, 
                                        dtype=autocast_dtype):
                    cur_outputs = model(ids, mask)
            else:
                cur_outputs = model(ids, mask)

            # Load the target results
            processed_targets = batch_data["targets"].tolist()
            # Compute the actual results through a sigmoid (values will range from 0 to 1)
            processed_outputs = (
                torch.sigmoid(cur_outputs).cpu().detach().numpy().tolist()
            )

            # Update the result arrays
            actual_outputs.extend(processed_outputs)
            target_outputs.extend(processed_targets)

    # Return the evaluated data
    return actual_outputs, target_outputs
