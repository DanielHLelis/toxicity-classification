from typing import Optional

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from ..utils.tqdm import tqdm

def trainer(
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
        data = batch_data["data"].to(device)
        target = batch_data["target"].to(device)

        if autocast:
            with torch.amp.autocast(device_type=device, dtype=autocast_dtype):
                # Run the model
                outputs = model(data)
                # Compute the results loss
                loss = loss_fn(outputs, target)
        else:
            # Run the model
            outputs = model(data)
            # Compute the results loss
            loss = loss_fn(outputs, target)


        loss.backward()
        optimizer.step()

        # Store Stats
        epoch_steps += 1
        epoch_loss += loss.item()

    return epoch_loss / epoch_steps


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
            data = batch_data["data"].to(device)

            # Get the model outputs
            if autocast:
                with torch.amp.autocast(device_type=device, 
                                        dtype=autocast_dtype):
                    cur_outputs = model(data)
            else:
                cur_outputs = model(data)

            # Load the target results
            processed_targets = batch_data["target"].tolist()
            # Compute the actual results through a sigmoid (values will range from 0 to 1)
            processed_outputs = (
                torch.sigmoid(cur_outputs).to(torch.float32).cpu().detach().numpy().tolist()
            )

            # Update the result arrays
            actual_outputs.extend(processed_outputs)
            target_outputs.extend(processed_targets)

    # Return the evaluated data
    return actual_outputs, target_outputs