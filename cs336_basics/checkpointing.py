import torch
import os
from typing import BinaryIO, IO

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    if isinstance(out, (str, os.PathLike)):
        torch.save(checkpoint, out)
    else:
        torch.save(checkpoint, out)


def load_checkpoint(
    checkpoint: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a checkpoint file, load the model and optimizer state dicts, and return the iteration number.

    Args:
        checkpoint (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to load the checkpoint from.
        model (torch.nn.Module): Load the model state dict into this model.
        optimizer (torch.optim.Optimizer): Load the optimizer state dict into this optimizer.

    Returns:
        int: The iteration number that was loaded from the checkpoint.
    """
    if isinstance(checkpoint, (str, os.PathLike)):
        checkpoint_data = torch.load(checkpoint)
    else:
        checkpoint_data = torch.load(checkpoint)

    model.load_state_dict(checkpoint_data["model_state_dict"])
    optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
    iteration = checkpoint_data["iteration"]

    return iteration