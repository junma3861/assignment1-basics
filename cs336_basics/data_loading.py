import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    # Randomly select starting indices for each sequence in the batch
    start_indices = torch.randint(0, len(dataset) - context_length, (batch_size,))
    input_sequences = torch.stack([
        torch.tensor(dataset[i:i + context_length], device=device, dtype=torch.long)
        for i in start_indices
    ])
    labels = torch.stack([
        torch.tensor(dataset[i + 1:i + context_length + 1], device=device, dtype=torch.long)
        for i in start_indices
    ])
    return input_sequences, labels