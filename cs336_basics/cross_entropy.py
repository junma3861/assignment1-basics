import torch
import torch.nn as nn

from torch import Tensor


def cross_entropy_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    # Compute the cross-entropy loss for each example
    # and then average across the batch.
    # You may find the following functions useful:
    # - torch.logsumexp
    # - torch.gather
    # - torch.mean

    # Subtract the largest element for numerical stability
    max_logits = torch.max(logits, dim=1, keepdim=True).values
    shifted_logits = logits - max_logits

    # Compute log-sum-exp for each example with shifted logits
    log_sum_exp = max_logits.squeeze(1) + torch.log(torch.sum(torch.exp(shifted_logits), dim=1))

    # Gather the logits corresponding to the correct classes
    correct_class_logits = torch.gather(logits, 1, targets.unsqueeze(1)).squeeze(1)

    # Compute the cross-entropy loss for each example
    loss = log_sum_exp - correct_class_logits

    # Average the loss across the batch
    average_loss = torch.mean(loss)

    return average_loss