import math
import torch
import torch.nn as nn

def get_lr_cosine_schedule(it : int, max_learning_rate : float, min_learning_rate : float, warmup_iters : int, cosine_cycle_iters : int) -> float:
    """Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the total number of iterations for cosine annealing
            (from start of cosine phase to end)."""
    # (Warm-up) If t < T_w, then α_t = t/T_w * α_max
    if it < warmup_iters:
        lr = (it / warmup_iters) * max_learning_rate
    # (Cosine annealing) If T_w ≤ t ≤ T_c, then α_t = α_min + 1/2 * (1 + cos(π * (t - T_w) / (T_c - T_w))) * (α_max - α_min)
    elif it <= cosine_cycle_iters:
        t_cosine = it - warmup_iters
        lr = min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + math.cos(math.pi * t_cosine / (cosine_cycle_iters - warmup_iters)))
    # (Post-annealing) If t > T_c, then α_t = α_min
    else:
        lr = min_learning_rate
    
    return lr