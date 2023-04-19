import torch
import numpy as np


def is_valid_probability_distribution(input, dim=-1):
    """
    Check if the input tensor along given dimention (dim) is a valid probability distribution
    i.e. all elements are non-negative and sum up to 1.

    Args:
    input: torch.Tensor or np.ndarray

    Returns:
    bool: True if input is a valid probability distribution, False otherwise.
    """
    if not isinstance(input, (torch.Tensor, np.ndarray)):
        raise TypeError("Input must be a torch.Tensor or np.ndarray.")

    if (input < 0).any():
        return False

    eps = 1e-6
    if (abs(input.sum(dim=dim) - 1) > eps).all():
        return False

    return True
