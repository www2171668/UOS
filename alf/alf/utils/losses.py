
"""Various function/classes related to loss computation."""

import torch
import torch.nn.functional as F

import alf


@alf.configurable
def element_wise_huber_loss(x, y):
    """Elementwise Huber loss.

    Args:
        x (Tensor): label
        y (Tensor): prediction
    Returns:
        loss (Tensor)
    """
    return F.smooth_l1_loss(y, x, reduction="none")


@alf.configurable
def element_wise_squared_loss(x, y):
    """Elementwise squared loss.

    Args:
        x (Tensor): label
        y (Tensor): prediction
    Returns:
        loss (Tensor)
    """
    return F.mse_loss(y, x, reduction="none")
