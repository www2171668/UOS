""""""

from enum import Enum
import functools
import numpy as np
from typing import Callable

import torch
import torch.nn as nn
import torch.distributions as td

import alf
from alf.algorithms.config import TrainerConfig
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.sac_algorithm import _set_target_entropy
from alf.data_structures import LossInfo, namedtuple, TimeStep
from alf.data_structures import AlgStep, StepType
from alf.nest import nest
import alf.nest.utils as nest_utils
from alf.networks import ActorDistributionNetwork, CriticNetwork
from alf.networks.preprocessors import EmbeddingPreprocessor
from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.utils import common, dist_utils, losses, math_ops, tensor_utils
from alf.utils.conditional_ops import conditional_update
from alf.utils.summary_utils import safe_mean_hist_summary

def _discounted_return(rewards, values, is_lasts, discounts):
    """Computes discounted return for the first T-1 steps.

       return = next_reward + next_discount * next_value if is not the last step;
       otherwise will set return = current_discount * current_value.
    """
    assert values.shape[0] >= 2, ("The sequence length needs to be at least 2. Got {s}".format(
        s=values.shape[0]))

    is_lasts = is_lasts.to(dtype=torch.float32)
    is_lasts = common.expand_dims_as(is_lasts, values)
    discounts = common.expand_dims_as(discounts, values)

    rets = torch.zeros_like(values)  # yi
    rets[-1] = values[-1]
    acc_values = rets.clone()

    with torch.no_grad():
        for t in reversed(range(rewards.shape[0] - 1)):
            rets[t] = acc_values[t + 1] * discounts[t + 1] + rewards[t + 1]  # γ * target_Q + r
            acc_values[t] = is_lasts[t] * values[t] + (1 - is_lasts[t]) * rets[t]  # target_Q or yi

    rets = rets[:-1]
    return rets.detach()

def _one_step_discounted_return(rewards, values, is_lasts, discounts):
    """Computes discounted return for the first T-1 steps.

       return = next_reward + next_discount * next_value if is not the last step;
       otherwise will set return = current_discount * current_value.

    Args:
        rewards (Tensor): shape is ``[T,B]`` (or ``[T]``)
        values (Tensor): shape is ``[T,B]`` (or ``[T]``)
        is_lasts (Tensor): shape is ``[T,B]`` (or ``[T]``)
        discounts (Tensor): shape is ``[T,B]`` (or ``[T]``)
    Returns:
        Tensor: A tensor with shape ``[T-1,B]`` (or ``[T-1]``) representing the discounted returns.
    """
    assert values.shape[0] >= 2, ("The sequence length needs to be at least 2. Got {s}".format(
        s=values.shape[0]))

    is_lasts = is_lasts.to(dtype=torch.float32)
    is_lasts = common.expand_dims_as(is_lasts, values)
    discounts = common.expand_dims_as(discounts, values)

    discounted_values = discounts * values
    rets = (1 - is_lasts[:-1]) * (rewards[1:] + discounted_values[1:]) + \
           is_lasts[:-1] * discounted_values[:-1]
    return rets.detach()

@alf.configurable
class TAACTDLoss(nn.Module):
    r"""This TD loss implements the compare-through multi-step Q operator
    :math:`\mathcal{T}^{\pi^{\text{ta}}}` proposed in the TAAC paper. For a sampled
    trajectory, it compares the beta action :math:`\tilde{b}_n` sampled from the
    current policy with the historical rollout beta action :math:`b_n` step by step,
    and uses the minimum :math:`n` that has :math:`\tilde{b}_n\lor b_n=1` as the
    target step for boostrapping.
    """

    def __init__(self,
                 gamma=0.98,
                 td_error_loss_fn=losses.element_wise_squared_loss,
                 debug_summaries=False,
                 name="TAACTDLoss"):
        """
        Args:
            gamma (float|list[float]): A discount factor for future rewards. For
                multi-dim reward, this can also be a list of discounts, each
                discount applies to a reward dim.
            td_errors_loss_fn (Callable): A function for computing the TD errors
                loss. This function takes as input the target and the estimated
                Q values and returns the loss for each element of the batch.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this loss.
        """
        super().__init__()
        self._name = name
        self._gamma = torch.tensor(gamma)
        self._debug_summaries = debug_summaries
        self._td_error_loss_fn = td_error_loss_fn

    @property
    def gamma(self):
        """Return the :math:`\gamma` value for discounting future rewards.

        Returns:
            Tensor: a rank-0 or rank-1 (multi-dim reward) floating tensor.
        """
        return self._gamma.clone()

    def forward(self, info, value, target_value):
        r"""Calculate the TD loss. The first dimension of all the tensors is the
        time dimension and the second dimesion is the batch dimension [T,B,...].

        Args:
            info (TaacInfo): TaacInfo collected from train_step().
            value (torch.Tensor): the tensor for the value at each time step.
                The loss is between this and the calculated return.
            target_value (torch.Tensor): the tensor for the value at each time step.
                This is used to calculate return.
        Returns:
            LossInfo: TD loss with the ``extra`` field same as the loss.
        """
        discounts = info.discount * self._gamma  # [T, B]

        # % 1.TAAC 判断is_lasts
        train_b = info.b  # [1] -> True
        rollout_b = info.rollout_b
        # td return till the first action switching.
        b = (rollout_b | train_b).to(torch.bool)
        # b at step 0 doesn't affect the bootstrapping of any step
        b[0, :] = False

        # combine is_last and b
        is_lasts = (info.step_type == StepType.LAST)
        is_lasts |= b

        # % 2.base td_loss
        discounted_return = _discounted_return if len(is_lasts) > 2 else _one_step_discounted_return
        returns = discounted_return(rewards=info.reward,
                                     values=target_value,
                                     is_lasts=is_lasts,
                                     discounts=discounts)

        value = value[:-1]
        loss = self._td_error_loss_fn(returns.detach(), value)  # (yi-Q)^2
        loss = tensor_utils.tensor_extend_zero(loss)

        return LossInfo(loss=loss, extra=loss)