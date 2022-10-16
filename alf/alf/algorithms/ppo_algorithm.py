
"""PPO algorithm."""

import torch

import alf
from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.algorithms.ppo_loss import PPOLoss
from alf.data_structures import namedtuple, TimeStep
from alf.utils import value_ops, tensor_utils

PPOInfo = namedtuple(
    "PPOInfo", [
        "step_type", "discount", "reward", "action", "rollout_log_prob",
        "rollout_action_distribution", "returns", "advantages",
        "action_distribution", "value", "reward_weights"
    ],
    default_value=())


@alf.configurable
class PPOAlgorithm(ActorCriticAlgorithm):
    """PPO Algorithm.
    Implement the simplified surrogate loss in equation (9) of "Proximal
    Policy Optimization Algorithms" https://arxiv.org/abs/1707.06347

    It works with ``ppo_loss.PPOLoss``. It should have same behavior as `baselines.ppo2`.
    """

    @property
    def on_policy(self):
        return False

    def train_step(self, inputs: TimeStep, state, rollout_info):
        alg_step = self._rollout_step(inputs, state)
        return alg_step._replace(
            info=rollout_info._replace(
                step_type=alg_step.info.step_type,
                reward=alg_step.info.reward,
                discount=alg_step.info.discount,
                action_distribution=alg_step.info.action_distribution,
                value=alg_step.info.value,
                reward_weights=alg_step.info.reward_weights))

    def preprocess_experience(self, root_inputs: TimeStep, rollout_info, batch_info):
        """Compute advantages and put it into exp.rollout_info."""

        if rollout_info.reward.ndim == 3:  # [B, T, D] or [B, T, 1]
            discounts = rollout_info.discount.unsqueeze(-1) * self._loss.gamma
        else:  # [B, T]
            discounts = rollout_info.discount * self._loss.gamma

        advantages = value_ops.generalized_advantage_estimation(
            rewards=rollout_info.reward,
            values=rollout_info.value,
            step_types=rollout_info.step_type,
            discounts=discounts,
            td_lambda=self._loss._lambda,
            time_major=False)
        advantages = tensor_utils.tensor_extend_zero(advantages, dim=1)

        returns = rollout_info.value + advantages
        ppo_info = PPOInfo(
            rollout_action_distribution=rollout_info.action_distribution,
            rollout_log_prob=rollout_info.log_prob,
            returns=returns,
            action=rollout_info.action,
            advantages=advantages)
        return root_inputs, ppo_info
