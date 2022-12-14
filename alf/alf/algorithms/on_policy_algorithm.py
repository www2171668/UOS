
"""Base class for on-policy RL algorithms."""

from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm


class OnPolicyAlgorithm(OffPolicyAlgorithm):
    """OnPolicyAlgorithm implements the basic on-policy training procedure.

    User needs to implement ``rollout_step()`` and ``calc_loss()``.

    ``rollout_step()`` is called to generate actions for every environment step.
    It also needs to generate necessary information for training.

    ``update_with_gradient()`` is called every ``unroll_length`` steps (specified in
    ``config.TrainerConfig``). All the training information collected by every
    ``rollout_step()`` are batched and provided as arguments for
    ``calc_loss()``.

    The following is the pseudo code to illustrate how ``OnPolicyAlgorithm`` can
    be used:

    .. code-block:: python

        for _ in range(unroll_length):
            policy_step = rollout_step(time_step, policy_step.state)
            collect information from time_step into experience
            collect information from policy_step.info into train_info
            time_step = env.step(policy_step.output)
        loss = calc_loss(experience, train_info)
        update_with_gradient(loss)
    """

    @property
    def on_policy(self):
        return True

    # Implement train_step() to allow off-policy training for an OnPolicyAlgorithm
    def train_step(self, inputs, state, rollout_info):
        return self.rollout_step(inputs, state)
