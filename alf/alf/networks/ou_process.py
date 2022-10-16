
"""Ornstein-Uhlenbeck process."""
import torch

import alf
from .network import Network


class OUProcess(Network):
    """A zero-mean Ornstein-Uhlenbeck process."""

    def __init__(self, state_spec, damping=0.15, stddev=0.2):
        """A Class for generating noise from a zero-mean Ornstein-Uhlenbeck process.

        The Ornstein-Uhlenbeck process is a process that generates temporally
        correlated noise via a random walk with damping. This process describes
        the velocity of a particle undergoing brownian motion in the presence of
        friction. This can be useful for exploration in continuous action
        environments with momentum.

        The temporal update equation is:
        `x_next = (1 - damping) * x + N(0, std_dev)`

        Args:
            state_spec (nested TensorSpec): spec of the state
            damping (float): The rate at which the noise trajectory is damped towards the
                mean. We must have 0 <= damping <= 1, where a value of 0 gives an
                undamped random walk and a value of 1 gives uncorrelated Gaussian noise.
                Hence in most applications a small non-zero value is appropriate.
            stddev (float): Standard deviation of the Gaussian component.
        """
        super().__init__(input_tensor_spec=(), name="OUProcess")
        self._state_spec = state_spec
        self._1_sub_damping = 1 - damping
        self._stddev = stddev

    def forward(self, state):
        def _forward(x):
            return self._1_sub_damping * x + torch.randn_like(x) * self._stddev

        state = alf.nest.map_structure(_forward, state)
        return state, state

    @property
    def state_spec(self):
        return self._state_spec
