
"""A simple parameterless action encoder."""

import numpy as np
import torch
import torch.nn.functional as F

import alf

from .network import Network


class SimpleActionEncoder(Network):
    """A simple encoder for action.

    It encodes discrete action to one hot representation and use the original
    continous actions. The output is the concat of all of them after flattening.
    """

    def __init__(self, action_spec):
        """

        Args:
            action_spec (nested BoundedTensorSpec): spec for actions
        """

        def check_supported_spec(spec):
            if spec.is_discrete:
                assert np.min(spec.minimum) == np.max(spec.minimum) == 0
                assert np.min(spec.maximum) == np.max(spec.maximum)

        alf.nest.map_structure(check_supported_spec, action_spec)
        self._action_spec = action_spec
        super().__init__(input_tensor_spec=action_spec, name="ActionEncoder")

    def forward(self, inputs, state=()):
        """Generate encoded actions.

        Args:
            inputs (nested Tensor): action tensors.
        Returns:
            nested Tensor with the same structure as inputs.

        """
        alf.nest.assert_same_structure(inputs, self._action_spec)
        actions = inputs
        outer_rank = alf.nest.utils.get_outer_rank(inputs, self._action_spec)

        def _encode_one_action(action, spec):
            if spec.is_discrete:
                num_actions = spec.maximum - spec.minimum + 1
                if num_actions.ndim == 0:
                    num_actions = int(num_actions)
                else:
                    num_actions = int(num_actins[0])
                a = F.one_hot(action, num_actions).to(torch.float32)
            else:
                a = action
            if outer_rank > 0:
                return a.reshape(*a.shape[:outer_rank], -1)
            else:
                return a.reshape(-1)

        actions = alf.nest.map_structure(_encode_one_action, actions,
                                         self._action_spec)

        return torch.cat(alf.nest.flatten(actions), dim=-1), ()
