
"""Test cases adpated from tf_agents' py_environment_test.py."""

import numpy as np
import torch

import alf
from alf.environments.random_alf_environment import RandomAlfEnvironment
import alf.nest as nest
from alf.tensor_specs import TensorSpec, BoundedTensorSpec


class AlfEnvironmentTest(alf.test.TestCase):
    def testResetSavesCurrentTimeStep(self):
        obs_spec = BoundedTensorSpec((1, ), torch.int32)
        action_spec = BoundedTensorSpec((1, ), torch.int64)

        random_env = RandomAlfEnvironment(
            observation_spec=obs_spec, action_spec=action_spec)

        time_step = random_env.reset()
        current_time_step = random_env.current_time_step()
        nest.map_structure(self.assertEqual, time_step, current_time_step)

    def testStepSavesCurrentTimeStep(self):
        obs_spec = BoundedTensorSpec((1, ), torch.int32)
        action_spec = BoundedTensorSpec((1, ), torch.int64)

        random_env = RandomAlfEnvironment(
            observation_spec=obs_spec, action_spec=action_spec)

        random_env.reset()
        time_step = random_env.step(action=torch.ones((1, )))
        current_time_step = random_env.current_time_step()
        nest.map_structure(self.assertEqual, time_step, current_time_step)


if __name__ == '__main__':
    alf.test.main()
