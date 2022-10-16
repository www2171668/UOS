

from absl.testing import parameterized
from functools import partial
import pprint
import torch

import alf
import alf.data_structures as ds
from alf.algorithms.lagrangian_reward_weight_algorithm import LagrangianRewardWeightAlgorithm
from alf.utils import common, dist_utils
from alf.optimizers import Adam


class LagrangianRewardWeightAlgorithmTest(parameterized.TestCase,
                                          alf.test.TestCase):
    @parameterized.parameters((False, ), (True, ))
    def test_lagrangian_algorithm(self, reward_weight_normalization):
        batch_size = 2
        obs_dim = 3
        observation_spec = alf.TensorSpec([obs_dim])
        action_spec = alf.BoundedTensorSpec((1, ),
                                            minimum=0,
                                            maximum=1,
                                            dtype=torch.float32)
        reward_spec = alf.TensorSpec((4, ))
        time_step_spec = ds.time_step_spec(observation_spec, action_spec,
                                           reward_spec)

        alg = LagrangianRewardWeightAlgorithm(
            reward_spec,
            reward_thresholds=[None, 0.1, -0.1, None],
            reward_weight_normalization=reward_weight_normalization,
            optimizer=Adam(lr=1.),
            init_weights=1.)

        # rewards will be all zeros
        time_step = common.zero_tensor_from_nested_spec(
            time_step_spec, batch_size)
        time_step = time_step._replace(untransformed=time_step)
        alg_step = alg.rollout_step(time_step, state=())
        alg_step_spec = dist_utils.extract_spec(alg_step)

        experience = ds.make_experience(time_step, alg_step, state=())

        alg.after_train_iter(experience, experience.rollout_info)

        reward_weights = alg.reward_weights

        if not reward_weight_normalization:
            # No thresholds, no training
            self.assertEqual(float(reward_weights[0]), 1.)
            self.assertEqual(float(reward_weights[3]), 1.)
            # weight increased (0 < 0.1)
            self.assertGreater(float(reward_weights[1]), 1.)
            # weight decreased (0 > -0.1)
            self.assertGreater(1., float(reward_weights[2]))
        else:
            self.assertTensorClose(reward_weights.sum(), torch.tensor(1.))


if __name__ == '__main__':
    alf.test.main()
