

from absl.testing import parameterized
import functools

import torch

import alf
import alf.networks.actor_networks as actor_network
from alf.tensor_specs import TensorSpec, BoundedTensorSpec


class ActorNetworkTest(alf.test.TestCase, parameterized.TestCase):
    def _init(self, lstm_hidden_size):
        if lstm_hidden_size is not None:
            actor_fc_layer_params = (6, 4)
            network_ctor = functools.partial(
                actor_network.ActorRNNNetwork,
                lstm_hidden_size=lstm_hidden_size,
                actor_fc_layer_params=actor_fc_layer_params)
            if isinstance(lstm_hidden_size, int):
                lstm_hidden_size = [lstm_hidden_size]
            state = [()]
            for size in lstm_hidden_size:
                state.append((torch.randn((
                    1,
                    size,
                ), dtype=torch.float32), ) * 2)
            state.append(())
        else:
            network_ctor = actor_network.ActorNetwork
            state = ()
        return network_ctor, state

    @parameterized.parameters((100, ), (None, ), ((200, 100), ))
    def test_actor_networks(self, lstm_hidden_size):
        obs_spec = TensorSpec((3, 20, 20), torch.float32)
        action_spec = BoundedTensorSpec((5, ), torch.float32, 2., 3.)
        conv_layer_params = ((8, 3, 1), (16, 3, 2, 1))
        fc_layer_params = (10, 8)

        image = obs_spec.zeros(outer_dims=(1, ))

        network_ctor, state = self._init(lstm_hidden_size)

        actor_net = network_ctor(
            obs_spec,
            action_spec,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params)

        action, state = actor_net(image, state)

        # (batch_size, num_actions)
        self.assertEqual(action.shape, (1, 5))


if __name__ == '__main__':
    alf.test.main()
