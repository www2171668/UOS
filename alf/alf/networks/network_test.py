
"""Tests for alf.networks.network."""

from absl.testing import parameterized

import torch
import torch.nn as nn

import alf
from alf.tensor_specs import TensorSpec
from alf.initializers import _numerical_calculate_gain
from alf.initializers import _calculate_gain
from alf.networks import EncodingNetwork, LSTMEncodingNetwork
from alf.networks.network import NaiveParallelNetwork


def test_net_copy(net):
    """Test whether net.copy() is correctly implemented"""
    new_net = net.copy()
    params = dict(net.named_parameters())
    new_params = dict(new_net.named_parameters())
    for n, p in new_params.items():
        assert p.shape == params[n].shape, (
            "The shape of the parameter of the "
            "copied network is different from that of the original network: "
            " %s vs %s" % (p.shape, params[n].shape))
        assert id(p) != id(
            params[n]), ("The parameter of the copied parameter "
                         "is the same parameter of the original network")


class BaseNetwork(alf.networks.Network):
    def __init__(self, v1, **kwargs):
        super().__init__(v1, **kwargs)


class MockNetwork(BaseNetwork):
    def __init__(self, param1, param2, kwarg1=2, kwarg2=3):
        self.param1 = param1
        self.param2 = param2
        self.kwarg1 = kwarg1
        self.kwarg2 = kwarg2

        super().__init__(param1, name='mock')

        self.var1 = nn.Parameter(torch.tensor(1., requires_grad=False))
        self.var2 = nn.Parameter(torch.tensor(2., requires_grad=True))

    def forward(self, observations, network_state=None):
        return self.var1 + self.var2


class NoInitNetwork(MockNetwork):
    pass


class NetworkTest(alf.test.TestCase):
    def test_copy_works(self):
        # pass a TensorSpec to prevent assertion error in Network
        network1 = MockNetwork(TensorSpec([2]), 1)
        network2 = network1.copy()

        self.assertNotEqual(network1, network2)
        self.assertEqual(TensorSpec([2]), network2.param1)
        self.assertEqual(1, network2.param2)
        self.assertEqual(2, network2.kwarg1)
        self.assertEqual(3, network2.kwarg2)

    def test_noinit_copy_works(self):
        # pass a TensorSpec to prevent assertion error in Network
        network1 = NoInitNetwork(TensorSpec([2]), 1)
        network2 = network1.copy()

        self.assertNotEqual(network1, network2)
        self.assertEqual(TensorSpec([2]), network2.param1)
        self.assertEqual(1, network2.param2)
        self.assertEqual(2, network2.kwarg1)
        self.assertEqual(3, network2.kwarg2)

    def test_too_many_args_raises_appropriate_error(self):
        self.assertRaises(TypeError, MockNetwork, 0, 1, 2, 3, 4, 5, 6)


class InitializerTest(parameterized.TestCase, alf.test.TestCase):
    @parameterized.parameters((torch.relu), (alf.utils.math_ops.identity, ),
                              (torch.tanh, ), (torch.sigmoid, ),
                              (torch.nn.functional.elu, ),
                              (torch.nn.functional.leaky_relu, ))
    def test_numerical_calculate_gain(self, activation):
        numerical_gain = _numerical_calculate_gain(activation)
        if activation.__name__ == "identity":
            gain = _calculate_gain("linear")
        else:
            gain = _calculate_gain(activation.__name__)
        print(activation.__name__, numerical_gain, gain)
        self.assertLess(abs(numerical_gain - gain), 0.1)


class NaiveParallelNetworkTest(alf.test.TestCase):
    def test_non_rnn(self):
        input_spec = TensorSpec((100, ), torch.float32)
        embedding = input_spec.zeros(outer_dims=(6, ))

        network = EncodingNetwork(
            input_tensor_spec=input_spec,
            fc_layer_params=(30, 40, 50),
            activation=torch.tanh)
        replicas = 4
        num_layers = 3

        pnet = NaiveParallelNetwork(network, replicas)

        self.assertEqual(
            len(list(pnet.parameters())), num_layers * 2 * replicas)

        output, _ = pnet(embedding)
        self.assertEqual(output.shape, (6, replicas, 50))
        self.assertEqual(pnet.output_spec.shape, (replicas, 50))

    def test_rnn(self):
        input_spec = TensorSpec((100, ), torch.float32)
        embedding = input_spec.zeros(outer_dims=(6, ))

        network = LSTMEncodingNetwork(
            input_tensor_spec=input_spec, hidden_size=(30, 40))
        replicas = 4
        pnet = NaiveParallelNetwork(network, replicas)

        self.assertEqual(pnet.state_spec,
                         [(TensorSpec((4, 30)), TensorSpec((4, 30))),
                          (TensorSpec((4, 40)), TensorSpec((4, 40)))])
        state = alf.utils.common.zero_tensor_from_nested_spec(
            pnet.state_spec, 6)
        output, state = pnet(embedding, state)
        self.assertEqual(output.shape, (6, replicas, 40))
        self.assertEqual(pnet.output_spec.shape, (replicas, 40))
        self.assertEqual(
            alf.utils.dist_utils.extract_spec(state),
            [(TensorSpec((4, 30)), TensorSpec((4, 30))),
             (TensorSpec((4, 40)), TensorSpec((4, 40)))])


if __name__ == '__main__':
    alf.test.main()
