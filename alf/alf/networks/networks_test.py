

import torch

import alf

from alf.networks import TemporalPool
from alf.utils.common import zero_tensor_from_nested_spec


class NestworksTest(alf.test.TestCase):
    def test_temporal_pool_skip(self):
        batch_size = 2
        dim = 3

        x = torch.randn((batch_size, 20, dim))
        l = TemporalPool(dim, 5, 3)
        self.assertEqual(l.output_spec, alf.TensorSpec((5, dim)))
        state = alf.utils.common.zero_tensor_from_nested_spec(
            l.state_spec, batch_size)

        for i in range(0, 3):
            o, state = l(x[:, i, :], state)
            self.assertEqual(o[:, 4, :], x[:, 0, :])
            self.assertEqual(o[:, :4, :], torch.zeros((batch_size, 4, dim)))

        for i in range(3, 6):
            o, state = l(x[:, i, :], state)
            self.assertEqual(o[:, 3:, :], x[:, 0:6:3, :])
            self.assertEqual(o[:, :3, :], torch.zeros((batch_size, 3, dim)))

        for i in range(6, 9):
            o, state = l(x[:, i, :], state)
            self.assertEqual(o[:, 2:, :], x[:, 0:9:3, :])
            self.assertEqual(o[:, :2, :], torch.zeros((batch_size, 2, dim)))

        for i in range(9, 12):
            o, state = l(x[:, i, :], state)
            self.assertEqual(o[:, 1:, :], x[:, 0:12:3, :])
            self.assertEqual(o[:, :1, :], torch.zeros((batch_size, 1, dim)))

        for i in range(12, 15):
            o, state = l(x[:, i, :], state)
            self.assertEqual(o, x[:, 0:15:3, :])

        for i in range(15, 18):
            o, state = l(x[:, i, :], state)
            self.assertEqual(o, x[:, 3:18:3, :])

    def test_temporal_pool_max(self):
        batch_size = 2
        dim = 3

        x = torch.randn((batch_size, 20, dim))
        l = TemporalPool(dim, 5, 3, mode='max')
        self.assertEqual(l.output_spec, alf.TensorSpec((5, dim)))
        state = alf.utils.common.zero_tensor_from_nested_spec(
            l.state_spec, batch_size)

        for i in range(0, 2):
            o, state = l(x[:, i, :], state)
            self.assertEqual(o[:, :5, :], torch.zeros((batch_size, 5, dim)))

        for i in range(2, 5):
            o, state = l(x[:, i, :], state)
            self.assertEqual(o[:, 4, :], x[:, 0:3, :].max(dim=1)[0])
            self.assertEqual(o[:, :4, :], torch.zeros((batch_size, 4, dim)))

        for i in range(5, 8):
            o, state = l(x[:, i, :], state)
            self.assertEqual(
                o[:, 3:, :], x[:, 0:6, :].reshape(batch_size, 2, 3,
                                                  dim).max(dim=2)[0])
            self.assertEqual(o[:, :3, :], torch.zeros((batch_size, 3, dim)))

        for i in range(8, 11):
            o, state = l(x[:, i, :], state)
            self.assertEqual(
                o[:, 2:, :], x[:, 0:9, :].reshape(batch_size, 3, 3,
                                                  dim).max(dim=2)[0])
            self.assertEqual(o[:, :2:], torch.zeros((batch_size, 2, dim)))

        for i in range(11, 14):
            o, state = l(x[:, i, :], state)
            self.assertEqual(
                o[:, 1:, :], x[:, 0:12, :].reshape(batch_size, 4, 3,
                                                   dim).max(dim=2)[0])
            self.assertEqual(o[:, :1:], torch.zeros((batch_size, 1, dim)))

        for i in range(14, 17):
            o, state = l(x[:, i, :], state)
            self.assertEqual(
                o, x[:, 0:15, :].reshape(batch_size, 5, 3, dim).max(dim=2)[0])

        for i in range(17, 20):
            o, state = l(x[:, i, :], state)
            self.assertEqual(
                o, x[:, 3:18, :].reshape(batch_size, 5, 3, dim).max(dim=2)[0])

    def test_temporal_pool_avg(self):
        batch_size = 2
        dim = 3

        x = torch.randn((batch_size, 20, dim))
        l = TemporalPool(dim, 5, 3, mode='avg')
        self.assertEqual(l.output_spec, alf.TensorSpec((5, dim)))
        state = alf.utils.common.zero_tensor_from_nested_spec(
            l.state_spec, batch_size)

        for i in range(0, 2):
            o, state = l(x[:, i, :], state)
            self.assertEqual(o[:, :5, :], torch.zeros((batch_size, 5, dim)))

        for i in range(2, 5):
            o, state = l(x[:, i, :], state)
            self.assertTensorClose(o[:, 4, :], x[:, 0:3, :].mean(dim=1))
            self.assertEqual(o[:, :4, :], torch.zeros((batch_size, 4, dim)))

        for i in range(5, 8):
            o, state = l(x[:, i, :], state)
            self.assertTensorClose(
                o[:, 3:, :], x[:, 0:6, :].reshape(batch_size, 2, 3,
                                                  dim).mean(dim=2))
            self.assertEqual(o[:, :3, :], torch.zeros((batch_size, 3, dim)))

        for i in range(8, 11):
            o, state = l(x[:, i, :], state)
            self.assertTensorClose(
                o[:, 2:, :], x[:, 0:9, :].reshape(batch_size, 3, 3,
                                                  dim).mean(dim=2))
            self.assertEqual(o[:, :2:], torch.zeros((batch_size, 2, dim)))

        for i in range(11, 14):
            o, state = l(x[:, i, :], state)
            self.assertTensorClose(
                o[:, 1:, :], x[:, 0:12, :].reshape(batch_size, 4, 3,
                                                   dim).mean(dim=2))
            self.assertEqual(o[:, :1:], torch.zeros((batch_size, 1, dim)))

        for i in range(14, 17):
            o, state = l(x[:, i, :], state)
            self.assertTensorClose(
                o, x[:, 0:15, :].reshape(batch_size, 5, 3, dim).mean(dim=2))

        for i in range(17, 20):
            o, state = l(x[:, i, :], state)
            self.assertTensorClose(
                o, x[:, 3:18, :].reshape(batch_size, 5, 3, dim).mean(dim=2))

    def test_delay0(self):
        spec = alf.TensorSpec((2, 4))
        l = alf.nn.Delay(spec, 0)
        self.assertEqual(l.input_tensor_spec, spec)
        self.assertEqual(l.state_spec, ())
        x = spec.randn((4, ))
        out, state = l(x, ())
        self.assertTensorEqual(out, x)
        self.assertEqual(state, ())

    def test_delay1(self):
        spec = alf.TensorSpec((2, 4))
        l = alf.nn.Delay(spec)
        self.assertEqual(l.input_tensor_spec, spec)
        self.assertEqual(l.state_spec, spec)
        batch_size = 4
        x1 = spec.randn((batch_size, ))
        state = zero_tensor_from_nested_spec(l.state_spec, batch_size)
        out, state = l(x1, state)
        self.assertTensorEqual(out, spec.zeros((batch_size, )))
        self.assertTensorEqual(state, x1)
        x2 = spec.randn((batch_size, ))
        out, state = l(x2, state)
        self.assertTensorEqual(out, x1)
        self.assertTensorEqual(state, x2)

    def test_delay2(self):
        spec = alf.TensorSpec((2, 4))
        l = alf.nn.Delay(spec, delay=2)
        self.assertEqual(l.input_tensor_spec, spec)
        self.assertEqual(l.state_spec, (spec, spec))
        batch_size = 4

        x1 = spec.randn((batch_size, ))
        state = zero_tensor_from_nested_spec(l.state_spec, batch_size)
        out, state = l(x1, state)
        self.assertTensorEqual(out, spec.zeros((batch_size, )))
        self.assertEqual(type(state), tuple)
        self.assertEqual(len(state), 2)
        self.assertTensorEqual(state[0], spec.zeros((batch_size, )))
        self.assertTensorEqual(state[1], x1)

        x2 = spec.randn((batch_size, ))
        out, state = l(x2, state)
        self.assertTensorEqual(out, spec.zeros((batch_size, )))
        self.assertTensorEqual(state[0], x1)
        self.assertTensorEqual(state[1], x2)

        x3 = spec.randn((batch_size, ))
        out, state = l(x3, state)
        self.assertTensorEqual(out, x1)
        self.assertTensorEqual(state[0], x2)
        self.assertTensorEqual(state[1], x3)


if __name__ == '__main__':
    alf.test.main()
