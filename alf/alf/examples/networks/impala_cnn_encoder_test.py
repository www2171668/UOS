

import torch

import alf
from alf.tensor_specs import TensorSpec
from alf.examples.networks import impala_cnn_encoder


class TestImpalaCnnEncoder(alf.test.TestCase):
    def test_residual_cnn_block_shape(self):
        # Such a residual CNN block does not chnage the shape of the input
        input_tensor_spec = TensorSpec((3, 20, 20), torch.float32)
        block = impala_cnn_encoder._create_residual_cnn_block(
            input_tensor_spec)
        self.assertEqual(input_tensor_spec, block.output_spec)

    def test_create_downsampling_cnn_stack_shape(self):
        input_tensor_spec = TensorSpec((3, 64, 64), torch.float32)
        single_stack = impala_cnn_encoder._create_downsampling_cnn_stack(
            input_tensor_spec=input_tensor_spec,
            output_channels=9,
            num_residual_blocks=2)
        self.assertEqual((9, 32, 32), single_stack.output_spec.shape)

    def test_impala_cnn_encoder_shape(self):
        observation_spec = TensorSpec((3, 64, 64), torch.float32)
        encoder = impala_cnn_encoder.create(
            input_tensor_spec=observation_spec,
            cnn_channel_list=(16, 32, 32),
            num_blocks_per_stack=2,
            output_size=256)
        self.assertEqual((256, ), encoder.output_spec.shape)
