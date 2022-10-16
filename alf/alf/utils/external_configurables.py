
"""Make various external gin-configurable objects."""

import gin
import gin.torch
import gym
import torch

import alf

# This allows the environment creation arguments to be configurable by supplying
# gym.envs.registration.EnvSpec.make.ARG_NAME=VALUE
gym.envs.registration.EnvSpec.make = gin.external_configurable(
    gym.envs.registration.EnvSpec.make, 'gym.envs.registration.EnvSpec.make')

# Activation functions.
gin.external_configurable(torch.exp, 'torch.exp')
gin.external_configurable(torch.tanh, 'torch.tanh')
gin.external_configurable(torch.relu, 'torch.relu')
gin.external_configurable(torch.relu_, 'torch.relu_')
gin.external_configurable(torch.sigmoid, 'torch.sigmoid')
gin.external_configurable(torch.sigmoid_, 'torch.sigmoid_')
gin.external_configurable(torch.nn.functional.elu, 'torch.nn.functional.elu')
gin.external_configurable(torch.nn.functional.elu_, 'torch.nn.functional.elu_')
gin.external_configurable(torch.nn.functional.leaky_relu_,
                          'torch.nn.functional.leaky_relu_')
gin.external_configurable(alf.math.softsign, 'alf.math.softsign')
gin.external_configurable(alf.math.softsign_, 'alf.math.softsign_')

gin.external_configurable(torch.nn.LeakyReLU, 'torch.nn.LeakyReLU')

gin.external_configurable(torch.nn.MSELoss, 'torch.nn.MSELoss')
gin.external_configurable(torch.nn.BCELoss, 'torch.nn.BCELoss')
gin.external_configurable(torch.nn.CrossEntropyLoss,
                          'torch.nn.CrossEntropyLoss')
