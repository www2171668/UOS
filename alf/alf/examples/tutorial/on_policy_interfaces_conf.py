

from functools import partial

import torch

import alf
from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm
from alf.data_structures import namedtuple

MyACInfo = namedtuple("MyACInfo", ["ac", "zeros"])


class MyACAlgorithm(ActorCriticAlgorithm):
    def rollout_step(self, inputs, state):
        alg_step = super().rollout_step(inputs, state)
        action = alg_step.output
        zeros = torch.zeros_like(action)
        print("rollout_step: ", zeros.shape)
        alg_step = alg_step._replace(
            info=MyACInfo(ac=alg_step.info, zeros=zeros))
        return alg_step

    def calc_loss(self, info: MyACInfo):
        zeros = info.zeros
        print("calc_loss: ", zeros.shape)
        return super().calc_loss(info.ac)

    def after_update(self, root_inputs, info: MyACInfo):
        zeros = info.zeros
        print("after_update: ", zeros.shape)
        super().after_update(root_inputs, info.ac)

    def after_train_iter(self, root_inputs, rollout_info: MyACInfo):
        zeros = rollout_info.zeros
        print("after_train_iter: ", zeros.shape)
        super().after_train_iter(root_inputs, rollout_info.ac)


# configure which RL algorithm to use
alf.config(
    'TrainerConfig',
    algorithm_ctor=partial(
        MyACAlgorithm, optimizer=alf.optimizers.Adam(lr=1e-3)),
    num_iterations=1)
