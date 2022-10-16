

from functools import partial
import alf
from alf.algorithms.actor_critic_algorithm import ActorCriticAlgorithm

# configure which RL algorithm to use
alf.config(
    'TrainerConfig',
    algorithm_ctor=partial(
        ActorCriticAlgorithm, optimizer=alf.optimizers.Adam(lr=1e-3)),
    num_iterations=1)
