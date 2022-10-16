
import alf
from alf.algorithms.agent import Agent
from alf.algorithms.sac_algorithm import SacAlgorithm

alf.config('Agent', rl_algorithm_cls=SacAlgorithm)

alf.config(
    'TrainerConfig',
    algorithm_ctor=Agent,
    whole_replay_buffer_training=False,
    clear_replay_buffer=False)
