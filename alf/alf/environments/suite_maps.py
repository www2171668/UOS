from alf.environments.Oly.olympics.generator import create_scenario
from alf.environments.Oly.olympics.scenario.running_off import Running_Off
from alf.environments.Oly.olympics.scenario.running_on import Running_On

import time
import json
import functools
import numpy as np
import gym

import alf
from alf.environments import suite_gym, alf_wrappers, process_environment
from alf.environments.utils import UnwrappedEnvChecker

@alf.configurable
def load(environment_name,
         env_id=None,
         discount=1.0,
         max_episode_steps=None,
         expand_states=True,
         sparse_reward=True,
         running_off=True,
         gym_env_wrappers=(),
         alf_env_wrappers=(),
         wrap_with_process=False):
    """Loads the selected environment and wraps it with the specified wrappers.

    Note that by default a ``TimeLimit`` wrapper is used to limit episode lengths
    to the default benchmarks defined by the registered environments.

    Args:
        environment_name: Name for the environment to load.
        env_id: A scalar ``Tensor`` of the environment ID of the time step.
        discount: Discount to use for the environment.
        max_episode_steps: If None the ``max_episode_steps`` will be set to the default
            step limit defined in the environment's spec. No limit is applied if set
            to 0 or if there is no ``timestep_limit`` set in the environment's spec.
        sparse_reward (bool): If True, the game ends once the goal is achieved.
            Rewards will be added by 1, changed from -1/0 to 0/1.
        gym_env_wrappers: Iterable with references to wrapper classes to use
            directly on the gym environment.
        alf_env_wrappers: Iterable with references to wrapper classes to use on
            the torch environment.
    Returns:
        An AlfEnvironment instance.
    """
    if max_episode_steps is None:
        max_episode_steps = 200

    running_map = 'maps_off.json' if running_off else 'maps_on.json'

    Gamemap = create_scenario(environment_name, running_map)
    if running_off:
        env = Running_Off(Gamemap, environment_name, None, max_episode_steps, expand_states, sparse_reward)
    else:
        env = Running_On(Gamemap, environment_name, None, max_episode_steps, expand_states, sparse_reward)

    return suite_gym.wrap_env(
        env,
        env_id=env_id,
        discount=discount,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        alf_env_wrappers=alf_env_wrappers,
        image_channel_first=False)
