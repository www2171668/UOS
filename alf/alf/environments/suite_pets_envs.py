try:
    import mujoco_py
except ImportError as e:
    mujoco_py = None

from alf.environments.pets_envs.pusher import PusherEnv
from alf.environments.pets_envs.reacher import Reacher3DEnv

import functools
import numpy as np
import gym

import gin
from alf.environments import suite_gym, alf_wrappers, process_environment
from alf.environments.utils import UnwrappedEnvChecker

_unwrapped_env_checker_ = UnwrappedEnvChecker()

def is_available():
    return mujoco_py is not None

@gin.configurable
class SuccessWrapper(gym.Wrapper):
    """Retrieve the success info from the environment return."""

    def __init__(self, env, since_episode_steps):
        super().__init__(env)
        self._since_episode_steps = since_episode_steps

    def reset(self, **kwargs):
        self._steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._steps += 1

        info["success"] = 0.0   # * 记录成功次数
        # only count success after certain steps
        if self._steps >= self._since_episode_steps and info["is_success"] == 1:
            info["success"] = 1.0

        info.pop("is_success")  # from gym, we remove it here
        return obs, reward, done, info

@gin.configurable
class PETSSuccessWrapper(SuccessWrapper):
    def __init__(self, env, since_episode_steps):
        """如果不想改参数，下面可以不用写"""
        super().__init__(env=env, since_episode_steps=since_episode_steps)
        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            # 'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

class ActionScalingWrapper(gym.Wrapper):
    """scale the action space to (-1, 1) for better entropy calc"""

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        high = self.action_space.high
        low = self.action_space.low
        assert np.all(np.abs(low) == high)
        self._magnitude = high
        self.action_space.high = np.ones_like(high)
        self.action_space.low = np.ones_like(high) * -1

    def step(self, action):
        """scale action back to its original magnitude"""
        action *= self._magnitude
        return self.env.step(action)

@gin.configurable
def load(env_name,
         env_id=None,
         discount=1.0,
         max_episode_steps=None,
         use_success_wrapper=True,
         gym_env_wrappers=(),
         alf_env_wrappers=(),
         wrap_with_process=False):
    """Loads the selected environment and wraps it with the specified wrappers.

    Note that by default a ``TimeLimit`` wrapper is used to limit episode lengths
    to the default benchmarks defined by the registered environments.

    Args:
        env_name: Ignored, but required for create_environment in utils.py
        discount: Discount to use for the environment.
        max_episode_steps: If None the ``max_episode_steps`` will be set to the default
            step limit defined in the environment's spec. No limit is applied if set
            to 0 or if there is no ``timestep_limit`` set in the environment's spec.
        gym_env_wrappers: Iterable with references to wrapper classes to use
            directly on the gym environment.
        alf_env_wrappers: Iterable with references to wrapper classes to use on
            the torch environment.

    Returns:
        An AlfEnvironment instance.
    """
    _unwrapped_env_checker_.check_and_update(wrap_with_process)

    # % 加载环境
    if env_name == "Pusher":
        env = PusherEnv()
    elif env_name == "Reacher":
        env = Reacher3DEnv()
    env = ActionScalingWrapper(env)
    if use_success_wrapper:
        env = SuccessWrapper(env, max_episode_steps)

    def env_ctor(env_id=None):
        return suite_gym.wrap_env(
            env,
            env_id=env_id,
            discount=discount,
            max_episode_steps=max_episode_steps,
            gym_env_wrappers=gym_env_wrappers,
            alf_env_wrappers=alf_env_wrappers)

    if wrap_with_process:
        process_env = process_environment.ProcessEnvironment(functools.partial(env_ctor))
        process_env.start()
        torch_env = alf_wrappers.AlfEnvironmentBaseWrapper(process_env)
    else:  # 当前使用
        torch_env = env_ctor(env_id=env_id)

    return torch_env
