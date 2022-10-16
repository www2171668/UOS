import time
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math

from olympics.core import OlympicsBase
from olympics.viewer import Viewer

class Running_Off(OlympicsBase):
    def __init__(self, map, map_name, seed=None, max_episode_steps=1200, expand_state=False, sparse_reward=False):
        self.expand_state = expand_state

        for object in map["objects"]:
            if object.color == "red":  # 记录最后一个red
                self.endpoint = [(object.l1[0] + object.l2[0]) / 2, (object.l1[1] + object.l2[1]) / 2]
        super(Running_Off, self).__init__(map, seed)

        self.restitution = 0.5
        self.print_log = False
        self.print_log2 = False
        self.tau = 0.1

        self.speed_cap = 100

        self.draw_obs = False
        self.show_traj = True

        self.end_state=False

        observation_shape = 4 if expand_state else 3
        observation_shape = observation_shape + 2 if self.end_state else observation_shape
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(observation_shape,), dtype='float32')
        self.action_space = spaces.Box(low=np.array([-100, -30]), high=np.array([200, 30]), dtype='float32')

        self._max_episode_steps = max_episode_steps
        self.max_step = self._max_episode_steps

        self.sparse_reward = sparse_reward
        self.distance_threshold = 0.2
        self.reward_range = (-10, 101)
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': 30
        }
        self.info = {'is_success': False}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_reward(self):
        agent_reward = -0.1
        if self.agent_list[0].finished:  # 完成奖励为100
            agent_reward = 100.

        return agent_reward

    def get_intensive_reward(self, pos):
        """This function is used for implement of the intensive reward."""
        np_pos = np.array(pos[:2])
        np_endpoint = np.array(self.endpoint)
        agent_reward = (-np.sum(np.square(np_pos - np_endpoint)) ** 0.5) / 100

        return agent_reward

    def is_terminal(self, step_reward):
        done = False
        info = {'is_success': False}

        if self.step_cnt >= self.max_step:
            done = True
        elif self.sparse_reward and self.agent_list[0].finished:  # 成功,停止
            done = True
            info = {'is_success': True}
        elif not self.sparse_reward and step_reward > -self.distance_threshold:  # 成功
            done = True
            info = {'is_success': True}

        return done, info

    def reset(self):
        self.set_seed()
        self.init_state()
        self.step_cnt = 0
        self.done = False

        self.viewer = Viewer(self.view_setting)
        self.display_mode = False
        self.get_obs()
        return self.get_obs_myself()

    def get_obs_myself(self):
        """[x, y, theta]"""
        obs_myself = []
        for index in range(self.agent_num):
            agent_x, agent_y = self.agent_pos[index][0], self.agent_pos[index][1]
            theta = self.agent_theta[index][0]

            if self.expand_state:
                visibility = self.agent_list[index].visibility
                obs_myself.append([agent_x, agent_y, theta, visibility])
            else:
                obs_myself.append([agent_x, agent_y, theta])

            if self.end_state:
                obs_myself[0].extend(self.endpoint)
        return np.array(obs_myself[0])

    def step(self, actions_list):
        previous_pos = self.agent_pos
        self.stepPhysics([actions_list], self.step_cnt)  # env.step(action)
        self.speed_limit()
        self.cross_detect(previous_pos, self.agent_pos)  # 通过终点判定
        obs_myself = self.get_obs_myself()  # 获得状态
        self.change_inner_state()

        self.step_cnt += 1
        if self.sparse_reward:
            step_reward = self.get_reward()
        else:
            step_reward = self.get_intensive_reward(obs_myself)

        done, info = self.is_terminal(step_reward)  # done由步数和成功一起控制

        return obs_myself, step_reward, done, info

    def close(self):
        pass
