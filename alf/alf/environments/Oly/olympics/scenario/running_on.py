import time
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math

from olympics.core import OlympicsBase
from olympics.viewer import Viewer

def isPointInsideSegment(point, line_point1, line_point2):
    # 求Cos∠PP1P2
    x0, y0 = point
    x1, y1 = line_point1
    x2, y2 = line_point2
    dx10 = x0 - x1;
    dy10 = y0 - y1;
    m10 = math.hypot(dx10, dy10);
    dx12 = x2 - x1;
    dy12 = y2 - y1;
    m12 = math.hypot(dx12, dy12);
    if ((dx10 * dx12 + dy10 * dy12) / m10 / m12 < 0): return False

    # 求Cos∠PP2P1
    dx20 = x0 - x2;
    dy20 = y0 - y2;
    m20 = math.hypot(dx20, dy20);
    dx21 = x1 - x2;
    dy21 = y1 - y2;
    m21 = math.hypot(dx21, dy21);
    return (dx20 * dx21 + dy20 * dy21) / m20 / m21 >= 0

def point_distance_line(point, line_point1, line_point2):
    in_line = isPointInsideSegment(point, line_point1, line_point2)
    if in_line:  # 点在线段内
        vec1 = line_point1 - point
        vec2 = line_point2 - point
        distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1 - line_point2)
    else:
        distance = 100

    return distance

class Running_On(OlympicsBase):
    def __init__(self, map, map_name, seed=None, max_episode_steps=200,
                 expand_state=False, sparse_reward=True, gamma=0.98, tau=0.1):
        self.expand_state = expand_state

        for object in map["objects"]:
            if object.color == "dark":  # 记录最后一个black
                self.vec1 = np.array(object.l1)
                self.vec2 = np.array(object.l2)
            if object.color == "red":  # 记录最后一个red
                self.endpoint = [(object.l1[0] + object.l2[0]) / 2, (object.l1[1] + object.l2[1]) / 2]
        super(Running_On, self).__init__(map, seed, gamma, tau)

        self.restitution = 0.5
        self.print_log = False
        self.print_log2 = False

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
        self.distance_threshold = 30 if map_name in ['map_a', 'map_b'] else 50
        self.reward_range = (-10, 10)
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': 30
        }
        self.info = {'is_success': False}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_reward(self, pos):
        agent_reward = -1.

        np_pos = np.array(pos[:2])  # (x,y)
        d = point_distance_line(np_pos, self.vec1, self.vec2)
        if d < self.distance_threshold:
            agent_reward = 1.

        return agent_reward

    def get_intensive_reward(self, pos):
        """This function is used for implement of the intensive reward."""
        np_pos = np.array(pos[:2])
        np_endpoint = np.array(self.endpoint)
        agent_reward = (-np.sum(np.square(np_pos - np_endpoint)) ** 0.5) / 100

        return agent_reward

    def is_terminal(self, step_reward):
        info = {'success': False}

        if self.sparse_reward and step_reward > 0:  # 成功,不停止
            info = {'success': True}
        elif not self.sparse_reward and step_reward > -self.distance_threshold:  # 成功,不停止
            info = {'success': True}

        return info

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
            step_reward = self.get_reward(obs_myself)
        else:
            step_reward = self.get_intensive_reward(obs_myself)

        done = False
        if self.step_cnt >= self.max_step:
            done = True
        info = self.is_terminal(step_reward)  # done完全由步数控制

        return obs_myself, step_reward, done, info

    def close(self):
        pass
