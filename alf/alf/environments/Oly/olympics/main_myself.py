import time
import json
import numpy as np
from scenario.running_off import Running_Off  # 自定义
from scenario.running_on import Running_On  # 自定义
from generator import create_scenario

from agent import *

# % 模型相关
def store(record, name):
    with open('logs/' + name + '.json', 'w') as f:
        f.write(json.dumps(record))

def load_record(path):
    file = open(path, "rb")
    filejson = json.load(file)
    return filejson

# % 运行
if __name__ == "__main__":
    # % 加载环境
    maps = "map_d"
    Gamemap = create_scenario(maps, "maps_off.json")
    env = Running_Off(Gamemap, map_name=maps, seed=None, max_episode_steps=1200, expand_state=False, sparse_reward=True)
    obs = env.reset()

    # % 加载算法
    agent = SAC_agent()

    done = False
    step = 0
    while True:
        if done:
            obs = env.reset()
        time.sleep(0.01)
        step += 1
        action = agent.act(obs)
        print(np.round(obs,3))
        env.render()

        obs, reward, done, _ = env.step(action)

        print(np.round(action,3))
        print('--------------')
        # print(obs)
