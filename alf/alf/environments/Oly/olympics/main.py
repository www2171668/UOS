import sys
from os import path
father_path = path.dirname(__file__)
sys.path.append(str(father_path))
from generator import create_scenario
import argparse
from agent import *
import time
from scenario.running import Running


import random
import numpy as np
import matplotlib.pyplot as plt
import json

def store(record, name):
    with open('logs/'+name+'.json', 'w') as f:
        f.write(json.dumps(record))

def load_record(path):
    file = open(path, "rb")
    filejson = json.load(file)
    return filejson

RENDER = True

if __name__ == "__main__":
    agent1 = random_agent()
    agent2 = random_agent()
    for i in range(20):
        print("==========================================")
        Gamemap = create_scenario("map1", "maps.json")
        game = Running(Gamemap, seed = 1)

        obs = game.reset()
        done = False
        step = 0
        if RENDER: game.render()

        while not done:
            step += 1
            action1 = agent1.act(obs[0])
            action2 = agent2.act(obs[1])
            obs, reward, done, _ = game.step([action1, action2])

            if RENDER: game.render()

        print('Episode Reward = {}'.format(reward))

