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
    with open('logs/' + name + '.json', 'w') as f:
        f.write(json.dumps(record))

def load_record(path):
    file = open(path, "rb")
    filejson = json.load(file)
    return filejson

RENDER = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', default="map_a", type=str)
    parser.add_argument("--seed", default=1, type=int)
    args = parser.parse_args()

    agent1 = random_agent()

    map_index_seq = list(range(1, 5))
    time_s = time.time()
    for i in range(20):
        print("==========================================")
        Gamemap = create_scenario(args.map, "maps_off.json")

        game = Running(Gamemap)

        obs = game.reset()
        if RENDER:
            game.render()

        done = False
        step = 0
        if RENDER:
            game.render()

        while not done:
            step += 1
            action1 = agent1.act(obs[0])
            obs, reward, done, _ = game.step([action1])

            if RENDER:
                game.render()

        print('Episode Reward = {}'.format(reward))
