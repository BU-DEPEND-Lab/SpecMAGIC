import sys
import time
sys.path.insert(0, 'src')

import random
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import trange

from envs import make_env_safety
from ltl import FixedSampler
from model.model import build_model, build_model_safety
from model.agent import Agent, MultiAgent
from config import model_configs
from sequence.search import ExhaustiveSearchSafety, NoPathsException, SpotBFS
from utils.model_store import ModelStore
from visualize.zones import draw_trajectories
import imageio
import os


env_name = 'SafetyPointMultiGoal0-v0.ma10'  # .ma3 / .ma5 / .ma10, must match the number of agents in `formula`
exp = 'specmagic'

seed = 22

formula = { 
    "agent_0": '!(magenta_0 | yellow_0) U (blue_0 & F green_0)',
    "agent_1": '!magenta_1 U ((green_1 | blue_1) & F yellow_1)',
    "agent_2": '(!magenta_2 U (blue_2 | green_2)) & (F magenta_2)',
    "agent_3": '(F magenta_3) & (!yellow_3 U (blue_3 | green_3))',
    "agent_4": '!magenta_4 U ((blue_4 | yellow_4) & F green_4)',
    "agent_5": '!(green_5 | yellow_5) U (magenta_5 & F blue_5)',
    "agent_6": '(F blue_6) & (!blue_6 U (green_6 & F yellow_6))',
    "agent_7": '!(blue_7 | yellow_7) U (green_7 & F magenta_7)',
    "agent_8": '!green_8 U (yellow_8 & F (blue_8 | magenta_8))',
    "agent_9": '!(magenta_9 | yellow_9) U (blue_9 & F green_9)',
    "req": "(blue_0 Req green_1) & (blue_1 Req green_2) & (blue_2 Req magenta_3) & (blue_3 Req blue_4) & "
           "(green_4 Req magenta_5) & (blue_5 Req green_6) & (blue_6 Req green_7) & (magenta_7 Req blue_8) & (yellow_8 Req blue_9)",
    "cnt": "Async[{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, blue, 9]",
    # "agent_0": '(GF blue_0) & (GF green_0) & (G !yellow_0)',
    # "agent_1": '(GF yellow_1) & (GF green_1) & (G !blue_1)',
    # "agent_2": '(GF magenta_2) & (GF blue_2) & (G !green_2)',
    # "req": "(blue_0 EachReq green_1) & (yellow_1 EachReq blue_2)",
    # "cnt": "!Async[{0, 1, 2}, magenta, 2]",
}

render_mode = "rgb_array"

random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

print(f"DEBUG: formula = {formula}")
sampler = FixedSampler.partial(formula)
deterministic = True
max_steps = None # will use default episode length
verbose = False
finite = False if any("GF" in f or "FG" in f for f in formula.values()) else True

env = make_env_safety(env_name, sampler, render_mode=render_mode, max_steps=max_steps)
config = model_configs[env_name]
model_store = ModelStore(env_name, exp, seed, None)

training_status = model_store.load_training_status(map_location='cpu')
model = build_model_safety(env, training_status, config)

props = env.get_propositions()
search = SpotBFS(env, model, props, num_loops=1)
agent = MultiAgent(env, model, search=search, propositions=props, seed=seed, verbose=verbose)

num_episodes = 20
target_success_nums = 2
current_success_nums = 0
target_failure_nums = 4
current_failure_nums = 0
trajectories = []
zone_poss = []
titles = []
success, violation = 0, 0
num_unreachable = 0

env.reset(seed=seed)

pbar = trange(num_episodes)
total_frames = []
num_accepting_visits = []

for i in pbar:
    print(f"DEBUG: Starting episode {i+1}/{num_episodes}")
    print(f"DEBUG: current success_nums = {current_success_nums}, target_success_nums = {target_success_nums}")
    obs, info = env.reset()
    agent.reset()
    done = {a: False for a in env.possible_agents}
    agent_traj = []
    episode_frames = []
    
    while not any(done.values()):
        action = agent.get_action(obs, info, deterministic=deterministic)
        obs, reward, done, info = env.step(action)
        frame = env.render()
        episode_frames.append(frame)

        if any(done.values()):
            # total_frames += episode_frames
            if finite:
                if "success" in info:
                    titles.append(f"{formula}_success")
                    total_frames += episode_frames
                    current_success_nums += 1
                    success += 1
                elif "violation" in info:
                    titles.append(f"{formula}_violation")
                    violation += 1
                else:
                    titles.append(f"{formula}_not_finish")
                
                pbar.set_postfix({
                                'S': success / (i+1),
                                'V': violation / (i+1),
                            })
            else:
                if "violation" in info:
                    violation += 1
                else:
                    num_accepting_visits.append(info['num_accepting_visits'])
                    total_frames += episode_frames
                    current_success_nums += 1
                pbar.set_postfix({
                    'A': np.mean(num_accepting_visits),
                    'V': violation / (i + 1),
                })
        if current_success_nums >= target_success_nums:
            break
print(f"Formula: {formula}, Success: {success}, Violation: {violation}")
env.close()


mp4_name = f"ma10_finite.mp4"
print(f"DEBUG: saving video to {mp4_name}")
imageio.mimsave(mp4_name, total_frames, fps=90)
