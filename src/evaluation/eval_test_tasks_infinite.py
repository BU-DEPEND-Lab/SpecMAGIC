import sys
sys.path.insert(0, 'src')

import os

import pandas as pd
from tqdm import tqdm

from evaluation.simulate import simulate

import argparse
import multiprocessing as mp
import random

import numpy as np
import torch
from tqdm import tqdm

from envs import make_env_safety
from ltl import FixedSampler
from model.model import build_model_safety
from model.agent import MultiAgent
from config import model_configs
from sequence.search import ExhaustiveSearchSafety, SpotBFS
from utils.model_store import ModelStore
import argparse



env_to_tasks = {
    "SafetyPointMultiGoal0-v0":{
        'ma3_ind':
        [
            { 
                "agent_0": '(GF blue_0) & (GF green_0) & (G !yellow_0)',
                "agent_1": '(GF yellow_1) & (GF green_1) & (G !blue_1)',
                "agent_2": '(GF magenta_2) & (GF blue_2) & (G !green_2)',
            },
            {
                "agent_0": '(GF yellow_0) & (GF magenta_0) & (G !green_0)',
                "agent_1": '(GF blue_1) & (GF magenta_1) & (G !yellow_1)',
                "agent_2": '(GF green_2) & (GF yellow_2) & (G !blue_2)',
            },
            {
                "agent_0": '(GF green_0) & (GF blue_0) & (G !magenta_0)',
                "agent_1": '(GF magenta_1) & (GF yellow_1) & (G !green_1)',
                "agent_2": '(GF yellow_2) & (GF blue_2) & (G !magenta_2)',
            },
        ],
        'ma3_ind_req':
        [
            { 
                "agent_0": '(GF blue_0) & (GF green_0) & (G !yellow_0)',
                "agent_1": '(GF yellow_1) & (GF green_1) & (G !blue_1)',
                "agent_2": '(GF magenta_2) & (GF blue_2) & (G !green_2)',
                "req": "(blue_0 EachReq green_1) & (yellow_1 EachReq blue_2)",
            },
            {
                "agent_0": '(GF yellow_0) & (GF magenta_0) & (G !green_0)',
                "agent_1": '(GF blue_1) & (GF magenta_1) & (G !yellow_1)',
                "agent_2": '(GF green_2) & (GF yellow_2) & (G !blue_2)',
                "req": "(yellow_0 EachReq magenta_1) & (blue_1 EachReq yellow_2)",
            },
            {
                "agent_0": '(GF green_0) & (GF blue_0) & (G !magenta_0)',
                "agent_1": '(GF magenta_1) & (GF yellow_1) & (G !green_1)',
                "agent_2": '(GF yellow_2) & (GF blue_2) & (G !magenta_2)',
                "req": "(green_0 EachReq yellow_1) & (magenta_1 EachReq blue_2)",
            },
        ],
        'ma3_ind_cnt':
        [
            { 
                "agent_0": '(GF blue_0) & (GF green_0) & (G !yellow_0)',
                "agent_1": '(GF yellow_1) & (GF green_1) & (G !blue_1)',
                "agent_2": '(GF magenta_2) & (GF blue_2) & (G !green_2)',
                "cnt": "!Async[{0, 1, 2}, magenta, 2]",
            },
            {
                "agent_0": '(GF yellow_0) & (GF magenta_0) & (G !green_0)',
                "agent_1": '(GF blue_1) & (GF magenta_1) & (G !yellow_1)',
                "agent_2": '(GF green_2) & (GF yellow_2) & (G !blue_2)',
                "cnt": "!Sync[{0, 1, 2}, blue, 2]",
            },
            {
                "agent_0": '(GF green_0) & (GF blue_0) & (G !magenta_0)',
                "agent_1": '(GF magenta_1) & (GF yellow_1) & (G !green_1)',
                "agent_2": '(GF yellow_2) & (GF blue_2) & (G !magenta_2)',
                "cnt": "!Async[{0, 1, 2}, green, 2]",
            },
        ],
        'ma3_ind_req_cnt':
        [

            # 3 agents
            { 
                "agent_0": '(GF blue_0) & (GF green_0) & (G !yellow_0)',
                "agent_1": '(GF yellow_1) & (GF green_1) & (G !blue_1)',
                "agent_2": '(GF magenta_2) & (GF blue_2) & (G !green_2)',
                "req": "(blue_0 EachReq green_1) & (yellow_1 EachReq blue_2)",
                "cnt": "!Async[{0, 1, 2}, magenta, 2]",
            },
            {
                "agent_0": '(GF yellow_0) & (GF magenta_0) & (G !green_0)',
                "agent_1": '(GF blue_1) & (GF magenta_1) & (G !yellow_1)',
                "agent_2": '(GF green_2) & (GF yellow_2) & (G !blue_2)',
                "req": "(yellow_0 EachReq magenta_1) & (blue_1 EachReq yellow_2)",
                "cnt": "!Sync[{0, 1, 2}, blue, 2]",
            },
            {
                "agent_0": '(GF green_0) & (GF blue_0) & (G !magenta_0)',
                "agent_1": '(GF magenta_1) & (GF yellow_1) & (G !green_1)',
                "agent_2": '(GF yellow_2) & (GF blue_2) & (G !magenta_2)',
                "req": "(green_0 EachReq yellow_1) & (magenta_1 EachReq blue_2)",
                "cnt": "!Async[{0, 1, 2}, green, 2]",
            },
        ],
    }
}


def simulate(env, gamma, exp, seed, num_episodes, formula, finite, render, deterministic=True, key=None):

    env_name = env
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    sampler = FixedSampler.partial(formula)
    max_steps = None
    print(f"DEBUG: max_steps = {max_steps}, deterministic = {deterministic}")
    
    env = make_env_safety(env_name, sampler, max_steps, render_mode='human' if render else None)
    config = model_configs[env_name]
    model_store = ModelStore(env_name, exp, seed, None)
    training_status = model_store.load_training_status(map_location='cpu')

    model = build_model_safety(env, training_status, config)
    props = env.get_propositions()
    search = SpotBFS(env, model, props, num_loops=5)
    agent = MultiAgent(env, model, search=search, propositions=props, seed=seed, verbose=render)

    num_successes = 0
    num_violations = 0
    num_collisions = 0
    num_unreachable = 0
    num_accepting_visits = []
    steps = []
    rets = []

    env.reset(seed=seed)

    pbar = range(num_episodes)
    if not render:
        pbar = tqdm(pbar)
    for i in pbar:
        obs, info = env.reset()
        if render:
            print(obs['goal'])
        agent.reset()
        done = {a: False for a in env.possible_agents}
        num_steps = 0

        while not any(done.values()):

            action = agent.get_action(obs, info, deterministic=deterministic)
            if isinstance(action, np.ndarray):
                action = action.flatten()
                if action.shape == (1,):
                    action = action[0]
            obs, reward, done, info = env.step(action)
            num_steps += 1

            if any(done.values()):
                if finite:
                    final_reward = int('success' in info)
                    if 'success' in info:
                        num_successes += 1
                        steps.append(num_steps)
                    elif 'violation' in info:
                        num_violations += 1
                        if 'violation_collision' in info:
                            num_collisions += 1
                    rets.append(final_reward * gamma ** (num_steps - 1))
                    if not render:
                        pbar.set_postfix({
                            'S': num_successes / (i + 1),
                            'V': num_violations / (i + 1),
                            'CV': num_collisions / num_violations if num_violations > 0 else 0,
                            'ADR': np.mean(rets),
                            'AS': np.mean(steps),
                        })
                else:
                    if 'violation' in info:
                        num_violations += 1
                        if 'violation_collision' in info:
                            num_collisions += 1
                    else:
                        num_accepting_visits.append(info['num_accepting_visits'])
                    if not render:
                        pbar.set_postfix({
                            'A': np.mean(num_accepting_visits),
                            'V': num_violations / (i + 1),
                            'CV': num_collisions / num_violations if num_violations > 0 else 0,
                        })

    env.close()
    if finite:
        success_rate = num_successes / num_episodes
        violation_rate = num_violations / num_episodes
        collision_rate = num_collisions / num_violations if num_violations > 0 else 0
        unreachable_rate = num_unreachable / num_episodes
        average_steps = np.mean(steps)
        adr = np.mean(rets)
        print(f'{seed}: success_rate={success_rate:.3f},violation_rate={violation_rate:.3f},collision_rate={collision_rate:.3f},unreachable_rate={unreachable_rate:.3f},adr={adr:.3f},average_steps={average_steps:.3f}')
        return num_successes, num_violations, num_collisions, average_steps
    else:
        average_visits = np.mean(num_accepting_visits)
        violation_rate = num_violations / num_episodes
        collision_rate = num_collisions / num_violations if num_violations > 0 else 0
        unreachable_rate = num_unreachable / num_episodes
        print(f'{seed}: average_visits={average_visits:.3f},violation_rate={violation_rate:.3f},collision_rate={collision_rate:.3f},unreachable_rate={unreachable_rate:.3f}')
        return average_visits, num_violations, num_collisions


def main(env, exp, seeds, key, num_episodes=100):
    base_env = env.split('.')[0]  # e.g. SafetyPointMultiGoal0-v0.ma3 -> SafetyPointMultiGoal0-v0
    tasks = env_to_tasks[base_env][key]
    gamma = 0.99

    file_path = f'results_infinite/{env}_{exp}_{key}.csv'
    print(f"save to {file_path}")
    
    num_procs = len(seeds)*len(tasks)
    with mp.Pool(num_procs) as pool:
        args = []
        for task in tasks:
            print(f'Running task: {task}')
            for seed in seeds:
                args.append([env, gamma, exp, seed, num_episodes, task, False, False, True, key])
        for result in tqdm(pool.imap_unordered(eval_task, args), total=len(seeds)*len(tasks)):
            results = []
            (accepting_visits, vn, cn), seed, task = result
            results.append([exp, task, seed, accepting_visits, vn, cn])
            df = pd.DataFrame(results, columns=['method', 'task', 'seed', 'accepting_visits', 'violation_rate', 'collision_rate'])
            os.makedirs('results_infinite', exist_ok=True)

            if os.path.exists(file_path):
                df.to_csv(file_path, mode='a', header=False, index=False)
            else:
                df.to_csv(file_path, index=False)


def eval_task(simulate_args):
    return simulate(*simulate_args), simulate_args[3], simulate_args[5]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='SafetyPointMultiGoal0-v0.ma3')
    parser.add_argument('--exp', type=str, default='GenZ-LTL')
    parser.add_argument('--seed', type=int, nargs='+', default=[1], help="List of seeds")
    parser.add_argument('--key', type=str, )
    parser.add_argument('--num_episodes', type=int, default=100)
    args = parser.parse_args()
    main(args.env, args.exp, args.seed, args.key, args.num_episodes)
