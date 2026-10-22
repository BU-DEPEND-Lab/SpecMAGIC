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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, choices=['PointLtl2-v0', 'PointLtlSafety2-v0', 'LetterEnv-v0', 'FlatWorld-v0'], default='PointLtl2-v0')
    parser.add_argument('--exp', type=str, default='deepset')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--formula', type=str, default='(F blue) & (!blue U (green & F yellow))')  # !(red | green) U magenta  !blue U (magenta & red)
    parser.add_argument('--finite', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--render', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    gamma = 0.94 if args.env == 'LetterEnv-v0' else 0.998 if args.env == 'PointLtl2-v0' or args.env == 'PointLtlSafety2-v0' else 0.98
    return simulate(args.env, gamma, args.exp, args.seed, args.num_episodes, args.formula, args.finite, args.render, args.deterministic)


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
    search = SpotBFS(env, model, props, num_loops=1)
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


if __name__ == '__main__':
    main()
