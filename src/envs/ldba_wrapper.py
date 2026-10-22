import functools
from typing import Any, SupportsFloat, List
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import gymnasium
from gymnasium.core import WrapperObsType, WrapperActType
from gymnasium import spaces
from collections import deque

from envs import get_env_attr
from itertools import chain
from ltl.automata import ltl2ldba, ltl2spotldba, LDBA, LDBASequence
from ltl.logic import Assignment, FrozenAssignment
from ltl.req_cnt import build_cnt_monitor, build_req_monitor
from ltl.req_cnt import req_spec, cnt_spec


@dataclass
class CurrentState:
    """
    Holds one possible path that can be taken in the Buchi automaton.
    """
    state: int
    accepting: bool
    num_accepting_visits: int = 0

    def get_successor(self, state, accepting) -> 'CurrentState':
        return CurrentState(
            state,
            accepting,
            num_accepting_visits=self.num_accepting_visits + int(accepting),
        )


class MALDBAWrapperRRT(gymnasium.Wrapper):
    """
    Wrapper that keeps track of LTL goal satisfaction using an LDBA, which is added to the observation space.
    """

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        print(f"DEBUG: Using MALDBAWrapperRRT for env {env.spec.id}")
        if "PointLtlSafety" in env.spec.id or "SafetyPointMultiGoal" in env.spec.id:
            self.observation_space = spaces.Dict({
                # 16 dim for agent status, 16 dim for reach, and 16 dim for avoid
                'features': spaces.Box(-np.inf, np.inf, (48,), dtype=np.float32)
                # 'features': spaces.Box(-np.inf, np.inf, (80,), dtype=np.float32)
            })
        elif "LetterSafetyEnv" in env.spec.id:
            obs_dim = env.observation_space['features'].shape[0]
            self.observation_space = spaces.Dict({
                'features': spaces.Box(0, 1, (obs_dim, obs_dim, 1), dtype=np.float32)
            })

        self.possible_agents = self.env.possible_agents # + ['req', 'cnt']
        self.agent_obs_keys = ["accelerometer", "velocimeter", "gyro", "magnetometer", "wall_sensor"]
        self.region_order = env.get_propositions()
        self.terminate_on_acceptance = {agent: False for agent in self.possible_agents}
        self.ldba = {agent: None for agent in self.possible_agents}
        # LDBAs depend only on the goal formulas, not on episode state. Keep the
        # last formulas so repeated resets can reuse the (expensive) automata.
        self._ldba_formulas = None
        self.states = {agent: [] for agent in self.possible_agents}
        self.violating_states = {agent: [] for agent in self.possible_agents}
        self.is_success = {agent: False for agent in self.possible_agents}
        self.num_accepting_visits = {agent: 0 for agent in self.possible_agents}

    @property
    def ldba_state(self):
        """
        The first item is the de facto current state.
        This property returns its state number (index).
        Returns None if there are no possible states (happens when all paths were violating).
        """
        if self.states:
            return self.states[0].state
        else:
            return None

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        self.current_step += 1

        # Update trajectory
        props = info['propositions']
        props_set = set(chain.from_iterable(props.values()))
        props["req"] = props["cnt"] = props_set

        prev_state_indices = {agent: [s.state for s in self.states[agent]] for agent in self.possible_agents}

        # Update the possible states on every step: an empty letter is the
        # assignment where every proposition is false, which can change the
        # automaton state (e.g. leaving alpha in alpha EachReq beta).
        # Mapping from (state index, accepting) to state
        # If the same state can be reached in multiple ways, use the trajectory with the most accepting visits
        new_states = {agent: {} for agent in self.possible_agents}
        for agent in self.possible_agents:
            for state in self.states[agent]:
                for key in self.ldba[agent].get_next_states(state.state, props[agent]):
                    successor = state.get_successor(*key)
                    if self.ldba[agent].is_state_violating(successor.state):
                        self.violating_states[agent].append(successor)
                    elif key not in new_states[agent] or new_states[agent][key].num_accepting_visits < successor.num_accepting_visits:
                        new_states[agent][key] = successor
            self.states[agent] = [new_states[agent][k] for k in sorted(new_states[agent].keys())]

        if not all(self.states[agent] for agent in self.possible_agents):
            for a in self.possible_agents:
                terminated[a] = True
            info['violation'] = True

        was_success = self.is_success.copy()
        for agent in self.possible_agents:
            accepting_indices = [
                i for i, state in enumerate(self.states[agent]) if state.accepting]
            if (
                not was_success[agent]
                and self.terminate_on_acceptance[agent]
                and accepting_indices
            ):
                self.is_success[agent] = True
                # Update self.states such that an accepting path is first.
                if (i := accepting_indices[0]) != 0:
                    self.states[agent][0], self.states[agent][i] = (
                        self.states[agent][i], self.states[agent][0],)

        if all(self.is_success[agent] for agent in self.possible_agents):
            for a in self.possible_agents:
                terminated[a] = True
            info['success'] = True

        info['ldba_state_changed'] = {
            agent: (
                False
                if was_success[agent]
                else (
                    prev_state_indices[agent] != [s.state for s in self.states[agent]]
                    or self.is_success[agent]
                )
            )
            for agent in self.possible_agents
        }
        # print(f"DEBUG: ldba_state_changed = {info['ldba_state_changed']}")
            
        for agent in self.possible_agents:
            self.num_accepting_visits[agent] = max(
                [s.num_accepting_visits for s in (self.states[agent] if self.states[agent] else self.violating_states[agent])])
        info['num_accepting_visits'] = min(self.num_accepting_visits.values())

        self.complete_observation(obs, info)

        for i, a in enumerate(self.env.possible_agents):
            if info[a]['cost_collision']:
                info['violation'] = True
                terminated[a] = True
                info['violation_collision'] = True

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        WrapperObsType, dict[str, Any]]:

        obs, info = super().reset(seed=seed, options=options)
        self.obs = obs
        self.info= info
        formulas = dict(obs['goal'])
        if self._ldba_formulas != formulas:
            self.ldba = self.construct_ldba(formulas)
            self._ldba_formulas = formulas
        self.terminate_on_acceptance = {agent: self.ldba[agent].is_finite_specification() for agent in self.possible_agents}
        self.states = {
            agent: [
                CurrentState(
                    state=self.ldba[agent].initial_state,
                    # NOTE: LDBA implementation holds accepting info in transitions.
                    # As a result, initial state cannot be accepting.
                    accepting=False,
                )
            ]
            for agent in self.possible_agents
        }
        self.violating_states = {agent: [] for agent in self.possible_agents}
        self.complete_observation(obs, info)
        info['ldba_state_changed'] = {agent: True for agent in self.possible_agents}
        self.is_success = {agent: False for agent in self.possible_agents}
        self.num_accepting_visits = {agent: 0 for agent in self.possible_agents}
        self.current_step = 0
        self.previous_props_set = set()
        self.current_subgoals = {"reach": None, "avoid": None}

        return obs, info

    def complete_observation(self, obs: WrapperObsType, info: dict[str, Any] = None):
        obs['ldba'] = self.ldba
        obs['ldba_states'] = {
            agent: [s.state for s in self.states[agent]]
            for agent in self.possible_agents
        }
        obs['ldba_states_accepting'] = {
            agent: [s.accepting for s in self.states[agent]]
            for agent in self.possible_agents
        }
        obs['propositions'] = info['propositions']

    def construct_ldba(self, formula: dict[str, str]) -> dict[str, LDBA]:
        # construct ldba for each agent, and store them in a dictionary
        self.possible_agents = list(formula.keys())
        dict_ldba = {}
        for agent in self.possible_agents:
            if "req" in agent or "cnt" in agent:
                spec = req_spec.parse_req_spec(formula['req']).to_ltl().to_str('spot') if "req" in agent \
                    else cnt_spec.parse_cnt_spec(formula['cnt']).to_ltl().to_str('spot')
            else:
                spec = formula[agent]
            ldba = ltl2spotldba(spec)

            dict_ldba[agent] = ldba

        return dict_ldba

    def process_subgoals(self, subgoals: frozenset[FrozenAssignment], 
                         is_avoid: bool) -> dict[str, List[str]]:
        processed_subgoals = {agent: [] for agent in self.env.possible_agents}

        # for avoid, add gremlins lidar (used to sense other agents) first
        if is_avoid:
            for i, agent in enumerate(self.env.possible_agents):
                processed_subgoals[agent].append(f'gremlins_lidar_{i}')

        for sg in subgoals:
            color, agent_index = sg.to_string()[0].split("_")
            agent_key = f"agent_{agent_index}"
            processed_subgoals[agent_key].append(f"{color}_zones_lidar_{agent_index}")
        return processed_subgoals
    
    def pre_process_obs_zones(self, reach_zones, avoid_zones) -> np.ndarray:
        """
        pre-process the observation
        """
        original_obs = self.task.original_obs
        lidar_dim = self.task.lidar_conf.num_bins

        processed_obs = {agent: None for agent in self.env.possible_agents}
        for i, agent in enumerate(self.env.possible_agents):
            # reach subgoal observations
            if len(reach_zones[agent]) == 0:
                reach_obs = np.zeros(lidar_dim)
            else:
                reach_obs = np.vstack([original_obs[agent][r] for r in reach_zones[agent]]) # len(reach_zones) x lidar_dim
                reach_obs = np.max(reach_obs, axis=0) # lidar_dim
            # avoid subgoal observations
            if len(avoid_zones[agent]) == 0:
                avoid_obs = np.zeros(lidar_dim)
            else:
                avoid_obs = np.vstack([original_obs[agent][a] for a in avoid_zones[agent]]) # len(avoid_zones) x lidar_dim
                avoid_obs = np.max(avoid_obs, axis=0) # lidar_dim
            
            # atomic proposition independent observations
            agent_obs = np.concatenate([original_obs[agent][f"{key}_{i}"] for key in self.agent_obs_keys])
            processed_obs[agent] = np.concatenate([agent_obs, reach_obs, avoid_obs])

        # need stack those observations, maintain the order of agents
        processed_obs_array = np.stack([processed_obs[agent] for agent in self.env.possible_agents])
        assert processed_obs_array.shape == (len(self.env.possible_agents), 16 + 2 * lidar_dim), \
            f"observation shape is {processed_obs_array.shape}, but it should be {lidar_dim * 3}"

        return processed_obs_array
