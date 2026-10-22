from typing import Any, SupportsFloat, Callable, List

import numpy as np
import gymnasium
from gymnasium import spaces
from itertools import chain
from gymnasium.core import WrapperObsType, WrapperActType

from ltl.automata import LDBASequence
from ltl.logic import Assignment, FrozenAssignment


class SequenceWrapper(gymnasium.Wrapper):
    """
    Wrapper that adds a reach-avoid sequence of propositions to the observation space.
    """

    def __init__(self, env: gymnasium.Env, sample_sequence: Callable[[], LDBASequence], partial_reward=False):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            'features': env.observation_space,
        })
        self.sample_sequence = sample_sequence
        self.goal_seq = None
        self.num_reached = 0
        self.propositions = set(env.get_propositions())
        self.partial_reward = partial_reward
        self.obs = None
        self.info = None

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if (action == LDBASequence.EPSILON).all():
            obs, _, terminated, truncated, info = self.apply_epsilon_action()
            reward = 0.
        else:
            assert not (action == LDBASequence.EPSILON).any()
            obs, reward, terminated, truncated, info = super().step(action)
        reach, avoid = self.goal_seq[self.num_reached]
        active_props = info['propositions']
        assignment = Assignment({p: (p in active_props) for p in self.propositions}).to_frozen()
        if assignment in avoid:
            reward = -1.
            info['violation'] = True
            terminated = True
        elif reach != LDBASequence.EPSILON and assignment in reach:
            self.num_reached += 1
            terminated = self.num_reached >= len(self.goal_seq)
            if terminated:
                info['success'] = True
            if self.partial_reward:
                reward = 1. if terminated else 1 / (len(self.goal_seq) - self.num_reached + 1)
            else:
                reward = 1. if terminated else 0
        
        cost = 1.0 if reward == -1. else 0.0
        reach, avoid = self.goal_seq[self.num_reached] \
            if self.num_reached < len(self.goal_seq) else self.goal_seq[-1]
        obs = self.complete_observation(obs, info)
        
        self.obs = obs
        self.info = info
        # obs = self.complete_observation(obs, info)
        return obs, (reward, cost), terminated, truncated, info

    def apply_epsilon_action(self):
        assert self.goal_seq[self.num_reached][0] == LDBASequence.EPSILON
        self.num_reached += 1
        return self.obs, 0.0, False, False, self.info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.goal_seq = self.sample_sequence()
        self.num_reached = 0
        obs = self.complete_observation(obs, info)
        self.obs = obs
        self.info = info
        return obs, info

    def complete_observation(self, obs: WrapperObsType, info: dict[str, Any] = None) -> WrapperObsType:
        return {
            'features': obs,
            'goal': self.goal_seq[self.num_reached:],
            'initial_goal': self.goal_seq,
            'propositions': info['propositions'],
        }


class SequenceSafetyWrapper(gymnasium.Wrapper):
    """
    Wrapper that adds a reach-avoid sequence of propositions to the observation space.
    """

    def __init__(self, env: gymnasium.Env, sample_sequence: Callable[[], LDBASequence], partial_reward=False):
        super().__init__(env)
        if "PointLtlSafety" in env.spec.id or "SafetyPointMultiGoal" in env.spec.id:
            self.observation_space = spaces.Dict({
                # 16 dim for agent status, 16 dim for reach, and 16 dim for avoid
                'features': spaces.Box(-np.inf, np.inf, (48,), dtype=np.float32)
            })
        elif "LetterSafetyEnv" in env.spec.id:
            obs_dim = env.observation_space.shape[0]
            self.observation_space = spaces.Dict({
                'features': spaces.Box(0, 1, (obs_dim, obs_dim, 1), dtype=np.float32)
            })
        self.sample_sequence = sample_sequence
        self.goal_seq = None
        self.num_reached = 0 # always 0
        self.agent_obs_keys = ["accelerometer", "velocimeter", "gyro", "magnetometer", "wall_sensor"] # from safety_gymnasium
        self.region_order = env.get_propositions()
        self.propositions = set(self.region_order)
        # self.trained_agents = ["agent_0", "agent_1", "agent_2"]

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        # terminated = terminated["agent_0"]
        # truncated = truncated["agent_0"]

        reach, avoid = self.goal_seq[self.num_reached]
        reach_set = set(r.to_string()[0] for r in reach)
        avoid_set = set(a.to_string()[0] for a in avoid)
        
        active_props = info['propositions']
        # assignment = {}
        # for p in active_props:
        #     zone, agent = p.split('_')
        #     assignment[agent] = Assignment({p: (p == zone) for p in self.env.colors}).to_frozen()

        # assignment = Assignment({p: (p in active_props) for p in self.propositions}).to_frozen()
        # assignment = frozenset([Assignment.single_proposition(p, self.propositions).to_frozen() for p in active_props])
        # assignment = set(active_props)
        assignment = set(chain.from_iterable(active_props.values()))
        # print(f"DEBUG: assignment = {assignment}, props_set = {props_set}")
        # if active_props:
        #     print(f"DEBUG: assignment = {assignment}, reach = {reach_set}, avoid = {avoid_set}, is_reach = {not assignment.isdisjoint(reach_set)}, is_avoid = {not assignment.isdisjoint(avoid_set)}")
        #     print(f"DEBUG: active_props = {active_props}, assignment = {assignment}")
        
        
        # reward = 0.0; cost = -1.0; terminated = False
        reward = {agent: 0.0 for agent in self.env.possible_agents}
        cost = {agent: -1.0 for agent in self.env.possible_agents}
        terminated = {agent: False for agent in self.env.possible_agents}
        # reward = {agent: 0.0 for agent in self.trained_agents}
        # cost = {agent: 0.0 for agent in self.trained_agents}
        # terminated = {agent: False for agent in self.trained_agents}

        for agent in self.env.possible_agents:
        # for agent in self.trained_agents:
            info[agent]["violation_sources"] = []
        # reach the "avoid" area
        # if assignment in avoid:
        
        if not assignment.isdisjoint(avoid_set):
            # print(f"DEBUG: violation! reached = {assignment}, avoid = {avoid}")
            
            # get the overlaps and only assign cost and termination to the agents that are involved in the violation
            overlaps = assignment.intersection(avoid_set)
            for p in overlaps:
                _, agent_index = p.split('_')
                agent = f"agent_{agent_index}"
                # if agent not in self.trained_agents:
                #     continue
                cost[agent] = 1.0; terminated[agent] = True
                info[agent]['violation'] = True
                info[agent]["violation_sources"].append("avoid")
            
            # print(f"DEBUG: violation! reached {assignment}")
        # reach the "reach" area
        # elif assignment in reach:
        elif not assignment.isdisjoint(reach_set):
            # print(f"DEBUG: success! reached = {assignment}, reach = {reach}")
            # reward = 1.0; info['success'] = True

            # also need to check which agent reaches the goal and only assign reward to that agent
            overlaps = assignment.intersection(reach_set)
            for p in overlaps:
                _, agent_index = p.split('_')
                agent = f"agent_{agent_index}"
                reward[agent] = 1.0; info[agent]['success'] = True

            # sample new subgoal
            self.goal_seq = self.sample_sequence(assignment)
            # print(f"DEBUG: new goal_seq = {self.goal_seq}")
            reach, avoid = self.goal_seq[self.num_reached]
        # # reach the boundary of the environment
        # elif 'cost_ltl_walls' in info["agent_0"] and info["agent_0"]['cost_ltl_walls'] > 0:
        #     # print(f"DEBUG: hit wall!")
        #     cost = 1.0; terminated = True
        # elif 'cost_collision' in info["agent_0"] and info["agent_0"]['cost_collision'] > 0:
        #     # print(f"DEBUG: collision!")
        #     cost = 1.0; terminated = True

        # check if any agent reaches the boundary or collides with other agent, if so, assign cost and termination to the involved agents
        for agent in self.env.possible_agents:
        # for agent in self.trained_agents:
            if info[agent]['cost_ltl_walls'] > 0:
                cost[agent] = 1.0; terminated[agent] = True
                info[agent]["violation_sources"].append("wall")
                # print(f"DEBUG: {agent} hit wall!")
            if info[agent]['cost_collision'] > 0:
                cost[agent] = 1.0; terminated[agent] = True
                info[agent]['violation'] = True
                info[agent]["violation_sources"].append("collision")
                # print(f"DEBUG: {agent} collision!")

        # v2: any agent violating its current subgoal ends the rollout for every agent.
        if any(cost[agent] > 0 for agent in self.env.possible_agents):
            terminated = {agent: True for agent in self.env.possible_agents}
        # if any(cost[agent] > 0 for agent in self.trained_agents):
        #     terminated = {agent: True for agent in self.trained_agents}

        # v3: conservative reward: if the agent violates its subgoal, then 0 reward else reward[agent]
        for agent in self.env.possible_agents:
        # for agent in self.trained_agents:
            if cost[agent] > 0:
                reward[agent] = 0
                info[agent].pop("success", None)
                # assert info[agent]["violation"] == True

        obs = self.pre_process_obs(reach, avoid)
        obs = self.complete_observation_current(obs, info)
        self.obs = obs
        self.info = info
        # obs = self.complete_observation(obs, info)
        return obs, (reward, cost), terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.goal_seq = self.sample_sequence()
        self.num_reached = 0
        reach, avoid = self.goal_seq[self.num_reached]
        
        obs = self.pre_process_obs(reach, avoid)
        obs = self.complete_observation_current(obs, info)
        return obs, info

    def pre_process_obs(self, reach, avoid):
        if "PointLtlSafety" in self.env.spec.id or "SafetyPointMultiGoal" in self.env.spec.id:
            obs = self.pre_process_obs_zones(reach, avoid)
        elif "LetterSafetyEnv" in self.env.spec.id:
            obs = self.pre_process_obs_letter(reach, avoid)
        return obs

    # def pre_process_obs_zones(self,
    #                     reach: frozenset[FrozenAssignment], 
    #                     avoid: frozenset[FrozenAssignment]) -> np.ndarray:
    #     """
    #     observation reduction
    #     """
    #     original_obs = self.task.original_obs
    #     lidar_dim = self.task.lidar_conf.num_bins
    #     agent_obs = np.concatenate([original_obs[key] for key in self.agent_obs_keys])

    #     reach_zones = [r.to_string()[0] + "_zones_lidar" for r in list(reach)]
    #     avoid_zones = [a.to_string()[0] + "_zones_lidar" for a in list(avoid) if a.to_string()]
        
    #     reach_obs = np.vstack([original_obs[color] for color in reach_zones])
    #     reach_obs = np.max(reach_obs, axis=0) # lidar_dim
    #     if len(avoid_zones):
    #         avoid_obs = np.vstack([original_obs[color] for color in avoid_zones])
    #         avoid_obs = np.max(avoid_obs, axis=0) # lidar_dim
    #     else:
    #         avoid_obs = np.zeros(lidar_dim)
            
    #     assert agent_obs.shape == reach_obs.shape == avoid_obs.shape == (lidar_dim,)
    #     return np.concatenate([agent_obs, reach_obs, avoid_obs])

    def process_subgoals(self, subgoals: frozenset[FrozenAssignment], 
                         is_avoid: bool) -> dict[str, List[str]]:
        # print(f"DEBUG: is_avoid = {is_avoid}")
        processed_subgoals = {agent: [] for agent in self.env.possible_agents}
        # processed_subgoals = {agent: [] for agent in self.trained_agents}

        # for avoid, add gremlins lidar (used to sense other agents) first
        if is_avoid:
            for i, agent in enumerate(self.env.possible_agents):
            # for i, agent in enumerate(self.trained_agents):
                # if i > 0: continue # only process for agent 0
                # agent_index = agent.split("_")[1] 
                # agent_index = "" if int(agent_index) == 0 else agent_index
                processed_subgoals[agent].append(f'gremlins_lidar_{i}')

        for sg in subgoals:
            color, agent_index = sg.to_string()[0].split("_")
            # if agent_index != "0": continue  # only process for agent 0
            agent_key = f"agent_{agent_index}"
            # if agent_key not in processed_subgoals:
            #     continue
            # agent_index = "" if int(agent_index) == 0 else agent_index
            processed_subgoals[agent_key].append(f"{color}_zones_lidar_{agent_index}")
        return processed_subgoals

    def pre_process_obs_zones(self,
                        reach: frozenset[FrozenAssignment], 
                        avoid: frozenset[FrozenAssignment]) -> np.ndarray:
        """
        pre-process the observation
        """
        original_obs = self.task.original_obs
        lidar_dim = self.task.lidar_conf.num_bins
        # print(f"DEBUG: original_obs = {original_obs}")
        
        reach_zones = self.process_subgoals(reach, False)
        # print(f"DEBUG: reach_zones = {reach_zones}")

        avoid_zones = self.process_subgoals(avoid, True)
        # print(f"DEBUG: avoid_zones = {avoid_zones}")

        # if "blue_zones_lidar" not in reach_zones["agent_0"]:
        #     reach_zones = {agent: [] for agent in self.env.agents}
        #     avoid_zones = {"agent_0": avoid_zones["agent_0"] + ["blue_zones_lidar"], "agent_1" : avoid_zones["agent_1"]}
        # print(f"DEBUG: reach = {reach_zones}, avoid = {avoid_zones}")
        # other_agents_lidar = {agent: [] for agent in self.env.agents}
        # for agent in self.env.agents:
        #     agent_index = agent.split("_")[1] 
        #     agent_index = "" if int(agent_index) == 0 else agent_index
        #     other_agents_lidar[agent] = [f'gremlins_lidar{agent_index}']
        # # {agent: [f'gremlins_lidar{agent_index}']} for agent in self.env.agents agent_index = agent.split("_")[1] 
        # avoid_zones.update(other_agents_lidar)
        # print(f"DEBUG: avoid_zones = {avoid_zones}")

        # reach_obs0 = np.vstack([original_obs["agent_0"][color] \
        #                         for color in reach_zones if color in original_obs["agent_0"]]) # len(reach_zones) x lidar_dim
        # reach_obs0 = np.max(reach_obs0, axis=0) # lidar_dim

        # reach_obs0 = np.vstack([original_obs["agent_0"][color] \
        #                         for color in reach_zones if color in original_obs["agent_0"]]) # len(reach_zones) x lidar_dim
        # reach_obs0 = np.max(reach_obs0, axis=0) # lidar_dim

        processed_obs = {agent: None for agent in self.env.possible_agents}
        for i, agent in enumerate(self.env.possible_agents):
        # processed_obs = {agent: None for agent in self.trained_agents}
        # for i, agent in enumerate(self.trained_agents):

            # if i > 0: continue # only process for agent 0

            # reach subgoal observations
            if len(reach_zones[agent]) == 0:
                reach_obs = np.zeros(lidar_dim)
            else:
                reach_obs = np.vstack([original_obs[agent][r] for r in reach_zones[agent]]) # len(reach_zones) x lidar_dim
                reach_obs = np.max(reach_obs, axis=0) # lidar_dim

            # avoid subgoal observations
            avoid_obs = np.vstack([original_obs[agent][a] for a in avoid_zones[agent]]) # len(avoid_zones) x lidar_dim
            avoid_obs = np.max(avoid_obs, axis=0) # lidar_dim

            # atomic proposition independent observations
            # suffix = str(i) if i > 0 else ""
            agent_obs = np.concatenate([original_obs[agent][f"{key}_{i}"] for key in self.agent_obs_keys])
            processed_obs[agent] = np.concatenate([agent_obs, reach_obs, avoid_obs])

        # print(f"DEBUG: processed_obs = {processed_obs}")
        # need stack thos observations, maintain the order of agents
        processed_obs_array = np.stack([processed_obs[agent] for agent in self.env.possible_agents if processed_obs[agent] is not None])
        # processed_obs_array = np.stack([processed_obs[agent] for agent in self.trained_agents if processed_obs[agent] is not None])

        # processed_obs_array = np.squeeze(processed_obs_array)  # remove the extra dimension
        # print(f"processed_obs_array = {processed_obs_array.shape}")
        # for agent in self.env.possible_agents:
        #     assert processed_obs[agent].shape == self.observation_space['features'].shape, \
        #     f"obs.shape = {processed_obs[agent].shape}, expected {self.observation_space['features'].shape}"

        # the shape of processed_obs_array should be the same as (num_agents, self.obs_space.shape[0])
        target_obs_shape = (self.num_agents, self.observation_space['features'].shape[0])
        # target_obs_shape = (len(self.trained_agents), self.observation_space['features'].shape[0])
        assert processed_obs_array.shape == target_obs_shape, \
            f"obs.shape = {processed_obs_array.shape}, expected {target_obs_shape}"
        # assert processed_obs_array.shape == (self.num_agents, self.observation_space['features'].shape, \
        #     f"obs.shape = {processed_obs_array.shape}, expected {self.observation_space['features'].shape}"
        # assert processed_obs_array.shape == (len(self.env.possible_agents), 16 + 2 * lidar_dim), \
        #     f"observation shape is {processed_obs_array.shape}, but it should be {lidar_dim * 3}"
        # print(f"DEBUG: processed_obs_array.shape = {processed_obs_array.shape}")
        return processed_obs_array
        # return processed_obs_array
        # ### avoid
        # avoid_zones = ["_".join(sorted(l)) + "_zones_lidar" for a in list(avoid) if (l := a.to_string())]
        # if len(avoid_zones):
        #     avoid_obs = np.vstack([original_obs[color] for color in avoid_zones]) # len(avoid_zones) x lidar_dim
        #     avoid_obs = np.max(avoid_obs, axis=0) # lidar_dim
        # else:
        #     avoid_obs = np.zeros(lidar_dim)
            
        # agent_obs = np.concatenate([original_obs[key] for key in self.agent_obs_keys])
        # assert agent_obs.shape == reach_obs.shape == avoid_obs.shape == (lidar_dim,)
        # return np.concatenate([agent_obs, reach_obs, avoid_obs])
    

    def pre_process_obs_letter(self,
                        reach: frozenset[FrozenAssignment], 
                        avoid: frozenset[FrozenAssignment]) -> np.ndarray:
        """
        observation reduction
        """
        obs = self.env.original_obs
        new_obs = np.zeros((obs.shape[0], obs.shape[1]), dtype=obs.dtype)
        letter_to_index = {letter: i for i, letter in enumerate(self.region_order)}
        
        reach_indices = [letter_to_index[r.to_string()[0]] for r in list(reach)]
        avoid_indices = [letter_to_index[a.to_string()[0]] for a in list(avoid)]
        
        reach_mask = np.any(obs[:, :, reach_indices] > 0, axis=2)
        avoid_mask = np.any(obs[:, :, avoid_indices] > 0, axis=2)
        agent_mask = obs[:, :, -1] > 0
        
        new_obs = np.zeros(obs.shape[:2], dtype=np.float32)
         # specific values do not matter as long as they are distinct
        new_obs[avoid_mask] = 0.5
        new_obs[reach_mask] = 1.0
        new_obs[agent_mask] = 0.2
        return new_obs[..., None]
        
    def complete_observation_current(self, obs: WrapperObsType, info: dict[str, Any] = None) -> WrapperObsType:
        return {
            'features': obs,
            'goal': self.goal_seq,
            'initial_goal': self.goal_seq,
            'propositions': info['propositions'],
        }

