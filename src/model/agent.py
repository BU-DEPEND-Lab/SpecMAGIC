import numpy as np
import torch

import preprocessing
from model.model import Model
from sequence.search import SequenceSearch
from sequence.samplers import ReachAvoidRRTSearch
from ltl.logic import Assignment
from sequence.samplers import post_process_reach_avoid_sequence
import torch_ac


class Agent:
    def __init__(self, env, model: Model, search: SequenceSearch, propositions: set[str], verbose=False, timeout=None):
        self.env = env
        self.model = model
        self.search = search
        self.propositions = propositions
        self.verbose = verbose
        self.sequence = None
        # Timeout mechanism
        self.timeout = 1000 # 300 # timeout if timeout else float('inf')
        self.current_goal_steps = 0

    def reset(self):
        self.sequence = None

    def get_action(self, obs, info, deterministic=False) -> np.ndarray:
        if 'ldba_state_changed' in info or self.sequence is None:
            prev_seq = self.sequence
            self.sequence = self.search(obs['ldba'], obs['ldba_states'], obs)
            if self.sequence != prev_seq:
                self.current_goal_steps = 0
            if self.verbose:
                print(f'Selected sequence: {self.sequence}')
        else:
            self.current_goal_steps += 1
            if self.current_goal_steps >= self.timeout:
                # Mark the current subgoal as unfeasible, but not for the accepting states.
                # Because staying in accepting states is the goal; we should never timeout in accepting states.
                unfeasible_states = [s for s, accepting in zip(obs['ldba_states'], obs['ldba_states_accepting']) if not accepting]
                if unfeasible_states:
                    true_props = set()
                    for a in self.sequence[0][0]:
                        true_props = true_props.union(a.get_true_propositions())
                    reach_assignment = Assignment.where(*true_props, propositions=obs['ldba'].propositions).to_frozen()
                    obs['ldba'].mark_unfeasible(unfeasible_states, reach_assignment)
                    prev_seq = self.sequence
                    self.sequence = self.search(obs['ldba'], obs['ldba_states'], obs)
                    # If there are no sequences, an exception will be raised by search.
                    # We don't need to check that condition here.
                    assert self.sequence != prev_seq
                # Reset the step counter for the next timeout.
                self.current_goal_steps = 0
        # assert self.sequence is not None
        if self.sequence is None:
            self.sequence = [('', '')]
        obs['goal'] = self.sequence
        # print(f"DEBUG: sequence = {self.sequence}")
        raw_actions = self.forward(obs, deterministic)
        # remove redundant 1-dim
        if raw_actions.ndim > 1:
            raw_actions = raw_actions.squeeze()
        # print(f"DEBUG: action after squeeze = {raw_actions}, shape = {raw_actions.shape}")
        actions = {}
        for i, a in enumerate(self.env.possible_agents):
            # assert not np.any(np.isnan(raw_actions)), f"NaN detected in action for agent {a}"
            actions[a] = raw_actions[i, :]
        return actions

    def forward(self, obs, deterministic=False) -> np.ndarray:
        
        if self.sequence is not None:
            reach, avoid = self.sequence[0]
            # print(f"DEBUG: obs = {obs}")
            # if len(obs["features"].shape) == 1:
            obs["features"] = self.env.pre_process_obs_zones(reach, avoid)
            # else:
            # obs["features"] = self.env.pre_process_obs_letter(reach, avoid)
        
        if not (isinstance(obs, list) or isinstance(obs, tuple)):
            obs = [obs]
        preprocessed = preprocessing.preprocess_obss(obs, self.propositions)
        with torch.no_grad():
            dist, _, _ = self.model(preprocessed)
            action = dist.mode if deterministic else dist.sample()
        return action.detach().numpy()


class MultiAgent:
    def __init__(self, env, model: Model, search: SequenceSearch, propositions: set[str], seed: int, verbose=False, timeout=None):
        self.env = env
        self.model = model
        self.search = search
        self.propositions = propositions
        self.verbose = verbose
        self.sequence = {agent: [] for agent in self.env.possible_agents}
        # self.selected_subgoal = {agent: [] for agent in self.env.possible_agents}
        self.selected_subgoal = {"reach": None, "avoid": None}
        self.rrt_sampler = ReachAvoidRRTSearch(seed)
        # Timeout mechanism
        self.timeout = 2000
        self.current_goal_steps = 0

    def reset(self):
        self.sequence = {agent: [] for agent in self.env.possible_agents}
        self.selected_subgoal = {"reach": None, "avoid": None}
        self.current_goal_steps = 0

    def get_action(self, obs, info, deterministic=False) -> np.ndarray:

        is_need_replan = False

        for agent in self.env.possible_agents:
            if info['ldba_state_changed'][agent]:
                self.sequence[agent] = self.search(obs['ldba'][agent], obs['ldba_states'][agent], obs)
                is_need_replan = True
            
            if self.env.is_success[agent]:
                self.sequence[agent] = []

        if is_need_replan:

            sequence = {agent: self.sequence[agent] for agent in self.env.possible_agents if len(self.sequence[agent]) > 0}
            final_sequence = self.rrt_sampler(sequence, max_iterations=100_000)

            result = self.rrt_sampler.last_result
            assert result is not None

            if result.path is None:
                print("\nRRT result: FAILURE")

            final_sequence = post_process_reach_avoid_sequence(final_sequence)
            
            subgoal = final_sequence[0]
            reach = subgoal.reach; avoid = subgoal.avoid
            reach_set = set(reach.split(" & ")) # reach is a conjunction of propositions
            avoid_set = set(avoid.split(" | ")) # avoid is DNF of propositions

            per_agent_reach = {agent: [] for agent in self.env.possible_agents}
            per_agent_avoid = {agent: [f"gremlins_lidar_{i}"] for i, agent in enumerate(self.env.possible_agents)}
            for reach in reach_set:
                if "_" not in reach: continue
                reach, agent_idx = reach.split("_")
                per_agent_reach[self.env.possible_agents[int(agent_idx)]].append(f"{reach}_zones_lidar_{agent_idx}")
            for avoid in avoid_set:
                if "_" not in avoid: continue
                avoid, agent_idx = avoid.split("_")
                per_agent_avoid[self.env.possible_agents[int(agent_idx)]].append(f"{avoid}_zones_lidar_{agent_idx}")    

            self.selected_subgoal["reach"] = per_agent_reach
            self.selected_subgoal["avoid"] = per_agent_avoid
            
        processed_obs = self.env.pre_process_obs_zones(
            self.selected_subgoal["reach"], self.selected_subgoal["avoid"])
        processed_obs = torch_ac.DictList({
                "features": torch.tensor(np.array(processed_obs), dtype=torch.float), # [num_subgoals, num_agents, obs_dim]
                "epsilon_mask": torch.zeros(len(processed_obs), dtype=torch.bool)})
        with torch.no_grad():
            dist, _, _, _ = self.model(processed_obs, collect=False)
            action = dist.mode if deterministic else dist.sample()
            action = action.detach().numpy()
        actions = {}
        for i, agent in enumerate(self.env.possible_agents):
            if "agent" not in agent: continue # last two agents are req and cnt, which don't have actions
            actions[agent] = action[i, :]

        self.current_goal_steps += 1
        return actions

    def forward(self, obs, deterministic=False) -> np.ndarray:
        
        if self.sequence is not None:
            reach, avoid = self.sequence[0]
            obs["features"] = self.env.pre_process_obs_zones(reach, avoid)
        
        if not (isinstance(obs, list) or isinstance(obs, tuple)):
            obs = [obs]
        preprocessed = preprocessing.preprocess_obss(obs, self.propositions)
        with torch.no_grad():
            dist, _, _ = self.model(preprocessed)
            action = dist.mode if deterministic else dist.sample()
        return action.detach().numpy()
