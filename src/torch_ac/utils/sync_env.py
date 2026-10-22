from multiprocessing import Process, Pipe
import gymnasium as gym


class SyncEnv(gym.Env):
    """A sequential execution of multiple environments."""

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self):
        results = [env.reset()[0] for env in self.envs]
        return results

    def step(self, actions):
        results = []
        for i, env in enumerate(self.envs):
            # the actions is flattened list of [env1_agent1, env1_agent2, ..., env2_agent1, env2_agent2, ...]
            action = {agent: actions[i*len(env.possible_agents) + j] for j, agent in enumerate(env.possible_agents)}
            
            obs, reward, done, info = env.step(action)
            if any(done.values()):
                obs, _ = env.reset()

            # convert reward, done, info from dicts to lists
            reward = ([reward[0][agent] for agent in env.possible_agents], [reward[1][agent] for agent in env.possible_agents])
            done = [done[agent] for agent in env.possible_agents]
            # info_list = []
            # for agent in env.possible_agents:
            #     if "success" in info[agent]:
            #         info_list.append("success")
            #     elif "violation" in info[agent]:
            #         info_list.append("violation")
            #     else:
            #         info_list.append("none")
            
            # Preserve the per-agent diagnostic fields. Callers can still use
            # expressions such as ``"success" in info`` with these mappings.
            info_list = [dict(info[agent]) for agent in env.possible_agents]

            results.append((obs, reward, done, info_list))
        results = zip(*results)
        return results

    def render(self):
        raise NotImplementedError
