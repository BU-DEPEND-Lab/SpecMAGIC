from multiprocessing import Process, Pipe
import gymnasium as gym


def _step_env(env, action):
    obs, reward, done, info = env.step(action)
    if any(done.values()):
        obs, _ = env.reset()

    # agents = ["agent_0", "agent_1", "agent_2"]
    reward = (
        [reward[0][agent] for agent in env.possible_agents],
        [reward[1][agent] for agent in env.possible_agents],
    )
    done = [done[agent] for agent in env.possible_agents]
    info = [dict(info[agent]) for agent in env.possible_agents]
    # reward = (
    #     [reward[0][agent] for agent in agents],
    #     [reward[1][agent] for agent in agents],
    # )
    # done = [done[agent] for agent in agents]
    # info = [dict(info[agent]) for agent in agents]

    return obs, reward, done, info


def worker(conn, env):
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "step":
                conn.send(_step_env(env, data))
            elif cmd == "reset":
                obs, _ = env.reset()
                conn.send(obs)
            elif cmd == "kill":
                return
            else:
                raise NotImplementedError
    finally:
        env.close()
        conn.close()


class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        self.locals = []
        self.processes = []
        self._closed = False
        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            self.processes.append(p)
            remote.close()

    def __del__(self):
        self.close()

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        first_obs, _ = self.envs[0].reset()
        results = [first_obs] + [local.recv() for local in self.locals]
        return results

    def step(self, actions):
        actions = [
            {
                agent: actions[env_index * len(env.possible_agents) + agent_index]
                for agent_index, agent in enumerate(env.possible_agents)
            }
            for env_index, env in enumerate(self.envs)
        ]

        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action))
        results = [_step_env(self.envs[0], actions[0])]
        results.extend(local.recv() for local in self.locals)
        return zip(*results)

    def render(self):
        return self.envs[0].render()

    def close(self):
        if getattr(self, "_closed", True):
            return
        self._closed = True

        locals_ = getattr(self, "locals", ())
        processes = getattr(self, "processes", ())

        for local in locals_:
            try:
                local.send(("kill", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join()
        for local in locals_:
            local.close()

        for env in self.envs:
            env.close()
        self.locals = []
        self.processes = []
