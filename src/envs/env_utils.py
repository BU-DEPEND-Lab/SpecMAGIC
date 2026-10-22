from typing import Callable, Optional

import gymnasium
# from gymnasium.wrappers import FlattenObservation, TimeLimit

from envs.remove_trunc_wrapper import RemoveTruncWrapper
from copy import deepcopy
import gymnasium as gym


# adapted from gymnasium.wrappers.time_limit.TimeLimit
class TimeLimit(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """This wrapper will issue a `truncated` signal if a maximum number of timesteps is exceeded.

    If a truncation is not defined inside the environment itself, this is the only place that the truncation signal is issued.
    Critically, this is different from the `terminated` signal that originates from the underlying environment as part of the MDP.

    Example:
       >>> import gymnasium as gym
       >>> from gymnasium.wrappers import TimeLimit
       >>> env = gym.make("CartPole-v1")
       >>> env = TimeLimit(env, max_episode_steps=1000)
    """

    def __init__(
        self,
        env: gym.Env,
        max_episode_steps: int,
    ):
        """Initializes the :class:`TimeLimit` wrapper with an environment and the number of steps after which truncation will occur.

        Args:
            env: The environment to apply the wrapper
            max_episode_steps: An optional max episode steps (if ``None``, ``env.spec.max_episode_steps`` is used)
        """
        gym.utils.RecordConstructorArgs.__init__(
            self, max_episode_steps=max_episode_steps
        )
        gym.Wrapper.__init__(self, env)

        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        """Steps through the environment and if the number of steps elapsed exceeds ``max_episode_steps`` then truncate.

        Args:
            action: The environment step action

        Returns:
            The environment step ``(observation, reward, terminated, truncated, info)`` with `truncated=True`
            if the number of steps elapsed >= max episode steps

        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1

        if self._elapsed_steps >= self._max_episode_steps:
            if isinstance(truncated, dict):
                for key in truncated:
                    truncated[key] = True
            else:
                truncated = True

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Resets the environment with :param:`**kwargs` and sets the number of steps elapsed to zero.

        Args:
            **kwargs: The kwargs to reset the environment with

        Returns:
            The reset environment
        """
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

    @property
    def spec(self): # -> EnvSpec | None:
        """Modifies the environment spec to include the `max_episode_steps=self._max_episode_steps`."""
        if self._cached_spec is not None:
            return self._cached_spec

        env_spec = self.env.spec
        if env_spec is not None:
            env_spec = deepcopy(env_spec)
            env_spec.max_episode_steps = self._max_episode_steps

        self._cached_spec = env_spec
        return env_spec


def get_env_attr(env, attr: str):
    if hasattr(env, attr):
        return getattr(env, attr)
    if hasattr(env, 'env'):
        return getattr(env.unwrapped, attr)
    else:
        raise AttributeError(f'Attribute {attr} not found in env.')


def make_env_safety(
        name: str,
        sampler: Callable[[list[str]], Callable],
        max_steps: Optional[int] = None,
        render_mode: str | None = None,
        sequence=False,
):
    from envs.pretraining.pretraining_env import PretrainingEnv
    from envs.seq_wrapper import SequenceSafetyWrapper
    from envs.ldba_wrapper import MALDBAWrapperRRT
    from envs.ltl_wrapper import LTLWrapper

    if is_safety_gym_env(name):
        env = make_safety_gym_env(name, render_mode)
        max_steps = max_steps or 2000
    elif name.startswith('Letter'):
        env = make_letter_env(name, render_mode)
        max_steps = max_steps or 75
    elif name.startswith('FlatWorld'):
        env = make_flatworld_env(name)
        max_steps = max_steps or 500
    else:
        raise ValueError(f'Unknown environment: {name}')

    propositions = get_env_attr(env, 'get_propositions')()
    sample_task = sampler(propositions)
    if not sequence:
        env = LTLWrapper(env, sample_task)
        env = MALDBAWrapperRRT(env)
    else:
        env = SequenceSafetyWrapper(env, sample_task)
    print(f"DEBUG: max_steps = {max_steps}")
    env = TimeLimit(env, max_episode_steps=max_steps)
    env = RemoveTruncWrapper(env)
    return env


def is_safety_gym_env(name: str) -> bool:
    return any([name.startswith(agent_name) for agent_name in ['Point', 'Car', 'Racecar', 'Doggo', 'Ant', 'Safety']])


def make_safety_gym_env(name: str, render_mode: str | None = None):
    # noinspection PyUnresolvedReferences
    import safety_gymnasium
    from envs.zones.safety_gym_wrapper import SafetyGymWrapper, MASafetyGymWrapper

    env = safety_gymnasium.make(name, render_mode=render_mode)
    # env = SafetyGymWrapper(env)
    env = MASafetyGymWrapper(env)
    # env = FlattenObservation(env)
    return env


def make_letter_env(name: str, render_mode: str | None = None):
    import envs.letter_world

    env = gymnasium.make(name, render_mode=render_mode)
    return env


def make_flatworld_env(name: str):
    import envs.flatworld

    env = gymnasium.make(name)
    return env
