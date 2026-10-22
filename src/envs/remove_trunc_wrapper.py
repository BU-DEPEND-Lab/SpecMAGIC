from typing import Any, SupportsFloat

import gymnasium
from gymnasium.core import WrapperObsType, WrapperActType


class RemoveTruncWrapper(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env):
        super().__init__(env)

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        
        if isinstance(truncated, dict):
            done = {key: terminated[key] or truncated[key] for key in terminated}
        else: 
            done = terminated or truncated  # This is very important
        return obs, reward, done, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> WrapperObsType:
        obs, info = super().reset(seed=seed, options=options)
        return obs, info
