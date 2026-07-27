from typing import Any

from jaxtyping import PRNGKeyArray

from jaxnasium._environment import TEnvState, TimeStep, TObservation
from jaxnasium._spaces import Box, Discrete

from ._wrappers import Wrapper


class NavixWrapper(Wrapper):
    """
    Wrapper for Navix environments to transform them into the Jaxnasium environment interface.

    **Arguments:**

    - `_env`: Navix environment.
    """

    _env: Any

    def reset(self, key: PRNGKeyArray) -> tuple[TObservation, TEnvState]:  # pyright: ignore[reportInvalidTypeVarUse]
        timestep_navix = self._env.reset(key)
        return timestep_navix.observation, timestep_navix

    def step(self, key: PRNGKeyArray, state: Any, action: int) -> tuple[TimeStep, Any]:
        timestep_navix = self._env._step(state, action)
        obs = timestep_navix.observation
        reward = timestep_navix.reward
        terminated = timestep_navix.step_type == 2
        truncated = timestep_navix.step_type == 1
        info = timestep_navix.info
        timestep_step = TimeStep(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        timestep, state = self.auto_reset(key, timestep_step, timestep_navix)
        return timestep, state

    @property
    def observation_space(self) -> Box:
        return Box(
            low=self._env.observation_space.minimum,
            high=self._env.observation_space.maximum,
            shape=self._env.observation_space.shape,
            dtype=self._env.observation_space.dtype,
        )

    @property
    def action_space(self) -> Discrete:
        num_actions = self._env.action_space.maximum
        # Add the "done" no-op action which is outside of the Navix action space (?)
        num_actions += 1
        return Discrete(num_actions)
