from typing import Any, Tuple

import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from jaxnasium._environment import (
    ORIGINAL_OBSERVATION_KEY,
    AgentObservation,
    TEnvState,
    TimeStep,
    TObservation,
)

from ._util import gymnasium_to_jaxnasium_space
from ._wrappers import Wrapper


class JumanjiWrapper(Wrapper):
    """
    Wrapper for Jumanji environments to transform them into the Jaxnasium environment interface.

    **Arguments:**

    - `_env`: Jumanji environment.
    """

    _env: Any

    def __init__(self, env: Any):
        from jumanji.wrappers import AutoResetWrapper

        self._env = AutoResetWrapper(env, next_obs_in_extras=True)

    def _convert_jumanji_obs(self, obs: Any) -> TObservation:  # pyright: ignore[reportInvalidTypeVarUse]
        if isinstance(obs, tuple) and hasattr(obs, "_asdict"):  # NamedTuple
            # Convert it to a dict and collect the action mask
            action_mask = getattr(obs, "action_mask", None)
            obs = {
                key: value
                for key, value in obs._asdict().items()  # pyright: ignore[reportAttributeAccessIssue]
                if key != "action_mask"
            }
            obs = AgentObservation(observation=obs, action_mask=action_mask)
        return obs  # type: ignore[reportGeneralTypeIssues]

    def reset(self, key: PRNGKeyArray) -> Tuple[TObservation, TEnvState]:  # pyright: ignore[reportInvalidTypeVarUse]
        state, timestep = self._env.reset(key)
        observation = self._convert_jumanji_obs(timestep.observation)
        return observation, state

    def step(
        self, key: PRNGKeyArray, state: TEnvState, action: int | float
    ) -> Tuple[TimeStep, TEnvState]:
        state, timestep = self._env.step(state, action)  # No key for Jumanji
        obs = self._convert_jumanji_obs(timestep.observation)

        truncated = jnp.logical_and(timestep.discount != 0, timestep.step_type == 2)
        terminated = jnp.logical_and(timestep.step_type == 2, ~truncated)

        info = timestep.extras
        info["DISCOUNT"] = timestep.discount
        next_obs = info.pop("next_obs", None)
        info[ORIGINAL_OBSERVATION_KEY] = self._convert_jumanji_obs(next_obs)

        timestep = TimeStep(
            observation=obs,
            reward=timestep.reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        return timestep, state

    def __convert_gymnasium_space_to_dict(self, space: Any) -> Any:
        """Recursively convert Gymnasium Dict spaces to regular dicts."""
        from gymnasium.spaces import Dict as GymnasiumDict

        if isinstance(space, GymnasiumDict):
            # Recursively convert nested spaces and exclude action_mask
            return {
                k: self.__convert_gymnasium_space_to_dict(v)
                for k, v in space.spaces.items()
                if k != "action_mask"
            }
        return space

    @property
    def observation_space(self) -> Any:
        from jumanji.specs import jumanji_specs_to_gym_spaces

        space = self._env.observation_spec
        space = jumanji_specs_to_gym_spaces(space)
        space = self.__convert_gymnasium_space_to_dict(space)
        return gymnasium_to_jaxnasium_space(space)
        return self.__convert_gymnasium_space_to_dict(space)

    @property
    def action_space(self) -> Any:
        from jumanji.specs import jumanji_specs_to_gym_spaces

        space = self._env.action_spec
        space = jumanji_specs_to_gym_spaces(space)
        space = self.__convert_gymnasium_space_to_dict(space)
        return gymnasium_to_jaxnasium_space(space)
        return self.__convert_gymnasium_space_to_dict(space)
