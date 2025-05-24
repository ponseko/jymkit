from typing import Any, Tuple

import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from ._environment import (
    ORIGINAL_OBSERVATION_KEY,
    AgentObservation,
    TEnvState,
    TimeStep,
    TObservation,
)
from ._spaces import Space
from ._wrappers import Wrapper


class GymnaxWrapper(Wrapper):
    """
    Wrapper for Gymnax environments to transform them into the Jymkit environment interface.

    **Arguments:**

    - `_env`: Gymnax environment.
    """

    _env: Any

    def reset(self, key: PRNGKeyArray) -> Tuple[TObservation, TEnvState]:  # pyright: ignore[reportInvalidTypeVarUse]
        params = getattr(self._env, "default_params", None)
        obs, env_state = self._env.reset(key, params)
        return obs, env_state

    def step(
        self, key: PRNGKeyArray, state: TEnvState, action: int | float
    ) -> Tuple[TimeStep, TEnvState]:
        obs, env_state, reward, done, info = self._env.step(key, state, action)
        terminated, truncated = done, False
        timestep = TimeStep(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        return timestep, env_state

    @property
    def observation_space(self) -> Space:
        params = self._env.default_params
        return self._env.observation_space(params)

    @property
    def action_space(self) -> Space:
        params = self._env.default_params
        return self._env.action_space(params)


class JumanjiWrapper(Wrapper):
    """
    Wrapper for Jumanji environments to transform them into the Jymkit environment interface.

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

    @property
    def observation_space(self) -> Any:
        from gymnasium.spaces import Dict as GymnasiumDict
        from jumanji.specs import jumanji_specs_to_gym_spaces

        space = self._env.observation_spec
        space = jumanji_specs_to_gym_spaces(space)
        if isinstance(space, GymnasiumDict):
            # exclude the action mask
            return {k: v for k, v in space.spaces.items() if k != "action_mask"}
        return space

    @property
    def action_space(self) -> Any:
        from gymnasium.spaces import Dict as GymnasiumDict
        from jumanji.specs import jumanji_specs_to_gym_spaces

        space = self._env.action_spec
        space = jumanji_specs_to_gym_spaces(space)
        if isinstance(space, GymnasiumDict):
            return {k: v for k, v in space.spaces.items()}
        return space


class BraxWrapper(Wrapper):
    """
    Wrapper for Brax environments to transform them into the Jymkit environment interface.

    **Arguments:**

    - `_env`: Brax environment.
    """

    _env: Any

    def reset(self, key: PRNGKeyArray) -> Tuple[TObservation, TEnvState]:  # pyright: ignore[reportInvalidTypeVarUse]
        env_state = self._env.reset(key)
        return env_state.obs, env_state

    def step(
        self, key: PRNGKeyArray, state: TEnvState, action: int | float
    ) -> Tuple[TimeStep, TEnvState]:
        env_state = self._env.step(state, action)
        timestep = TimeStep(
            observation=env_state.obs,
            reward=env_state.reward,
            terminated=env_state.done,
            truncated=False,
            info={},
        )
        return timestep, env_state

    @property
    def observation_space(self) -> Any:
        from brax.envs.wrappers import gym as braxGym

        return braxGym.GymWrapper(self._env).observation_space

    @property
    def action_space(self) -> Any:
        from brax.envs.wrappers import gym as braxGym

        return braxGym.GymWrapper(self._env).action_space
