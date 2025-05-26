from typing import Any, Tuple

import equinox as eqx
import jax
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
    - `handle_truncation`: If True, the wrapper will reimplement the autoreset behavior to include
        truncated information and the terminal_observation in the info dictionary. If False, the wrapper will mirror
        the Gymnax behavior by ignoring truncations. Default=True.
    """

    _env: Any
    handle_truncation: bool = True

    def reset(self, key: PRNGKeyArray) -> Tuple[TObservation, TEnvState]:  # pyright: ignore[reportInvalidTypeVarUse]
        params = getattr(self._env, "default_params", None)
        obs, env_state = self._env.reset(key, params)
        return obs, env_state

    def step(
        self, key: PRNGKeyArray, state: Any, action: int | float
    ) -> Tuple[TimeStep, Any]:
        if not self.handle_truncation:
            obs, env_state, reward, done, info = self._env.step(key, state, action)
            terminated, truncated = done, False
        else:
            # We increase max_steps_in_episode by 1 so that the done flag from Gymnax
            # only triggers when the episode terminates without truncation.
            # Then we set truncate manually in this wrapper based on the original max_steps_in_episode.
            _params = getattr(self._env, "default_params")  # is dataclass
            original_max_steps = _params.max_steps_in_episode
            altered_params = eqx.tree_at(
                lambda x: x.max_steps_in_episode, _params, replace_fn=lambda x: x + 1
            )
            obs_step, state_step, reward, done, info = self._env.step_env(
                key, state, action, altered_params
            )
            terminated = done  # did not truncate due to the +1 in max_steps_in_episode
            truncated = state_step.time >= original_max_steps

            # Auto-reset manually since we used gymnax_env.step_env(...) instead of gymnax_env.step(...)
            done = jnp.logical_or(terminated, truncated)
            obs_reset, state_reset = self.reset(key)
            env_state = jax.tree.map(
                lambda x, y: jax.lax.select(done, x, y), state_reset, state_step
            )
            obs = jax.tree.map(
                lambda x, y: jax.lax.select(done, x, y), obs_reset, obs_step
            )
            # Insert the original observation in info to bootstrap correctly
            info[ORIGINAL_OBSERVATION_KEY] = obs_step

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
        return self.__convert_gymnasium_space_to_dict(space)

    @property
    def action_space(self) -> Any:
        from jumanji.specs import jumanji_specs_to_gym_spaces

        space = self._env.action_spec
        space = jumanji_specs_to_gym_spaces(space)
        return self.__convert_gymnasium_space_to_dict(space)


class BraxWrapperState(eqx.Module):
    brax_env_state: Any  # The state of the Brax environment
    timestep: int = 0


class BraxWrapper(Wrapper):
    """
    Wrapper for Brax environments to transform them into the Jymkit environment interface.

    Note: Brax environments would typically be wrapped with a VmapWrapper, EpisodeWrapper and AutoResetWrapper
    VmapWrapper is not included here, as it is replaced by the Jymkit's `VecEnvWrapper`.
    The effects of EpisodeWrapper (truncation) and AutoResetWrapper are merged into this wrapper.

    **Arguments:**

    - `_env`: Brax environment.
    """

    _env: Any
    max_episode_steps: int = 1000  # Brax defaults to 1000

    def reset(self, key: PRNGKeyArray) -> Tuple[TObservation, BraxWrapperState]:  # pyright: ignore[reportInvalidTypeVarUse]
        env_state = self._env.reset(key)
        env_state = BraxWrapperState(brax_env_state=env_state, timestep=0)
        return env_state.brax_env_state.obs, env_state

    def step(
        self, key: PRNGKeyArray, state: BraxWrapperState, action: int | float
    ) -> Tuple[TimeStep, BraxWrapperState]:
        brax_env_state = self._env.step(state.brax_env_state, action)
        state_step = BraxWrapperState(
            brax_env_state=brax_env_state,
            timestep=state.timestep + 1,
        )
        truncated = state_step.timestep >= self.max_episode_steps
        terminated = brax_env_state.done
        info = brax_env_state.info

        # Auto-reset
        done = jnp.logical_or(terminated, truncated)
        obs_reset, state_reset = self.reset(key)
        env_state = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y), state_reset, state_step
        )
        obs = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y), obs_reset, brax_env_state.obs
        )
        # Insert the original observation in info to bootstrap correctly
        info[ORIGINAL_OBSERVATION_KEY] = brax_env_state.obs

        timestep = TimeStep(
            observation=obs,
            reward=brax_env_state.reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
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
