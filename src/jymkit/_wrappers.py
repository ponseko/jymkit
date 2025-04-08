from dataclasses import replace
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree

import jymkit as jym


class Wrapper(eqx.Module):
    """Base class for all wrappers."""

    env: jym.Environment

    def __getattr__(self, name):
        return getattr(self.env, name)


class LogEnvState(eqx.Module):
    env_state: jym.EnvState
    episode_returns: float | Array
    episode_lengths: int | Array
    returned_episode_returns: float | Array
    returned_episode_lengths: int | Array
    timestep: int = 0


class LogWrapper(Wrapper):
    """
    Log the episode returns and lengths.

    **Arguments:**
    - `env`: Environment to wrap.
    """

    def reset(self, key: PRNGKeyArray) -> Tuple[jym.Observation, LogEnvState]:
        obs, env_state = self.env.reset(key)
        structure = jax.tree.structure(
            obs, is_leaf=lambda x: isinstance(x, jym.AgentObservation)
        )
        initial_vals = jnp.zeros((structure.num_leaves,))
        state = LogEnvState(
            env_state=env_state,
            episode_returns=initial_vals,
            episode_lengths=initial_vals,
            returned_episode_returns=initial_vals,
            returned_episode_lengths=initial_vals,
            timestep=0,
        )
        return obs, state

    def step(
        self, key: PRNGKeyArray, state: LogEnvState, action: jym.Action
    ) -> Tuple[jym.TimeStep, LogEnvState]:
        timestep, env_state = self.env.step(key, state.env_state, action)
        reward = self._flat_reward(timestep.reward)
        done = jnp.logical_or(timestep.terminated, timestep.truncated)
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info = timestep.info
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return timestep._replace(info=info), state

    def _flat_reward(self, rewards: float | PyTree[float]):
        return jnp.array(jax.tree.leaves(rewards))


class GymnaxWrapper(Wrapper):
    """
    Wrapper for Gymnax environments.
    Since Gymnax does not expose truncated information, we can optionally
    retrieve it by taking an additional step in the environment with altered timestep
    information. Since this introduces additional overhead, it is disabled by default.

    **Arguments:**
    - `env`: Gymnax environment.
    - `retrieve_truncated_info`: If True, retrieves truncated information by taking an additional step.
    """

    retrieve_truncated_info: bool = False

    def step(
        self, key: PRNGKeyArray, state: jym.EnvState, action: jym.Action
    ) -> Tuple[jym.TimeStep, jym.Environment]:
        obs, env_state, done, reward, info = self.env.step(key, state, action)
        terminated, truncated = done, False
        if self.retrieve_truncated_info:
            # Retrieve truncated info by taking an additional step
            try:
                back_in_time_env_state = replace(state, time=0)
                _, _, done_alt, _, _ = self.env.step(
                    key, back_in_time_env_state, action
                )
                # terminated if done is True and done_alt is False
                terminated = jnp.logical_and(done, ~done_alt)
                truncated = jnp.logical_and(done, ~terminated)
            except Exception as e:
                print(
                    "retrieve_truncated_info is enabled, but retrieving truncated info failed."
                )
                raise e

        timestep = jym.TimeStep(
            obs=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        return timestep, env_state
