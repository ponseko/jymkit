from dataclasses import replace
from typing import Any, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from ._environment import (
    AgentObservation,
    Environment,
    TEnvState,
    TimeStep,
    TObservation,
)
from ._spaces import Box, Discrete, Space


def is_wrapped(wrapped_env: Environment, wrapper_class: type) -> bool:
    """
    Check if the environment is wrapped with a specific wrapper class.
    """
    current_env = wrapped_env
    while isinstance(current_env, Wrapper):
        if isinstance(current_env, wrapper_class):
            return True
        current_env = current_env._env
    return False


def remove_wrapper(wrapped_env: Environment, wrapper_class: type) -> Environment:
    """
    Remove a specific wrapper class from the environment.
    """
    current_env = wrapped_env
    while isinstance(current_env, Wrapper):
        if isinstance(current_env, wrapper_class):
            return current_env._env
        current_env = current_env._env
    return wrapped_env


class Wrapper(Environment):
    """Base class for all wrappers."""

    _env: Environment

    def reset_env(self, key: PRNGKeyArray) -> Tuple[TObservation, TEnvState]:  # pyright: ignore[reportInvalidTypeVarUse]
        return self._env.reset_env(key)

    def step_env(
        self, key: PRNGKeyArray, state: TEnvState, action: PyTree[int | float | Array]
    ) -> Tuple[TimeStep, TEnvState]:
        return self._env.step_env(key, state, action)

    @property
    def action_space(self) -> Space | PyTree[Space]:
        return self._env.action_space

    @property
    def observation_space(self) -> Space | PyTree[Space]:
        return self._env.observation_space

    @property
    def multi_agent(self) -> bool:
        # if multi_agent is not set, return it as False
        if not hasattr(self._env, "multi_agent"):
            return False
        return self._env.multi_agent

    def __getattr__(self, name):
        return getattr(self._env, name)


class VecEnvWrapper(Wrapper):
    def reset(self, key: PRNGKeyArray) -> Tuple[TObservation, Any]:  # pyright: ignore[reportInvalidTypeVarUse]
        obs, state = jax.vmap(self._env.reset)(key)
        return obs, state

    def step(
        self, key: PRNGKeyArray, state: TEnvState, action: PyTree[int | float | Array]
    ) -> Tuple[TimeStep, TEnvState]:
        timestep, state = jax.vmap(self._env.step)(key, state, action)
        return timestep, state


class LogEnvState(eqx.Module):
    env_state: TEnvState  # pyright: ignore[reportGeneralTypeIssues]
    episode_returns: float | Array
    episode_lengths: int | Array
    returned_episode_returns: float | Array
    returned_episode_lengths: int | Array
    timestep: int | Array = 0


class LogWrapper(Wrapper):
    """
    Log the episode returns and lengths.

    **Arguments:**
    - `_env`: Environment to wrap.
    """

    def reset(self, key: PRNGKeyArray) -> Tuple[TObservation, LogEnvState]:  # pyright: ignore[reportInvalidTypeVarUse]
        obs, env_state = self._env.reset(key)
        structure = self._env.agent_structure
        initial_vals = jnp.zeros(structure.num_leaves).squeeze()
        initial_timestep = 0
        if is_wrapped(self._env, VecEnvWrapper):
            vec_count = jax.tree.leaves(obs)[0].shape[0]
            initial_vals = jnp.zeros((vec_count, structure.num_leaves)).squeeze()
            initial_timestep = jnp.zeros((vec_count,)).squeeze()
        if initial_vals.ndim == 0:
            initial_vals = jnp.expand_dims(initial_vals, axis=0)
        initial_returns = jax.tree.unflatten(structure, initial_vals.T)
        state = LogEnvState(
            env_state=env_state,
            episode_returns=initial_returns,
            episode_lengths=initial_vals,
            returned_episode_returns=initial_returns,
            returned_episode_lengths=initial_vals,
            timestep=initial_timestep,
        )
        return obs, state

    def step(
        self, key: PRNGKeyArray, state: LogEnvState, action: PyTree[int | float | Array]
    ) -> Tuple[TimeStep, LogEnvState]:
        timestep, env_state = self._env.step(key, state.env_state, action)
        done = jnp.logical_or(timestep.terminated, timestep.truncated).any()
        new_episode_return = jax.tree.map(
            lambda _r, r: (_r + r), state.episode_returns, timestep.reward
        )
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=jax.tree.map(
                lambda n_r: n_r * (1 - done), new_episode_return
            ),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=jax.tree.map(
                lambda r, n_r: r * (1 - done) + n_r * done,
                state.returned_episode_returns,
                new_episode_return,
            ),
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
        return jnp.array(jax.tree.leaves(rewards)).squeeze()


class NormalizeVecObsState(eqx.Module):
    env_state: TEnvState  # pyright: ignore[reportGeneralTypeIssues]
    mean: Float[Array, "..."]
    var: Float[Array, "..."]
    count: float


class NormalizeVecObsWrapper(Wrapper):
    def __check_init__(self):
        if not is_wrapped(self._env, VecEnvWrapper):
            raise ValueError(
                "NormalizeVecReward wrapper must wrapped around a `VecEnvWrapper`.\n"
                " Please wrap the environment with `VecEnvWrapper` first."
            )

    def partition_obs_and_masks(self, tree):
        """
        Ensures we do not normalize the action masks if they are present.
        """
        observations = [tree]
        if self._env.multi_agent:
            observations, _ = eqx.tree_flatten_one_level(tree)
        if all(not isinstance(o, AgentObservation) for o in observations):
            filter_spec = True
        elif all(isinstance(o, AgentObservation) for o in observations):
            filter_spec = AgentObservation(observation=True, action_mask=False)
            filter_spec = jax.tree.map(
                lambda _: filter_spec,
                tree,
                is_leaf=lambda x: isinstance(x, AgentObservation),
            )
        else:
            raise ValueError(
                "Observations for all agents must be either AgentObservation or not."
            )
        return eqx.partition(tree, filter_spec=filter_spec)

    def update_state_and_get_obs(self, obs, state: NormalizeVecObsState):
        batch_mean = jax.tree.map(lambda o: jnp.mean(o, axis=0), obs)
        batch_var = jax.tree.map(lambda o: jnp.var(o, axis=0), obs)
        batch_count = jax.tree.leaves(obs)[0].shape[0]

        delta = jax.tree.map(lambda m, b: b - m, batch_mean, state.mean)
        tot_count = state.count + batch_count
        new_mean = jax.tree.map(
            lambda m, d: m + d * batch_count / tot_count,
            state.mean,
            delta,
        )

        m_a = jax.tree.map(lambda v: v * state.count, state.var)
        m_b = jax.tree.map(lambda v: v * batch_count, batch_var)
        M2 = jax.tree.map(
            lambda a, b, d: a
            + b
            + jnp.square(d) * state.count * batch_count / tot_count,
            m_a,
            m_b,
            delta,
        )
        new_var = jax.tree.map(lambda m: m / tot_count, M2)
        new_count = tot_count
        new_state = NormalizeVecObsState(
            env_state=state.env_state, mean=new_mean, var=new_var, count=new_count
        )

        normalized_obs = jax.tree.map(
            lambda o, m, v: (o - m) / jnp.sqrt(v + 1e-8), obs, new_mean, new_var
        )
        return normalized_obs, new_state

    def reset(self, key: PRNGKeyArray) -> Tuple[TObservation, NormalizeVecObsState]:  # pyright: ignore[reportInvalidTypeVarUse]
        obs, env_state = self._env.reset(key)
        obs, masks = self.partition_obs_and_masks(obs)
        state = NormalizeVecObsState(
            env_state=env_state,
            mean=jax.tree.map(jnp.zeros_like, obs),
            var=jax.tree.map(jnp.ones_like, obs),
            count=1e-4,
        )
        normalized_obs, state = self.update_state_and_get_obs(obs, state)
        normalized_obs = eqx.combine(normalized_obs, masks)
        return normalized_obs, state

    def step(
        self,
        key: PRNGKeyArray,
        state: NormalizeVecObsState,
        action: PyTree[int | float | Array],
    ) -> Tuple[TimeStep, NormalizeVecObsState]:
        timestep, env_state = self._env.step(key, state.env_state, action)
        obs = timestep.observation
        obs, masks = self.partition_obs_and_masks(obs)
        state = replace(state, env_state=env_state)
        normalized_obs, state = self.update_state_and_get_obs(obs, state)
        normalized_obs = eqx.combine(normalized_obs, masks)
        return timestep._replace(observation=normalized_obs), state


class NormalizeVecRewState(eqx.Module):
    env_state: TEnvState  # pyright: ignore[reportGeneralTypeIssues]
    mean: Float[Array, "..."]
    var: Float[Array, "..."]
    count: float
    return_val: Float[Array, "..."]


class NormalizeVecRewardWrapper(Wrapper):
    gamma: float = 0.99

    def __check_init__(self):
        if not is_wrapped(self._env, VecEnvWrapper):
            raise ValueError(
                "NormalizeVecReward wrapper must wrapped around a `VecEnvWrapper`.\n"
                " Please wrap the environment with `VecEnvWrapper` first."
            )

    def reset(self, key: PRNGKeyArray) -> Tuple[TObservation, NormalizeVecRewState]:  # pyright: ignore[reportInvalidTypeVarUse]
        obs, env_state = self._env.reset(key)
        batch_count = jax.tree.leaves(obs)[0].shape[0]
        num_agents = self._env.agent_structure.num_leaves
        state = NormalizeVecRewState(
            env_state=env_state,
            mean=jnp.zeros(num_agents).squeeze(),
            var=jnp.ones(num_agents).squeeze(),
            count=1e-4,
            return_val=jnp.zeros((num_agents, batch_count)).squeeze(),
        )

        return obs, state

    def step(
        self,
        key: PRNGKeyArray,
        state: NormalizeVecRewState,
        action: PyTree[int | float | Array],
    ) -> Tuple[TimeStep, NormalizeVecRewState]:
        (obs, reward, terminated, truncated, info), env_state = self._env.step(
            key, state.env_state, action
        )

        # get the rewards as a single matrix -- reconstruct later
        reward, reward_structure = jax.tree.flatten(reward)
        reward = jnp.array(reward).squeeze()
        done = jnp.logical_or(terminated, truncated)  # TODO ?
        return_val = state.return_val * self.gamma * (1 - done) + reward

        batch_mean = jnp.mean(return_val, axis=-1)
        batch_var = jnp.var(return_val, axis=-1)
        batch_count = jax.tree.leaves(obs)[0].shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecRewState(
            env_state=env_state,
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
        )

        if np.any(self._env.multi_agent):  # type: ignore[reportGeneralTypeIssues]
            reward = reward / jnp.sqrt(jnp.expand_dims(state.var, axis=-1) + 1e-8)
            reward = jax.tree.unflatten(reward_structure, reward)
        else:
            reward = reward / jnp.sqrt(state.var + 1e-8)

        return TimeStep(obs, reward, terminated, truncated, info), state


class GymnaxWrapper(Wrapper):
    """
    Wrapper for Gymnax environments.
    Since Gymnax does not expose truncated information, we can optionally
    retrieve it by taking an additional step in the environment with altered timestep
    information. Since this introduces additional overhead, it is disabled by default.

    **Arguments:**
    - `_env`: Gymnax environment.
    """

    _env: Any

    def reset(self, key: PRNGKeyArray) -> Tuple[TObservation, TEnvState]:  # pyright: ignore[reportInvalidTypeVarUse]
        params = self._env.default_params
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

    def _convert_gymnax_space_to_jymkit_space(self, space: Any) -> Space:
        # Convert Gymnax space to jymkit space
        if hasattr(space, "n"):
            return Discrete(space.n)
        elif hasattr(space, "low") and hasattr(space, "high"):
            return Box(
                low=space.low,
                high=space.high,
                shape=space.shape,
            )
        else:
            raise ValueError(
                "Gymnax action space is not supported. Please use a discrete or box action space."
            )

    @property
    def observation_space(self) -> Space:
        params = self._env.default_params
        observation_space = self._env.observation_space(params)
        return self._convert_gymnax_space_to_jymkit_space(observation_space)

    @property
    def action_space(self) -> Space:
        params = self._env.default_params
        action_space = self._env.action_space(params)
        return self._convert_gymnax_space_to_jymkit_space(action_space)
