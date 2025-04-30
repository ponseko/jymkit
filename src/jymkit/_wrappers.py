from dataclasses import replace
from typing import Any, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray, PyTree, PyTreeDef

import jymkit as jym

from ._environment import AbstractEnvironment, TObservation


def is_wrapped(env: AbstractEnvironment, wrapper_class: type) -> bool:
    """
    Check if the environment is wrapped with a specific wrapper class.
    """
    current_env = env
    while isinstance(current_env, Wrapper):
        if isinstance(current_env, wrapper_class):
            return True
        current_env = current_env.env
    return False


class Wrapper(AbstractEnvironment):
    """Base class for all wrappers."""

    env: AbstractEnvironment

    @property
    def action_space(self) -> jym.Space | PyTree[jym.Space]:
        return self.env.action_space

    @property
    def observation_space(self) -> jym.Space | PyTree[jym.Space]:
        return self.env.observation_space

    @property
    def agent_structure(self) -> PyTreeDef:
        return self.env.agent_structure

    @property
    def multi_agent(self) -> bool:
        return self.env.multi_agent

    def __getattr__(self, name):
        return getattr(self.env, name)


class VecEnvWrapper(Wrapper):
    def reset(self, key: PRNGKeyArray) -> Tuple[TObservation, Any]:  # pyright: ignore[reportInvalidTypeVarUse]
        obs, env = jax.vmap(self.env.reset)(key)
        return obs, replace(self, env=env)

    def step(
        self, key: PRNGKeyArray, action: PyTree[int | float | Array]
    ) -> Tuple[jym.TimeStep, Any]:
        timestep, env = jax.vmap(lambda env, key, action: env.step(key, action))(
            self.env, key, action
        )
        return timestep, replace(self, env=env)


class LogEnvState(eqx.Module):
    episode_returns: float | Array
    episode_lengths: int | Array
    returned_episode_returns: float | Array
    returned_episode_lengths: int | Array
    timestep: int | Array = 0


class LogWrapper(Wrapper):
    """
    Log the episode returns and lengths.

    **Arguments:**
    - `env`: Environment to wrap.
    """

    state: LogEnvState | None = None

    def reset(self, key: PRNGKeyArray) -> Tuple[TObservation, "LogWrapper"]:  # pyright: ignore[reportInvalidTypeVarUse]
        obs, env = self.env.reset(key)
        structure = self.env.agent_structure
        initial_vals = jnp.zeros(structure.num_leaves).squeeze()
        initial_timestep = 0
        if is_wrapped(self.env, VecEnvWrapper):
            vec_count = jax.tree.leaves(obs)[0].shape[0]
            initial_vals = jnp.zeros((vec_count, structure.num_leaves)).squeeze()
            initial_timestep = jnp.zeros((vec_count,)).squeeze()
        state = LogEnvState(
            episode_returns=initial_vals,
            episode_lengths=initial_vals,
            returned_episode_returns=initial_vals,
            returned_episode_lengths=initial_vals,
            timestep=initial_timestep,
        )
        return obs, replace(self, env=env, state=state)

    def step(
        self, key: PRNGKeyArray, action: PyTree[int | float | Array]
    ) -> Tuple[jym.TimeStep, "LogWrapper"]:
        assert self.state is not None, "Environment must be reset before stepping."
        timestep, env = self.env.step(key, action)
        reward = self._flat_reward(timestep.reward)
        done = jnp.logical_or(timestep.terminated, timestep.truncated)
        new_episode_return = self.state.episode_returns + reward
        new_episode_length = self.state.episode_lengths + 1
        state = LogEnvState(
            # env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=self.state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=self.state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=self.state.timestep + 1,
        )
        info = timestep.info
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return timestep._replace(info=info), replace(self, env=env, state=state)

    def _flat_reward(self, rewards: float | PyTree[float]):
        return jnp.array(jax.tree.leaves(rewards)).squeeze()


class NormalizeVecObsState(eqx.Module):
    mean: Float[Array, "..."]
    var: Float[Array, "..."]
    count: float


class NormalizeVecObsWrapper(Wrapper):
    state: NormalizeVecObsState | None = None

    def __check_init__(self):
        if not is_wrapped(self.env, VecEnvWrapper):
            raise ValueError(
                "NormalizeVecReward wrapper must wrapped around a `VecEnvWrapper`.\n"
                " Please wrap the environment with `VecEnvWrapper` first."
            )

    def partition_obs_and_masks(self, tree):
        """
        Ensures we do not normalize the action masks if they are present.
        """
        observations = [tree]
        if self.env.multi_agent:
            observations, _ = eqx.tree_flatten_one_level(tree)
        if all(not isinstance(o, jym.AgentObservation) for o in observations):
            filter_spec = True
        elif all(isinstance(o, jym.AgentObservation) for o in observations):
            filter_spec = jym.AgentObservation(observation=True, action_mask=False)
            filter_spec = jax.tree.map(
                lambda _: filter_spec,
                tree,
                is_leaf=lambda x: isinstance(x, jym.AgentObservation),
            )
        else:
            raise ValueError(
                "Observations for all agents must be either AgentObservation or not."
            )
        return eqx.partition(tree, filter_spec=filter_spec)

    def update_state_and_get_obs(self, obs):
        obs, masks = self.partition_obs_and_masks(obs)
        state = self.state
        if state is None:
            # initial init
            state = NormalizeVecObsState(
                mean=jax.tree.map(jnp.zeros_like, obs),
                var=jax.tree.map(jnp.ones_like, obs),
                count=1e-4,
            )
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
        new_state = NormalizeVecObsState(mean=new_mean, var=new_var, count=new_count)

        normalized_obs = jax.tree.map(
            lambda o, m, v: (o - m) / jnp.sqrt(v + 1e-8), obs, new_mean, new_var
        )
        normalized_obs = eqx.combine(normalized_obs, masks)
        return normalized_obs, new_state

    def reset(self, key: PRNGKeyArray) -> Tuple[TObservation, "NormalizeVecObsWrapper"]:  # pyright: ignore[reportInvalidTypeVarUse]
        obs, env = self.env.reset(key)
        obs, state = self.update_state_and_get_obs(obs)
        return obs, replace(self, env=env, state=state)

    def step(
        self, key: PRNGKeyArray, action: PyTree[int | float | Array]
    ) -> Tuple[jym.TimeStep, "NormalizeVecObsWrapper"]:
        assert self.state is not None, "Environment must be reset before stepping."
        timestep, env = self.env.step(key, action)
        obs = timestep.observation
        obs, state = self.update_state_and_get_obs(obs)
        return timestep._replace(observation=obs), replace(self, env=env, state=state)


class NormalizeVecRewState(eqx.Module):
    mean: Float[Array, "..."]
    var: Float[Array, "..."]
    count: float
    return_val: Float[Array, "..."]


class NormalizeVecRewardWrapper(Wrapper):
    gamma: float = 0.99
    state: NormalizeVecRewState | None = None

    def __check_init__(self):
        if not is_wrapped(self.env, VecEnvWrapper):
            raise ValueError(
                "NormalizeVecReward wrapper must wrapped around a `VecEnvWrapper`.\n"
                " Please wrap the environment with `VecEnvWrapper` first."
            )

    def first_init(self, obs) -> NormalizeVecRewState:
        batch_count = jax.tree.leaves(obs)[0].shape[0]
        num_agents = self.env.agent_structure.num_leaves
        return NormalizeVecRewState(
            mean=jnp.zeros(num_agents).squeeze(),
            var=jnp.ones(num_agents).squeeze(),
            count=1e-4,
            return_val=jnp.zeros((num_agents, batch_count)).squeeze(),
        )

    def reset(
        self, key: PRNGKeyArray
    ) -> Tuple[TObservation, "NormalizeVecRewardWrapper"]:  # pyright: ignore[reportInvalidTypeVarUse]
        obs, env = self.env.reset(key)
        if self.state is None:
            state = self.first_init(obs)
            return obs, replace(self, env=env, state=state)
        return obs, replace(self, env=env)

    def step(
        self,
        key: PRNGKeyArray,
        action: PyTree[int | float | Array],
    ) -> Tuple[jym.TimeStep, "NormalizeVecRewardWrapper"]:
        assert self.state is not None, "Environment must be reset before stepping."
        (obs, reward, terminated, truncated, info), env = self.env.step(key, action)
        state = self.state

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
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
        )

        if np.any(self.env.multi_agent):  # type: ignore[reportGeneralTypeIssues]
            reward = reward / jnp.sqrt(jnp.expand_dims(state.var, axis=-1) + 1e-8)
            reward = jax.tree.unflatten(reward_structure, reward)
        else:
            reward = reward / jnp.sqrt(state.var + 1e-8)

        return jym.TimeStep(
            obs,
            reward,
            terminated,
            truncated,
            info,
        ), replace(self, env=env, state=state)


# class GymnaxWrapper(Wrapper):
#     """
#     Wrapper for Gymnax environments.
#     Since Gymnax does not expose truncated information, we can optionally
#     retrieve it by taking an additional step in the environment with altered timestep
#     information. Since this introduces additional overhead, it is disabled by default.

#     **Arguments:**
#     - `env`: Gymnax environment.
#     - `retrieve_truncated_info`: If True, retrieves truncated information by taking an additional step.
#     """

#     env: Any
#     retrieve_truncated_info: bool = False

#     def step(
#         self, key: PRNGKeyArray, state: Any, action: int | float
#     ) -> Tuple[jym.TimeStep, "GymnaxWrapper"]:
#         obs, env_state, done, reward, info = self.env.step(key, state, action)
#         terminated, truncated = done, False
#         if self.retrieve_truncated_info:
#             # Retrieve truncated info by taking an additional step
#             try:
#                 back_in_time_env_state = replace(state, time=0)
#                 _, _, done_alt, _, _ = self.env.step(
#                     key, back_in_time_env_state, action
#                 )
#                 # terminated if done is True and done_alt is False
#                 terminated = jnp.logical_and(done, ~done_alt)
#                 truncated = jnp.logical_and(done, ~terminated)
#             except Exception as e:
#                 print(
#                     "retrieve_truncated_info is enabled, but retrieving truncated info failed."
#                 )
#                 raise e

#         timestep = jym.TimeStep(
#             observation=obs,
#             reward=reward,
#             terminated=terminated,
#             truncated=truncated,
#             info=info,
#         )
#         return timestep, env_state
