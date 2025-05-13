from abc import abstractmethod
from typing import Generic, Tuple, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree, PyTreeDef, Real

from ._spaces import Space
from ._types import AgentObservation, TimeStep

ORIGINAL_OBSERVATION_KEY = "_TERMINAL_OBSERVATION"

TObservation = TypeVar("TObservation")
TEnvState = TypeVar("TEnvState")


class Environment(eqx.Module, Generic[TEnvState]):
    """
    Base environment class for JAX-compatible environments. Create your environment by subclassing this.

    `step` and `reset` should typically not be overridden, as they merely handle the
    auto-reset logic. Instead, the environment-specific logic should be implemented in the
    `step_env` and `reset_env` methods.

    """

    def step(
        self,
        key: PRNGKeyArray,
        state: TEnvState,
        action: PyTree[Real[Array, "..."]],
    ) -> Tuple[TimeStep, TEnvState]:
        """
        Steps the environment forward with the given action and performs auto-reset when necessary.
        Additionally, this function inserts the original observation (before auto-resetting) in
        the info dictionary to bootstrap correctly on truncated episodes (`info={"_TERMINAL_OBSERVATION": obs, ...}`)

        This function should typically not be overridden. Instead, the environment-specific logic
        should be implemented in the `step_env` method.

        Returns a TimeStep object (observation, reward, terminated, truncated, info) and the new state.

        **Arguments:**

        - `key`: JAX PRNG key.
        - `state`: Current state of the environment.
        - `action`: Action to take in the environment.
        """

        (obs_step, reward, terminated, truncated, info), state_step = self.step_env(
            key, state, action
        )

        # Auto-reset
        obs_reset, state_reset = self.reset_env(key)
        done = jnp.any(jnp.logical_or(terminated, truncated))
        state = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y), state_reset, state_step
        )
        obs = jax.tree.map(lambda x, y: jax.lax.select(done, x, y), obs_reset, obs_step)

        # Insert the original observation in info to bootstrap correctly
        try:  # remove action mask if present
            obs_step = jax.tree.map(
                lambda o: o.observation,
                obs_step,
                is_leaf=lambda x: isinstance(x, AgentObservation),
            )
        except Exception:
            pass
        info[ORIGINAL_OBSERVATION_KEY] = obs_step

        return TimeStep(obs, reward, terminated, truncated, info), state

    def reset(self, key: PRNGKeyArray) -> Tuple[TObservation, TEnvState]:  # pyright: ignore[reportInvalidTypeVarUse]
        """
        Resets the environment to an initial state and returns the initial observation.
        Environment-specific logic is defined in the `reset_env` method. Typically, this function
        should not be overridden.

        Returns the initial observation and the initial state of the environment.

        **Arguments:**

        - `key`: JAX PRNG key.
        """
        obs, state = self.reset_env(key)
        return obs, state

    @abstractmethod
    def step_env(
        self, key: PRNGKeyArray, state: TEnvState, action: PyTree[Real[Array, "..."]]
    ) -> Tuple[TimeStep, TEnvState]:
        """
        Defines the environment-specific step logic. I.e. here the state of the environment is updated
        according to the transition function.

        Returns a [`TimeStep`](.#timestep) object (observation, reward, terminated, truncated, info) and the new state.

        **Arguments:**

        - `key`: JAX PRNG key.
        - `state`: Current state of the environment.
        - `action`: Action to take in the environment.
        """
        pass

    @abstractmethod
    def reset_env(self, key: PRNGKeyArray) -> Tuple[TObservation, TEnvState]:  # pyright: ignore[reportInvalidTypeVarUse]
        """
        Defines the environment-specific reset logic.

        Returns the initial observation and the initial state of the environment.

        **Arguments:**

        - `key`: JAX PRNG key.
        """
        pass

    @property
    @abstractmethod
    def action_space(self) -> Space | PyTree[Space]:
        """
        Defines the space of valid actions for the environment.
        For multi-agent environments, this should be a PyTree of spaces.
        See [`jymkit.spaces`](Spaces.md) for more information on how to define (composite) action spaces.
        """
        pass

    @property
    @abstractmethod
    def observation_space(self) -> Space | PyTree[Space]:
        """
        Defines the space of possible observations from the environment.
        For multi-agent environments, this should be a PyTree of spaces.
        See [`jymkit.spaces`](Spaces.md) for more information on how to define (composite) observation spaces.
        """
        pass

    @property
    def _multi_agent(self) -> bool:
        """
        Indicates if the environment is a multi-agent environment.
        For multi-agent environments, include a property `multi_agent = True` in the subclass.
        """
        return getattr(self, "multi_agent", False)

    @property
    def agent_structure(self) -> PyTreeDef:
        """
        Returns the structure of the agent space.
        This is useful for environments with multiple agents.
        """
        if not self._multi_agent:
            return jax.tree.structure(0)
        _, agent_structure = eqx.tree_flatten_one_level(self.action_space)
        return agent_structure
