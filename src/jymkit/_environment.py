from abc import abstractmethod
from typing import Generic, Tuple, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree, PyTreeDef

from ._spaces import Space
from ._types import AgentObservation, TimeStep

ORIGINAL_OBSERVATION_KEY = "_TERMINAL_OBSERVATION"

TObservation = TypeVar("TObservation")
TEnvState = TypeVar("TEnvState")


class Environment(eqx.Module, Generic[TEnvState]):
    """
    Abstract environment template for reinforcement learning environments in JAX.

    Provides a standardized interface for RL environments with JAX compatibility.
    Subclasses must implement the abstract methods to define specific environment behaviors.

    **Properties:**

    - `multi_agent`: Indicates if the environment supports multiple agents.

    """

    def step(
        self,
        key: PRNGKeyArray,
        state: TEnvState,
        action: PyTree[int | float | Array],
    ) -> Tuple[TimeStep, TEnvState]:
        """
        Steps the environment forward with the given action and performs auto-reset when necessary.
        Environment-specific logic is defined in the `step_env` method. In principle, this function
        should not be overridden.

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
        Environment-specific logic is defined in the `reset_env` method. In principle, this function
        should not be overridden.

        **Arguments:**

        - `key`: JAX PRNG key.
        """
        obs, state = self.reset_env(key)
        return obs, state

    @abstractmethod
    def step_env(
        self, key: PRNGKeyArray, state: TEnvState, action: PyTree[int | float | Array]
    ) -> Tuple[TimeStep, TEnvState]:
        """
        Defines the environment-specific step logic.

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

        **Arguments:**

        - `key`: JAX PRNG key.
        """
        pass

    @property
    @abstractmethod
    def action_space(self) -> Space | PyTree[Space]:
        """
        Defines the space of valid actions for the environment.
        """
        pass

    @property
    @abstractmethod
    def observation_space(self) -> Space | PyTree[Space]:
        """
        Defines the space of possible observations from the environment.
        """
        pass

    @property
    def multi_agent(self) -> bool:
        """
        Indicates if the environment supports multiple agents.
        """
        return False

    @property
    def agent_structure(self) -> PyTreeDef:
        """
        Returns the structure of the agent space.
        This is useful for environments with multiple agents.
        """
        if not self.multi_agent:
            return jax.tree.structure(0)
        _, agent_structure = eqx.tree_flatten_one_level(self.action_space)
        return agent_structure
