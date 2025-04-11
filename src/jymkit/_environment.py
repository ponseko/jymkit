from abc import abstractmethod
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, PyTree

from ._spaces import Space
from ._types import Action, EnvState, Observation, TimeStep

ORIGINAL_OBSERVATION_KEY = "_TERMINAL_OBSERVATION"


class Environment(eqx.Module):
    """
    Abstract environment template for reinforcement learning environments in JAX.

    Provides a standardized interface for RL environments with JAX compatibility.
    Subclasses must implement the abstract methods to define specific environment behaviors.

    **Properties:**

    - `max_episode_steps`: Maximum number of steps in an episode before truncation. If 0, no limit is enforced (default: 0)
    - `multi_agent`: Indicates if the environment supports multiple agents.

    """

    max_episode_steps: int = 0
    multi_agent: bool = False

    def step(
        self, key: PRNGKeyArray, state: EnvState, action: Action
    ) -> Tuple[TimeStep, EnvState]:
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

        # To bootstrap correctly on truncated episodes
        info[ORIGINAL_OBSERVATION_KEY] = obs_step

        return TimeStep(obs, reward, terminated, truncated, info), state

    def reset(self, key: PRNGKeyArray) -> Tuple[Observation, EnvState]:
        """
        Resets the environment to an initial state and returns the initial observation.
        Environment-specific logic is defined in the `reset_env` method. In principle, this function
        should not be overridden.

        **Arguments:**

        - `key`: JAX PRNG key.
        """
        return self.reset_env(key)

    @abstractmethod
    def step_env(
        self, key: PRNGKeyArray, state: EnvState, action: Action
    ) -> Tuple[TimeStep, EnvState]:
        """
        Defines the environment-specific step logic.

        **Arguments:**

        - `key`: JAX PRNG key.
        - `state`: Current state of the environment.
        - `action`: Action to take in the environment.
        """
        pass

    @abstractmethod
    def reset_env(self, key: PRNGKeyArray) -> Tuple[Observation, EnvState]:
        """
        Defines the environment-specific reset logic.

        **Arguments:**

        - `key`: JAX PRNG key.
        """
        pass

    @property
    @abstractmethod
    def action_space(self) -> PyTree[Space]:
        """
        Defines the space of valid actions for the environment.
        """
        pass

    @property
    @abstractmethod
    def observation_space(self) -> PyTree[Space]:
        """
        Defines the space of possible observations from the environment.
        """
        pass
