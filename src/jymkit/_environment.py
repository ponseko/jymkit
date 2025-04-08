from abc import abstractmethod
from typing import Tuple, Union

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

        state_step = self.step_env(key, state, action)
        obs_step, reward, terminated, truncated, info = (
            self.get_observation(state_step),
            self.get_reward(state_step, prev_state=state),
            self.get_terminated(state_step),
            self.get_truncated(state_step),
            self.get_info(state_step, action),
        )

        # Auto-reset
        state_reset = self.reset_env(key)
        obs_reset = self.get_observation(state_reset)
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
        state = self.reset_env(key)
        obs = self.get_observation(state)
        return obs, state

    @abstractmethod
    def step_env(self, key: PRNGKeyArray, state: EnvState, action: Action) -> EnvState:
        """
        Defines the environment-specific step logic.

        **Arguments:**

        - `key`: JAX PRNG key.
        - `state`: Current state of the environment.
        - `action`: Action to take in the environment.
        """
        pass

    @abstractmethod
    def reset_env(self, key: PRNGKeyArray) -> EnvState:
        """
        Defines the environment-specific reset logic.

        **Arguments:**

        - `key`: JAX PRNG key.
        """
        pass

    @abstractmethod
    def get_observation(self, state: EnvState) -> Observation:
        """
        Extracts an observation from the environment state.

        **Arguments:**

        - `state`: Current state of the environment.
        """
        pass

    @abstractmethod
    def get_reward(
        self, state: EnvState, prev_state: EnvState
    ) -> Union[float, jnp.ndarray]:
        """
        Calculates the reward based on the current and previous states.

        **Arguments:**

        - `state`: Current state of the environment.
        - `prev_state`: Previous state of the environment.
        """
        pass

    @abstractmethod
    def get_terminated(self, state: EnvState) -> bool:
        """
        Determines if the episode has terminated based on the environment state.

        **Arguments:**

        - `state`: Current state of the environment.
        """
        return False

    def get_truncated(self, state: EnvState) -> bool:
        """
        Determines if the episode has been truncated (e.g., due to reaching max steps).

        **Arguments:**

        - `state`: Current state of the environment.
        """
        return self.max_episode_steps > 0 and state.time >= self.max_episode_steps

    def get_info(self, state: EnvState, actions: Action) -> dict:
        """
        Provides additional diagnostic information from the environment state.

        **Arguments:**

        - `state`: Current state of the environment.
        - `actions`: Actions taken in the environment.
        """
        return {}

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
