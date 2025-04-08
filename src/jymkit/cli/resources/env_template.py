import equinox as eqx
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

import jymkit as jym


class EnvState(eqx.Module):
    x: int
    y: int
    time: int = 0

    @property
    def location(self):
        return (self.x, self.y)


class ExampleEnv(jym.Environment):
    max_episode_steps: int = 100

    def step_env(
        self, key: PRNGKeyArray, state: jym.EnvState, action: jym.Action
    ) -> jym.EnvState:
        """
        Update the environment state based on the action taken.
        """
        # action 0 -> move up
        # action 1 -> move right
        # action 2 -> move down
        # action 3 -> move left
        new_x = state.x + (action == 0) - (action == 2)
        new_y = state.y + (action == 1) - (action == 3)
        return EnvState(x=new_x, y=new_y, time=state.time + 1)

    def reset_env(self, key: PRNGKeyArray) -> jym.EnvState:
        """
        Reset the environment to its initial state.
        """
        return EnvState(x=5, y=5)  # Start in the center

    def get_observation(self, state: jym.EnvState) -> jym.Observation:
        """
        Get the observation from the environment state.
        """
        return state.location

    def get_reward(self, state: jym.EnvState, prev_state: jym.EnvState) -> float:
        """
        Get the reward from the environment state.
        """
        # Example reward function: 1 for each step taken
        return 1.0

    def get_terminated(self, state: jym.EnvState) -> bool:
        """
        Check if the episode has terminated.
        """
        # Example termination condition: if the agent moves out of bounds
        out_of_bounds_x = jnp.logical_or(state.x < 0, state.x >= 10)
        out_of_bounds_y = jnp.logical_or(state.y < 0, state.y >= 10)
        return jnp.logical_or(out_of_bounds_x, out_of_bounds_y)

    @property
    def observation_space(self) -> jym.Space:
        """
        Define the observation space of the environment.
        """
        return jym.Box(low=0, high=10, shape=(2,))

    @property
    def action_space(self) -> jym.Space:
        """
        Define the action space of the environment.
        """
        return jym.Discrete(n=4)
