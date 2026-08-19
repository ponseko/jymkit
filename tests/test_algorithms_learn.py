import _consts as TEST_CONSTS
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float, PRNGKeyArray

import jaxnasium as jym

# Only run these in "external" mode
pytestmark = pytest.mark.learn


class EnvState(eqx.Module):
    pos: Array  # agent position, shape (num_dimensions,)
    goal: Array  # target position, shape (num_dimensions,)
    time: int = 0


@jym.registry.register("PointReachEnv")
class PointReach(jym.Environment):
    max_episode_steps: int = 100

    # dynamics / geometry
    bound: float = 2.0  # world is [-bound, bound]^num_dimensions
    dt: float = 0.1  # action scaling (how far one step moves)
    goal_radius: float = 0.1  # success threshold
    action_scale: float = 1.0  # clip magnitude of action

    num_dimensions: int = 2

    def step_env(
        self, key: PRNGKeyArray, state: EnvState, action: Array
    ) -> tuple[jym.TimeStep, EnvState]:
        # clip action to a sane range, then integrate position
        action = jnp.clip(action, -self.action_scale, self.action_scale)
        new_pos = state.pos + self.dt * action
        new_pos = jnp.clip(new_pos, -self.bound, self.bound)

        new_state = EnvState(pos=new_pos, goal=state.goal, time=state.time + 1)

        timestep = jym.TimeStep(
            observation=self.get_observation(new_state),
            reward=self.get_reward(new_state),
            terminated=self.get_terminated(new_state),
            truncated=new_state.time >= self.max_episode_steps,
            info={"dist": self._dist(new_state)},
        )
        return timestep, new_state

    def reset_env(self, key: PRNGKeyArray) -> tuple[Array, EnvState]:
        k_pos, k_goal = jax.random.split(key)

        dims = jnp.arange(self.num_dimensions)
        pos = jnp.where(dims % 2 == 0, -self.bound, self.bound)
        goal = jnp.zeros(self.num_dimensions)
        # pos = jax.random.uniform(
        #     k_pos, (self.num_dimensions,), minval=-self.bound, maxval=self.bound
        # )
        # goal = jax.random.uniform(
        #     k_goal, (self.num_dimensions,), minval=-self.bound, maxval=self.bound
        # )
        del k_pos, k_goal
        state = EnvState(pos=pos, goal=goal, time=0)
        return self.get_observation(state), state

    def _dist(self, state: EnvState) -> Float[Array, ""]:
        return jnp.linalg.norm(state.pos - state.goal)

    def get_observation(self, state: EnvState) -> Array:
        # observe own position, goal, and relative vector to goal
        return jnp.concatenate([state.pos, state.goal, state.goal - state.pos])

    def get_reward(self, state: EnvState) -> Float[Array, ""]:
        # dense: negative distance; bonus for reaching the goal
        dist = self._dist(state)
        reached = dist < self.goal_radius
        return -dist + 10.0 * reached

    def get_terminated(self, state: EnvState) -> Float[Array, ""]:
        return self._dist(state) < self.goal_radius

    @property
    def observation_space(self) -> jym.Space:
        # pos + goal + relative, each in [-2*bound, 2*bound]
        obs_dim = 3 * self.num_dimensions
        return jym.Box(low=-2.0 * self.bound, high=2.0 * self.bound, shape=(obs_dim,))

    @property
    def action_space(self) -> jym.Space:
        return jym.Box(
            low=-self.action_scale,
            high=self.action_scale,
            shape=(self.num_dimensions,),
        )


@pytest.mark.parametrize("alg", TEST_CONSTS.DISCRETE_ALGS)
def test_discrete_is_learning(alg):
    # Confirm learning behavior on CartPole w/ default parameters
    env = jym.make("CartPole-v1")
    seed = jax.random.PRNGKey(1)
    seed1, seed2 = jax.random.split(seed)
    agent = alg(log_function=None)
    agent, _ = agent.train(seed1, env)

    rewards = agent.evaluate(seed2, env, num_eval_episodes=50)
    avg_reward = jnp.mean(rewards)
    assert avg_reward > 200, (
        f"Average reward too low: {avg_reward}. Training may have failed."
        f"Average reward: {avg_reward}, "
        f"Rewards array: {rewards}, "
        f"Rewards type: {type(rewards)}, "
        f"Rewards shape: {getattr(rewards, 'shape', 'no shape')}"
    )


@pytest.mark.parametrize("alg", TEST_CONSTS.CONTINUOUS_ALGS)
def test_continuous_is_learning(alg):
    # Confirm learning behavior on Pendulum w/ default parameters
    env = jym.make("PointReachEnv")
    seed = jax.random.PRNGKey(0)
    seed1, seed2 = jax.random.split(seed)
    agent = alg(total_timesteps=500_000, log_function=None)
    agent, _ = agent.train(seed1, env)

    rewards = agent.evaluate(seed2, env, num_eval_episodes=50)
    avg_reward = np.mean(rewards)
    assert avg_reward > -25, (
        f"Average reward too low: {avg_reward}. Training may have failed."
    )
