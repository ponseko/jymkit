import importlib.util

import jax
import pytest

import jymkit as jym
from jymkit.algorithms import PPO


def test_ppo_cartpole_with_wrappers():
    env = jym.make("CartPole-v1")
    env = jym.FlattenObservationWrapper(env)
    env = jym.ScaleRewardWrapper(env, scale=0.1)
    env = jym.LogWrapper(env)
    env = jym.VecEnvWrapper(env)
    env = jym.NormalizeVecRewardWrapper(env, gamma=0.99)
    env = jym.ScaleRewardWrapper(env, scale=2)
    env = jym.NormalizeVecObsWrapper(env)

    # test regular step
    seed = jax.random.PRNGKey(0)
    _, some_state = env.reset(jax.random.split(seed, 3))
    action = jax.vmap(env.action_space.sample)(jax.random.split(seed, 3))
    env.step(jax.random.split(seed, 3), some_state, action)

    # test training
    agent = PPO(num_envs=2, num_epochs=1, total_timesteps=10_000, log_function=None)
    agent = agent.train(seed, env)


def test_discretization_single_action():
    env = jym.make("Pendulum-v1")
    env = jym.DiscreteActionWrapper(env, num_actions=10)

    # test regular step
    seed = jax.random.PRNGKey(0)
    _, some_state = env.reset(seed)
    env.step(seed, some_state, env.action_space.sample(seed))

    agent = PPO(num_envs=2, num_epochs=1, total_timesteps=10_000, log_function=None)
    agent = agent.train(seed, env)


def test_discretization_multi_action():
    if importlib.util.find_spec("jumanji") is None:
        pytest.skip("Jumanji is not installed.")
    env = jym.make("hopper")
    env = jym.DiscreteActionWrapper(env, num_actions=10)

    # test regular step
    seed = jax.random.PRNGKey(0)
    _, some_state = env.reset(seed)
    env.step(seed, some_state, env.action_space.sample(seed))

    agent = PPO(num_envs=2, num_epochs=1, total_timesteps=10_000, log_function=None)
    agent = agent.train(seed, env)
