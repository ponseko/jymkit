import _consts as TEST_CONSTS
import jax
import numpy as np
import pytest

import jymkit as jym
from jymkit.algorithms import PPO
from jymkit.envs.cartpole import CartPole


@pytest.mark.parametrize("alg", TEST_CONSTS.DISCRETE_ALGS)
def test_discrete_is_learning(alg):
    # Confirm learning behavior on CartPole w/ default parameters
    env = CartPole()
    seed = jax.random.PRNGKey(0)
    seed1, seed2 = jax.random.split(seed)
    agent = alg(total_timesteps=250_000, log_function=None)
    agent = agent.train(seed1, env)

    rewards = agent.evaluate(seed2, env, num_eval_episodes=50)
    avg_reward = np.mean(rewards)
    assert avg_reward > 200, (
        f"Average reward too low: {avg_reward}. Training may have failed."
    )


@pytest.mark.parametrize("alg", TEST_CONSTS.CONTINUOUS_ALGS)
def test_continuous_is_learning(alg):
    # Confirm learning behavior on Pendulum w/ default parameters
    env = jym.make("Pendulum-v1")
    seed = jax.random.PRNGKey(0)
    seed1, seed2 = jax.random.split(seed)
    agent = alg(total_timesteps=250_000, log_function=None)
    agent = agent.train(seed1, env)

    rewards = agent.evaluate(seed2, env, num_eval_episodes=50)
    avg_reward = np.mean(rewards)
    assert avg_reward > -250, (
        f"Average reward too low: {avg_reward}. Training may have failed."
    )


@pytest.mark.parametrize("env_name", TEST_CONSTS.CLASSIC_CONTROL_ENVS)
def test_classic_control_envs(env_name):
    env = jym.make(env_name)
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    for i in range(100):
        key, sample_key, step_key = jax.random.split(key, 3)
        action = env.action_space.sample(sample_key)
        timestep, state = env.step(step_key, state, action)


@pytest.mark.parametrize("env_name", TEST_CONSTS.CLASSIC_CONTROL_ENVS)
def test_class_control_envs_ppo_short(env_name):
    env = jym.make(env_name)
    seed = jax.random.PRNGKey(0)
    agent = PPO(**TEST_CONSTS.PPO_MIN_CONFIG)
    agent = agent.train(seed, env)
