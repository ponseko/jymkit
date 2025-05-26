import jax
import numpy as np
import pytest

import jymkit as jym
from jymkit.algorithms import PPO
from jymkit.envs.cartpole import CartPole


def test_ppo_on_cartpole():
    env = CartPole()
    seed = jax.random.PRNGKey(0)
    agent = PPO(num_envs=4, num_steps=128, total_timesteps=250_000, log_function=None)
    agent = agent.train(seed, env)

    # Evaluate agent for 20 episodes
    rewards = []
    for i in range(20):
        key = jax.random.PRNGKey(i)
        reward = agent.evaluate(key, env)
        rewards.append(reward)
    avg_reward = np.mean(rewards)
    assert avg_reward > 200, (
        f"Average reward too low: {avg_reward}. Training may have failed."
    )


def test_ppo_cartpole_with_wrappers():
    env = jym.make("CartPole-v1")
    env = jym.FlattenObservationWrapper(env)
    env = jym.ScaleRewardWrapper(env, scale=0.1)
    env = jym.LogWrapper(env)
    env = jym.VecEnvWrapper(env)
    env = jym.NormalizeVecRewardWrapper(env, gamma=0.99)
    # env = jym.NormalizeVecObsWrapper(env)
    seed = jax.random.PRNGKey(0)
    agent = PPO(num_envs=2, num_epochs=1, total_timesteps=10_000, log_function=None)
    agent = agent.train(seed, env)


@pytest.mark.parametrize(
    "env_name",
    [
        "CartPolev-v1",
        "MountainCar-v0",
        "Acrobot-v1",
        "Pendulum-v1",
        "ContinuousMountainCar-v0",
    ],
)
def test_classic_control_envs_step(env_name):
    env = jym.make(env_name)
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    for i in range(100):
        key, sample_key, step_key = jax.random.split(key, 3)
        action = env.action_space.sample(sample_key)
        timestep, state = env.step(step_key, state, action)


@pytest.mark.parametrize(
    "env_name",
    [
        "CartPolev-v1",
        "MountainCar-v0",
        "Acrobot-v1",
        "Pendulum-v1",
        "ContinuousMountainCar-v0",
    ],
)
def test_class_control_envs_ppo_short(env_name):
    env = jym.make(env_name)
    seed = jax.random.PRNGKey(0)
    agent = PPO(num_envs=2, num_epochs=1, total_timesteps=1_000, log_function=None)
    agent = agent.train(seed, env)
