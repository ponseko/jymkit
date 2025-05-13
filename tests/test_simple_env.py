import jax
import numpy as np
import pytest
from simple_env import HeterogeneousMultiAgentSimpleEnv, SimpleMultiAgentEnv

import jymkit as jym
from jymkit.algorithms import PPO
from jymkit.algorithms.utils import split_key_over_agents, transform_multi_agent
from jymkit.envs.cartpole import CartPole


def test_ppo_on_cartpole():
    env = CartPole()
    seed = jax.random.PRNGKey(0)
    agent = PPO(num_envs=4, num_steps=128, total_timesteps=250_000, log_function=None)

    try:
        agent = agent.train(seed, env)
    except Exception as e:
        pytest.fail(f"PPO training failed with error: {e}")

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
    env = jym.make("CartPole")
    env = jym.FlattenObservationWrapper(env)
    env = jym.LogWrapper(env)
    env = jym.VecEnvWrapper(env)
    env = jym.NormalizeVecRewardWrapper(env, gamma=0.99)
    # env = jym.NormalizeVecObsWrapper(env)
    seed = jax.random.PRNGKey(0)
    agent = PPO(num_envs=2, num_epochs=1, total_timesteps=10_000, log_function=None)
    agent = agent.train(seed, env)


def test_simple_multi_agent_env_runs():
    env = SimpleMultiAgentEnv(num_agents=3, episode_length=100)
    obs, state = env.reset(jax.random.PRNGKey(0))
    agent_structure = jax.tree.structure(env.observation_space)
    for i in range(100):
        key = jax.random.PRNGKey(i)

        keys = split_key_over_agents(key, agent_structure)
        actions = transform_multi_agent(
            lambda space, k: space.sample(k), identity=False
        )(env.action_space, keys)
        (obs, reward, terminated, truncated, info), state = env.step(
            key, state, actions
        )
        print(reward)


def test_ppo_on_simple_multi_agent_env():
    env = SimpleMultiAgentEnv(num_agents=3, episode_length=100)
    seed = jax.random.PRNGKey(42)
    agent = PPO(
        num_envs=4, num_steps=64, num_epochs=1, total_timesteps=5000, log_function=None
    )

    try:
        agent = agent.train(seed, env)
    except Exception as e:
        pytest.fail(f"PPO training failed on SimpleMultiAgentEnv with error: {e}")

    # Evaluate agent for 5 episodes (since it's a toy env)
    rewards = []
    for i in range(5):
        key = jax.random.PRNGKey(i)
        reward = agent.evaluate(key, env)
        rewards.append(reward)
    avg_reward = sum(rewards) / len(rewards)
    print(avg_reward)
    # No strict threshold, just check it runs and returns a float


def test_heterogeneous_multi_agent_env_runs():
    env = HeterogeneousMultiAgentSimpleEnv(num_agents=2, episode_length=100)
    obs, state = env.reset(jax.random.PRNGKey(0))
    for i in range(100):
        key = jax.random.PRNGKey(i)

        keys = split_key_over_agents(key, env.agent_structure)
        actions = transform_multi_agent(lambda space, k: space.sample(k))(
            env.action_space, keys
        )
        (obs, reward, terminated, truncated, info), state = env.step(
            key, state, actions
        )
        print(reward)


def test_ppo_on_heterogeneous_multi_agent_env():
    env = HeterogeneousMultiAgentSimpleEnv(num_agents=2, episode_length=100)
    seed = jax.random.PRNGKey(42)
    agent = PPO(
        num_envs=4, num_steps=64, num_epochs=1, total_timesteps=5000, log_function=None
    )

    try:
        agent = agent.train(seed, env)
    except Exception as e:
        pytest.fail(
            f"PPO training failed on HeterogeneousMultiAgentEnv with error: {e}"
        )

    # Evaluate agent for 5 episodes (since it's a toy env)
    rewards = []
    for i in range(5):
        key = jax.random.PRNGKey(i)
        reward = agent.evaluate(key, env)
        rewards.append(reward)
    avg_reward = sum(rewards) / len(rewards)
    print(avg_reward)
    # No strict threshold, just check it runs and returns a float
