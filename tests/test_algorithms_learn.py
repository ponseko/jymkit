import _consts as TEST_CONSTS
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jymkit as jym
from jymkit.algorithms import SAC


def test_sac_cartpole_specific():
    """Test SAC specifically on CartPole to debug the reward issue."""
    env = jym.make("CartPole-v1")
    seed = jax.random.PRNGKey(1)
    seed1, seed2 = jax.random.split(seed)

    # Test SAC with minimal training to see if the issue occurs
    agent = SAC(total_timesteps=10_000, log_function=None)
    agent = agent.train(seed1, env)

    # Test evaluation with different episode counts
    for num_episodes in [1, 5, 10]:
        print(f"\nTesting SAC with {num_episodes} evaluation episodes:")
        rewards = agent.evaluate(seed2, env, num_eval_episodes=num_episodes)

        print(f"  Rewards type: {type(rewards)}")
        print(f"  Rewards shape: {getattr(rewards, 'shape', 'no shape')}")
        print(f"  Rewards content: {rewards}")
        print(f"  Rewards dtype: {getattr(rewards, 'dtype', 'no dtype')}")

        if hasattr(rewards, "ravel"):
            print(f"  Raveled rewards: {rewards.ravel()}")

        # Additional debugging for CI issues
        print(f"  Rewards length: {len(rewards)}")
        print(f"  Is array: {hasattr(rewards, '__array__')}")
        print(f"  Is jax array: {hasattr(rewards, 'block_until_ready')}")

        # Check if rewards are valid
        assert len(rewards) == num_episodes, (
            f"Expected {num_episodes} rewards, got {len(rewards)}"
        )

        # Check if rewards are numeric
        avg_reward = np.mean(rewards)
        print(f"  Average reward: {avg_reward}")
        assert np.isfinite(avg_reward), f"Average reward is not finite: {avg_reward}"


@pytest.mark.parametrize("alg", TEST_CONSTS.DISCRETE_ALGS)
def test_discrete_is_learning(alg):
    # Confirm learning behavior on CartPole w/ default parameters
    env = jym.make("CartPole-v1")
    seed = jax.random.PRNGKey(1)
    seed1, seed2 = jax.random.split(seed)
    agent = alg(total_timesteps=1_000_000, log_function=None)
    agent = agent.train(seed1, env)

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
    env = jym.make("Pendulum-v1")
    seed = jax.random.PRNGKey(0)
    seed1, seed2 = jax.random.split(seed)
    if alg == SAC:
        agent = alg(**TEST_CONSTS.SAC_CONTINUOUS_CONFIG)
    else:
        agent = alg(total_timesteps=500_000, log_function=None)
    agent = agent.train(seed1, env)

    rewards = agent.evaluate(seed2, env, num_eval_episodes=50)
    avg_reward = np.mean(rewards)
    assert avg_reward > -400, (
        f"Average reward too low: {avg_reward}. Training may have failed."
    )
