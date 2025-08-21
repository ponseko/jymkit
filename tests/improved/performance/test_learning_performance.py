"""Performance tests that verify actual learning - marked as slow."""

import jax
import numpy as np
import pytest

import jymkit
from jymkit.algorithms import PPO, SAC
from ..helpers import skip_if_missing


# Configurations for learning verification
LEARNING_PPO_CONFIG = {
    "num_envs": 4,
    "num_epochs": 4,
    "num_minibatches": 4,
    "total_timesteps": 100_000,  # Reduced from 1M for faster CI
    "log_function": None,
}

LEARNING_SAC_CONFIG = {
    "total_timesteps": 50_000,  # Reduced from 1M for faster CI  
    "num_envs": 4,
    "learning_rate": 0.001,
    "update_every": 4,
    "batch_size": 128,
    "replay_buffer_size": 10_000,
    "normalize_rew": False,
    "normalize_obs": False,
    "log_function": None,
}


@pytest.mark.performance
@pytest.mark.slow
class TestLearningVerification:
    """Verify that algorithms actually learn on standard benchmarks."""
    
    def test_ppo_cartpole_learning(self):
        """Verify PPO learns CartPole - this is the core learning test."""
        env = jymkit.make("CartPole-v1")
        agent = PPO(**LEARNING_PPO_CONFIG)
        
        # Train agent
        agent = agent.train(jax.random.PRNGKey(0), env)
        
        # Evaluate performance
        rewards = agent.evaluate(jax.random.PRNGKey(1), env, num_eval_episodes=20)
        avg_reward = np.mean(rewards)
        
        # CartPole should reach high performance
        assert avg_reward > 150, f"PPO failed to learn CartPole. Avg reward: {avg_reward}"
    
    def test_sac_pendulum_learning(self):
        """Verify SAC learns Pendulum - this is the core continuous learning test."""
        env = jymkit.make("Pendulum-v1") 
        agent = SAC(**LEARNING_SAC_CONFIG)
        
        # Train agent
        agent = agent.train(jax.random.PRNGKey(0), env)
        
        # Evaluate performance  
        rewards = agent.evaluate(jax.random.PRNGKey(1), env, num_eval_episodes=20)
        avg_reward = np.mean(rewards)
        
        # Pendulum baseline should improve significantly
        assert avg_reward > -500, f"SAC failed to learn Pendulum. Avg reward: {avg_reward}"


@pytest.mark.performance
@pytest.mark.slow
@pytest.mark.external
class TestExternalLibraryLearning:
    """Verify learning on external library environments."""
    
    def test_gymnax_learning_verification(self):
        """Verify learning works on Gymnax environments."""
        skip_if_missing("gymnax")
        
        env = jymkit.make("gymnax:CartPole-v1")
        agent = PPO(**LEARNING_PPO_CONFIG)
        
        # Train and verify basic learning occurred
        agent = agent.train(jax.random.PRNGKey(0), env)
        rewards = agent.evaluate(jax.random.PRNGKey(1), env, num_eval_episodes=10)
        avg_reward = np.mean(rewards)
        
        # Less strict than native CartPole but should still show learning
        assert avg_reward > 50, f"Learning failed on Gymnax CartPole. Avg reward: {avg_reward}"
    
    def test_brax_learning_verification(self):
        """Verify learning works on Brax environments."""
        skip_if_missing("brax")
        
        env = jymkit.make("halfcheetah") 
        env = jymkit.FlattenObservationWrapper(env)
        agent = PPO(**LEARNING_PPO_CONFIG)
        
        # Train and verify basic learning occurred  
        agent = agent.train(jax.random.PRNGKey(0), env)
        rewards = agent.evaluate(jax.random.PRNGKey(1), env, num_eval_episodes=10)
        avg_reward = np.mean(rewards)
        
        # Basic learning verification - should improve from random
        assert avg_reward > -1000, f"Learning failed on Brax HalfCheetah. Avg reward: {avg_reward}"


@pytest.mark.performance 
@pytest.mark.slow
class TestLearningStability:
    """Test learning stability across multiple seeds."""
    
    def test_ppo_multi_seed_stability(self):
        """Test that PPO learning is stable across different seeds."""
        env = jymkit.make("CartPole-v1")
        
        rewards_across_seeds = []
        for seed in range(3):  # Test multiple seeds
            agent = PPO(**LEARNING_PPO_CONFIG)
            agent = agent.train(jax.random.PRNGKey(seed), env)
            
            eval_rewards = agent.evaluate(jax.random.PRNGKey(seed + 100), env, num_eval_episodes=10)
            avg_reward = np.mean(eval_rewards)
            rewards_across_seeds.append(avg_reward)
        
        # All seeds should achieve reasonable performance
        assert all(r > 100 for r in rewards_across_seeds), (
            f"PPO learning unstable across seeds. Rewards: {rewards_across_seeds}"
        )
        
        # Variance shouldn't be too high
        std_reward = np.std(rewards_across_seeds)
        assert std_reward < 100, f"PPO learning too variable. Std: {std_reward}"