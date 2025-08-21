"""Unit tests for algorithm functionality - isolated and fast."""

import jax
import jax.numpy as jnp
import pytest

from jymkit.algorithms import PPO, SAC
from ..helpers import FAST_PPO_CONFIG, FAST_SAC_CONFIG, MockEnvironment


@pytest.mark.unit
@pytest.mark.fast
class TestPPOInitialization:
    """Test PPO algorithm initialization and basic functionality."""
    
    def test_ppo_init_with_config(self):
        """Test PPO initialization with configuration."""
        agent = PPO(**FAST_PPO_CONFIG)
        assert agent is not None
        assert hasattr(agent, 'train')
        assert hasattr(agent, 'evaluate')
    
    def test_ppo_init_with_environment(self):
        """Test PPO initialization with environment."""
        env = MockEnvironment()
        agent = PPO(**FAST_PPO_CONFIG)
        
        # Test initialization
        initialized_agent = agent.init(jax.random.PRNGKey(0), env)
        assert initialized_agent is not None
        assert hasattr(initialized_agent, 'state')
    
    def test_ppo_state_structure(self):
        """Test that PPO state has expected structure."""
        env = MockEnvironment()
        agent = PPO(**FAST_PPO_CONFIG)
        agent = agent.init(jax.random.PRNGKey(0), env)
        
        assert hasattr(agent.state, 'actor')
        assert hasattr(agent.state, 'critic')
        assert hasattr(agent.state, 'actor_opt_state')
        assert hasattr(agent.state, 'critic_opt_state')
    
    def test_ppo_quick_training_step(self):
        """Test a single training step without full training."""
        env = MockEnvironment()
        agent = PPO(**FAST_PPO_CONFIG)
        
        # Quick training test - just verify it doesn't crash
        try:
            agent = agent.train(jax.random.PRNGKey(0), env)
            assert agent is not None
        except Exception as e:
            pytest.fail(f"PPO quick training failed: {e}")


@pytest.mark.unit
@pytest.mark.fast
class TestSACInitialization:
    """Test SAC algorithm initialization and basic functionality."""
    
    def test_sac_init_with_config(self):
        """Test SAC initialization with configuration."""
        agent = SAC(**FAST_SAC_CONFIG)
        assert agent is not None
        assert hasattr(agent, 'train')
        assert hasattr(agent, 'evaluate')
    
    def test_sac_init_with_environment(self):
        """Test SAC initialization with environment."""
        env = MockEnvironment()
        agent = SAC(**FAST_SAC_CONFIG)
        
        # Test initialization
        initialized_agent = agent.init(jax.random.PRNGKey(0), env)
        assert initialized_agent is not None
        assert hasattr(initialized_agent, 'state')
    
    def test_sac_state_structure(self):
        """Test that SAC state has expected structure."""
        env = MockEnvironment()
        agent = SAC(**FAST_SAC_CONFIG)
        agent = agent.init(jax.random.PRNGKey(0), env)
        
        assert hasattr(agent.state, 'actor')
        assert hasattr(agent.state, 'critic')
        assert hasattr(agent.state, 'critic_target')
    
    def test_sac_quick_training_step(self):
        """Test a single training step without full training."""
        env = MockEnvironment()
        agent = SAC(**FAST_SAC_CONFIG)
        
        # Quick training test - just verify it doesn't crash
        try:
            agent = agent.train(jax.random.PRNGKey(0), env)
            assert agent is not None
        except Exception as e:
            pytest.fail(f"SAC quick training failed: {e}")


@pytest.mark.unit
@pytest.mark.fast 
class TestAlgorithmInterface:
    """Test common algorithm interface functionality."""
    
    @pytest.mark.parametrize("alg_cls,config", [
        (PPO, FAST_PPO_CONFIG),
        (SAC, FAST_SAC_CONFIG)
    ])
    def test_algorithm_evaluate_functionality(self, alg_cls, config):
        """Test that algorithms can evaluate without errors."""
        env = MockEnvironment()
        agent = alg_cls(**config)
        agent = agent.init(jax.random.PRNGKey(0), env)
        
        # Quick evaluation test
        rewards = agent.evaluate(jax.random.PRNGKey(1), env, num_eval_episodes=2)
        assert len(rewards) == 2
        assert all(isinstance(r, (int, float, jnp.ndarray)) for r in rewards)
    
    @pytest.mark.parametrize("alg_cls,config", [
        (PPO, FAST_PPO_CONFIG),
        (SAC, FAST_SAC_CONFIG)
    ])
    def test_algorithm_save_load_interface(self, alg_cls, config, tmp_path):
        """Test that algorithms have save/load interface."""
        env = MockEnvironment()
        agent = alg_cls(**config)
        agent = agent.init(jax.random.PRNGKey(0), env)
        
        # Test save/load interface exists
        assert hasattr(agent, 'save_state')
        assert hasattr(agent, 'load_state')
        
        # Quick save test
        save_path = tmp_path / "test_agent.eqx"
        try:
            agent.save_state(save_path)
            assert save_path.exists()
        except Exception as e:
            pytest.fail(f"Save failed: {e}")