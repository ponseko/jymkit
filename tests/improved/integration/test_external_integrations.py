"""Integration tests for external library compatibility - fast verification."""

import jax
import pytest

import jymkit
from ..helpers import FAST_PPO_CONFIG, quick_agent_test, quick_env_test, skip_if_missing


@pytest.mark.integration
@pytest.mark.fast
@pytest.mark.external
class TestGymnaxIntegration:
    """Test Gymnax environment integration with minimal overhead."""
    
    def test_gymnax_cartpole_basic(self):
        """Test basic Gymnax CartPole functionality."""
        skip_if_missing("gymnax")
        
        env = jymkit.make("gymnax:CartPole-v1")
        assert quick_env_test(env, steps=5)
    
    def test_gymnax_with_flatten_wrapper(self):
        """Test Gymnax environment with flattening wrapper."""
        skip_if_missing("gymnax")
        
        env = jymkit.make("Breakout-MinAtar")
        
        # Should require flattening - test the error handling
        agent = jymkit.algorithms.PPO(**FAST_PPO_CONFIG)
        with pytest.raises(AssertionError, match="Flatten the observations"):
            agent.train(jax.random.PRNGKey(0), env)
        
        # Should work with flattening
        env = jymkit.FlattenObservationWrapper(env)
        assert quick_env_test(env, steps=3)
    
    def test_gymnax_ppo_quick_integration(self):
        """Test Gymnax environment with PPO - minimal training."""
        skip_if_missing("gymnax")
        
        env = jymkit.make("gymnax:CartPole-v1")
        env = jymkit.FlattenObservationWrapper(env)
        assert quick_agent_test(PPO, env, FAST_PPO_CONFIG)


@pytest.mark.integration
@pytest.mark.fast 
@pytest.mark.external
class TestJumanjiIntegration:
    """Test Jumanji environment integration with minimal overhead."""
    
    def test_jumanji_snake_basic(self):
        """Test basic Jumanji Snake functionality."""
        skip_if_missing("jumanji")
        
        env = jymkit.make("Snake-v1")
        env = jymkit.FlattenObservationWrapper(env)
        assert quick_env_test(env, steps=3)
    
    def test_jumanji_ppo_quick_integration(self):
        """Test Jumanji environment with PPO - minimal training."""
        skip_if_missing("jumanji")
        
        env = jymkit.make("Game2048-v1")
        env = jymkit.FlattenObservationWrapper(env)
        assert quick_agent_test(PPO, env, FAST_PPO_CONFIG)


@pytest.mark.integration
@pytest.mark.fast
@pytest.mark.external
class TestBraxIntegration:
    """Test Brax environment integration with minimal overhead."""
    
    def test_brax_ant_basic(self):
        """Test basic Brax Ant functionality."""
        skip_if_missing("brax")
        
        env = jymkit.make("ant")
        assert quick_env_test(env, steps=3)
    
    def test_brax_ppo_quick_integration(self):
        """Test Brax environment with PPO - minimal training."""
        skip_if_missing("brax")
        
        env = jymkit.make("halfcheetah")
        env = jymkit.FlattenObservationWrapper(env)
        assert quick_agent_test(PPO, env, FAST_PPO_CONFIG)
    
    def test_brax_discrete_action_wrapper(self):
        """Test Brax with discrete action wrapper."""
        skip_if_missing("brax")
        
        env = jymkit.make("ant")
        env = jymkit.DiscreteActionWrapper(env, num_actions=5)
        
        # Test basic functionality
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        action = env.action_space.sample(key)
        timestep, new_state = env.step(key, state, action)
        
        # Verify discrete action
        assert 0 <= action < 5
        assert isinstance(action, (int, jnp.integer, jnp.ndarray))


@pytest.mark.integration
@pytest.mark.fast
@pytest.mark.external
class TestSpecializedLibraryIntegration:
    """Test integration with specialized external libraries."""
    
    def test_econojax_integration(self):
        """Test EconoJax integration - minimal test."""
        try:
            from econojax import EconoJax
        except ImportError:
            pytest.skip("EconoJax is not installed.")
        
        env = EconoJax(num_population=2)  # Smaller population for speed
        assert quick_env_test(env, steps=2)
    
    def test_rice_jax_integration(self):
        """Test Rice-JAX integration - minimal test."""
        try:
            from rice_jax import Rice
            from rice_jax.util import load_region_yamls
        except ImportError:
            pytest.skip("rice_jax is not installed.")
        
        region_yamls = load_region_yamls(2)  # Smaller regions for speed
        env = Rice(region_yamls)
        assert quick_env_test(env, steps=2)