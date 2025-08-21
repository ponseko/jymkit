"""Unit tests for environment functionality - fast, isolated tests."""

import jax
import jax.numpy as jnp
import pytest

from jymkit import Box, Discrete, Environment, TimeStep
from ..helpers import MockEnvironment, assert_spaces_compatible, create_minimal_test_env


@pytest.mark.unit
@pytest.mark.fast
class TestEnvironmentInterface:
    """Test the basic Environment interface without heavy computation."""
    
    def test_mock_environment_basic_functionality(self):
        """Test that our mock environment works correctly."""
        env = MockEnvironment(obs_dim=4, action_dim=2)
        key = jax.random.PRNGKey(42)
        
        # Test reset
        obs, state = env.reset(key)
        assert obs.shape == (4,)
        assert state["step"] == 0
        
        # Test step
        action = 0
        timestep, new_state = env.step(key, state, action)
        assert timestep.observation.shape == (4,)
        assert isinstance(timestep.reward, (float, jnp.ndarray))
        assert new_state["step"] == 1
    
    def test_environment_spaces(self):
        """Test that environment spaces are properly defined."""
        env = create_minimal_test_env()
        assert_spaces_compatible(env)
        
        # Test action sampling
        key = jax.random.PRNGKey(0)
        action = env.action_space.sample(key)
        assert action is not None
    
    def test_environment_reset_consistency(self):
        """Test that environment reset produces consistent results."""
        env = MockEnvironment()
        key = jax.random.PRNGKey(123)
        
        obs1, state1 = env.reset(key)
        obs2, state2 = env.reset(key)
        
        # Same key should produce same result
        assert jnp.allclose(obs1, obs2)
        assert state1["step"] == state2["step"]
    
    def test_environment_step_shapes(self):
        """Test that step outputs have correct shapes and types."""
        env = MockEnvironment(obs_dim=6, action_dim=3)
        key = jax.random.PRNGKey(0)
        
        obs, state = env.reset(key)
        action = env.action_space.sample(key)
        timestep, new_state = env.step(key, state, action)
        
        assert timestep.observation.shape == (6,)
        assert isinstance(timestep.reward, (float, jnp.ndarray))
        assert isinstance(timestep.terminated, (bool, jnp.ndarray))
        assert isinstance(timestep.truncated, (bool, jnp.ndarray))
        assert isinstance(timestep.info, dict)


@pytest.mark.unit
@pytest.mark.fast
class TestSpaces:
    """Test space functionality without environment overhead."""
    
    def test_discrete_space(self):
        """Test Discrete space functionality."""
        space = Discrete(5)
        key = jax.random.PRNGKey(0)
        
        # Test sampling
        action = space.sample(key)
        assert 0 <= action < 5
        
        # Test multiple samples
        actions = jax.vmap(space.sample)(jax.random.split(key, 10))
        assert actions.shape == (10,)
        assert jnp.all((actions >= 0) & (actions < 5))
    
    def test_box_space(self):
        """Test Box space functionality."""
        low = jnp.array([-1.0, -2.0])
        high = jnp.array([1.0, 2.0])
        space = Box(low=low, high=high, shape=(2,))
        
        key = jax.random.PRNGKey(0)
        action = space.sample(key)
        
        assert action.shape == (2,)
        assert jnp.all(action >= low)
        assert jnp.all(action <= high)


@pytest.mark.unit
@pytest.mark.fast
class TestTimeStep:
    """Test TimeStep functionality."""
    
    def test_timestep_creation(self):
        """Test TimeStep creation and attributes."""
        obs = jnp.array([1.0, 2.0])
        reward = 1.5
        terminated = False
        truncated = True
        info = {"test": "value"}
        
        timestep = TimeStep(obs, reward, terminated, truncated, info)
        
        assert jnp.allclose(timestep.observation, obs)
        assert timestep.reward == reward
        assert timestep.terminated == terminated
        assert timestep.truncated == truncated
        assert timestep.info == info
    
    def test_timestep_with_arrays(self):
        """Test TimeStep with JAX arrays."""
        obs = jax.random.normal(jax.random.PRNGKey(0), (4,))
        reward = jnp.array(2.0)
        terminated = jnp.array(False)
        truncated = jnp.array(True)
        
        timestep = TimeStep(obs, reward, terminated, truncated, {})
        
        assert timestep.observation.shape == (4,)
        assert timestep.reward.shape == ()
        assert timestep.terminated.shape == ()
        assert timestep.truncated.shape == ()