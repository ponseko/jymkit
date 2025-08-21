"""Unit tests for wrapper functionality - isolated and fast."""

import jax
import jax.numpy as jnp
import pytest

import jymkit
from ..helpers import MockEnvironment, assert_spaces_compatible


@pytest.mark.unit
@pytest.mark.fast
class TestFlattenObservationWrapper:
    """Test FlattenObservationWrapper in isolation."""
    
    def test_flattens_simple_observation(self):
        """Test basic observation flattening."""
        env = MockEnvironment(obs_dim=4)
        wrapped_env = jymkit.FlattenObservationWrapper(env)
        
        key = jax.random.PRNGKey(0)
        obs, state = wrapped_env.reset(key)
        
        # Should still be flat since mock env already returns flat obs
        assert obs.shape == (4,)
        assert_spaces_compatible(wrapped_env)
    
    def test_preserves_environment_interface(self):
        """Test that wrapper preserves the environment interface."""
        env = MockEnvironment()
        wrapped_env = jymkit.FlattenObservationWrapper(env)
        
        key = jax.random.PRNGKey(0)
        obs, state = wrapped_env.reset(key)
        action = wrapped_env.action_space.sample(key)
        timestep, new_state = wrapped_env.step(key, state, action)
        
        # Should maintain timestep structure
        assert hasattr(timestep, 'observation')
        assert hasattr(timestep, 'reward')
        assert hasattr(timestep, 'terminated')
        assert hasattr(timestep, 'truncated')
        assert hasattr(timestep, 'info')


@pytest.mark.unit
@pytest.mark.fast
class TestScaleRewardWrapper:
    """Test ScaleRewardWrapper functionality."""
    
    def test_scales_reward_correctly(self):
        """Test that rewards are scaled by the specified factor."""
        env = MockEnvironment()
        scale = 2.5
        wrapped_env = jymkit.ScaleRewardWrapper(env, scale=scale)
        
        key = jax.random.PRNGKey(0)
        obs, state = wrapped_env.reset(key)
        action = wrapped_env.action_space.sample(key)
        
        # Get original reward
        original_timestep, _ = env.step(key, state, action)
        
        # Get scaled reward
        scaled_timestep, _ = wrapped_env.step(key, state, action)
        
        expected_reward = original_timestep.reward * scale
        assert jnp.isclose(scaled_timestep.reward, expected_reward)
    
    def test_preserves_other_timestep_attributes(self):
        """Test that scaling only affects reward."""
        env = MockEnvironment()
        wrapped_env = jymkit.ScaleRewardWrapper(env, scale=0.5)
        
        key = jax.random.PRNGKey(0)
        obs, state = wrapped_env.reset(key)
        action = wrapped_env.action_space.sample(key)
        
        original_timestep, original_state = env.step(key, state, action)
        scaled_timestep, scaled_state = wrapped_env.step(key, state, action)
        
        # Everything except reward should be the same
        assert jnp.allclose(original_timestep.observation, scaled_timestep.observation)
        assert original_timestep.terminated == scaled_timestep.terminated
        assert original_timestep.truncated == scaled_timestep.truncated


@pytest.mark.unit
@pytest.mark.fast
class TestDiscreteActionWrapper:
    """Test DiscreteActionWrapper functionality."""
    
    def test_discretizes_continuous_actions(self):
        """Test that continuous actions are properly discretized."""
        # Create a simple continuous environment
        from jymkit import Box
        
        class ContinuousEnv(MockEnvironment):
            @property
            def action_space(self):
                return Box(low=-1.0, high=1.0, shape=(2,))
        
        env = ContinuousEnv()
        num_actions = 5
        wrapped_env = jymkit.DiscreteActionWrapper(env, num_actions=num_actions)
        
        # Action space should now be discrete
        assert isinstance(wrapped_env.action_space, type(jymkit.Discrete(1)))
        assert wrapped_env.action_space.n == num_actions
        
        # Test action sampling
        key = jax.random.PRNGKey(0)
        action = wrapped_env.action_space.sample(key)
        assert 0 <= action < num_actions
    
    def test_step_with_discrete_action(self):
        """Test stepping with discretized actions."""
        from jymkit import Box
        
        class ContinuousEnv(MockEnvironment):
            @property
            def action_space(self):
                return Box(low=-1.0, high=1.0, shape=(1,))
        
        env = ContinuousEnv()
        wrapped_env = jymkit.DiscreteActionWrapper(env, num_actions=3)
        
        key = jax.random.PRNGKey(0)
        obs, state = wrapped_env.reset(key)
        
        # Test with each discrete action
        for discrete_action in range(3):
            timestep, new_state = wrapped_env.step(key, state, discrete_action)
            assert timestep.observation is not None
            assert timestep.reward is not None


@pytest.mark.unit
@pytest.mark.fast
class TestVecEnvWrapper:
    """Test VecEnvWrapper functionality."""
    
    def test_vectorizes_environment(self):
        """Test that environment is properly vectorized."""
        env = MockEnvironment()
        num_envs = 4
        wrapped_env = jymkit.VecEnvWrapper(env, num_envs=num_envs)
        
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, num_envs)
        
        # Test reset
        obs, states = wrapped_env.reset(keys)
        assert obs.shape == (num_envs, 4)  # obs_dim=4 from MockEnvironment
        
        # Test step
        actions = jax.vmap(wrapped_env.action_space.sample)(keys)
        timestep, new_states = wrapped_env.step(keys, states, actions)
        
        assert timestep.observation.shape == (num_envs, 4)
        assert timestep.reward.shape == (num_envs,)
    
    def test_maintains_environment_interface(self):
        """Test that vectorized environment maintains proper interface."""
        env = MockEnvironment()
        wrapped_env = jymkit.VecEnvWrapper(env, num_envs=2)
        
        # Should still be a valid environment
        assert_spaces_compatible(wrapped_env)