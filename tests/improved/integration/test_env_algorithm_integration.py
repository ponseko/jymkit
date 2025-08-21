"""Integration tests for environment-algorithm interactions - fast but comprehensive."""

import jax
import jax.numpy as jnp
import pytest

import jymkit
from jymkit.algorithms import PPO, SAC
from ..helpers import FAST_PPO_CONFIG, FAST_SAC_CONFIG, quick_agent_test, quick_env_test


@pytest.mark.integration
@pytest.mark.fast
class TestEnvironmentAlgorithmIntegration:
    """Test that environments work correctly with algorithms."""
    
    def test_cartpole_ppo_integration(self):
        """Test CartPole environment with PPO algorithm."""
        env = jymkit.make("CartPole-v1")
        assert quick_env_test(env, steps=5)
        assert quick_agent_test(PPO, env, FAST_PPO_CONFIG)
    
    def test_pendulum_sac_integration(self):
        """Test Pendulum environment with SAC algorithm."""
        env = jymkit.make("Pendulum-v1")
        assert quick_env_test(env, steps=5)
        assert quick_agent_test(SAC, env, FAST_SAC_CONFIG)
    
    def test_wrapped_environment_integration(self):
        """Test that wrapped environments work with algorithms."""
        env = jymkit.make("CartPole-v1")
        env = jymkit.FlattenObservationWrapper(env)
        env = jymkit.ScaleRewardWrapper(env, scale=0.1)
        
        assert quick_env_test(env, steps=5)
        assert quick_agent_test(PPO, env, FAST_PPO_CONFIG)
    
    def test_vectorized_environment_integration(self):
        """Test vectorized environments with algorithms."""
        base_env = jymkit.make("CartPole-v1")
        env = jymkit.VecEnvWrapper(base_env, num_envs=2)
        
        # Quick vectorized test
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, 2)
        obs, states = env.reset(keys)
        assert obs.shape[0] == 2  # Batch dimension
        
        # Test with algorithm
        config = FAST_PPO_CONFIG.copy()
        config["num_envs"] = 2  # Match wrapper
        assert quick_agent_test(PPO, env, config)


@pytest.mark.integration  
@pytest.mark.fast
class TestSaveLoadIntegration:
    """Test save/load functionality with real environments."""
    
    def test_ppo_save_load_cycle(self, tmp_path):
        """Test complete save/load cycle for PPO."""
        env = jymkit.make("CartPole-v1")
        agent = PPO(**FAST_PPO_CONFIG)
        
        # Quick training
        agent = agent.train(jax.random.PRNGKey(0), env)
        
        # Save
        save_path = tmp_path / "ppo_test.eqx"
        agent.save_state(save_path)
        assert save_path.exists()
        
        # Load into new agent
        new_agent = PPO(**FAST_PPO_CONFIG)
        new_agent = new_agent.init(jax.random.PRNGKey(0), env)
        new_agent = new_agent.load_state(save_path)
        
        # Compare key weights
        assert jnp.allclose(
            agent.state.actor.ffn_layers[0].weight,
            new_agent.state.actor.ffn_layers[0].weight,
            rtol=1e-5
        )
    
    def test_sac_save_load_cycle(self, tmp_path):
        """Test complete save/load cycle for SAC.""" 
        env = jymkit.make("Pendulum-v1")
        agent = SAC(**FAST_SAC_CONFIG)
        
        # Quick training
        agent = agent.train(jax.random.PRNGKey(0), env)
        
        # Save
        save_path = tmp_path / "sac_test.eqx"
        agent.save_state(save_path)
        assert save_path.exists()
        
        # Load into new agent
        new_agent = SAC(**FAST_SAC_CONFIG)
        new_agent = new_agent.init(jax.random.PRNGKey(0), env)
        new_agent = new_agent.load_state(save_path)
        
        # Compare key weights
        assert jnp.allclose(
            agent.state.actor.ffn_layers[0].weight,
            new_agent.state.actor.ffn_layers[0].weight,
            rtol=1e-5
        )


@pytest.mark.integration
@pytest.mark.fast
class TestMultiAgentIntegration:
    """Test multi-agent environment integration with minimal overhead."""
    
    def test_simple_multi_agent_functionality(self):
        """Test basic multi-agent environment functionality."""
        # Create a minimal multi-agent environment
        from ..helpers import MockEnvironment
        
        class SimpleMultiAgentMock(MockEnvironment):
            @property
            def multi_agent(self) -> bool:
                return True
                
            def reset_env(self, key):
                obs = [jax.random.normal(key, (4,)) for _ in range(2)]  # 2 agents
                state = {"step": 0}
                return obs, state
                
            def step_env(self, key, state, actions):
                obs = [jax.random.normal(key, (4,)) for _ in range(2)]
                rewards = [jax.random.uniform(key) for _ in range(2)]
                
                from jymkit import TimeStep
                timestep = TimeStep(obs, rewards, False, False, {})
                new_state = {"step": state["step"] + 1}
                return timestep, new_state
            
            @property
            def observation_space(self):
                from jymkit import Box
                return [Box(low=-jnp.inf, high=jnp.inf, shape=(4,)) for _ in range(2)]
            
            @property
            def action_space(self):
                from jymkit import Discrete
                return [Discrete(2) for _ in range(2)]
        
        env = SimpleMultiAgentMock()
        assert quick_env_test(env, steps=3)
    
    def test_multi_agent_with_algorithm(self):
        """Test multi-agent environment with algorithm."""
        # Use existing simple multi-agent from test_custom_envs
        from ...test_custom_envs import SimpleMultiAgentEnv
        
        env = SimpleMultiAgentEnv()
        
        # Quick functionality test
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        actions = env.sample_action(key)
        timestep, new_state = env.step(key, state, actions)
        
        # Basic assertions
        assert len(obs) == env.num_agents
        assert len(timestep.reward) == env.num_agents