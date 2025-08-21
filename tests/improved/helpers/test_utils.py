"""Test utilities to reduce code duplication and improve test maintainability."""

import importlib.util
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import PRNGKeyArray

import jymkit
from jymkit import Environment
from jymkit.algorithms import PPO, SAC


# Minimal configurations for fast testing
FAST_PPO_CONFIG = {
    "num_envs": 2,
    "num_epochs": 1, 
    "num_minibatches": 1,
    "total_timesteps": 100,  # Much smaller for unit tests
    "log_function": None,
}

FAST_SAC_CONFIG = {
    "total_timesteps": 500,  # Much smaller for unit tests
    "num_envs": 2,
    "learning_rate": 0.01,
    "update_every": 4,
    "batch_size": 32,
    "replay_buffer_size": 1000,
    "log_function": None,
}

# Quick environment test configurations
QUICK_TEST_STEPS = 10
QUICK_EVAL_EPISODES = 3


def skip_if_missing(library: str) -> None:
    """Skip test if external library is not available."""
    if importlib.util.find_spec(library) is None:
        pytest.skip(f"{library} is not installed.")


def make_env_with_wrappers(env_name: str, wrappers: Optional[List[str]] = None) -> Environment:
    """Create environment with common wrappers applied."""
    env = jymkit.make(env_name)
    
    if wrappers is None:
        wrappers = ["FlattenObservationWrapper"]
    
    for wrapper_name in wrappers:
        wrapper_cls = getattr(jymkit, wrapper_name)
        env = wrapper_cls(env)
    
    return env


def quick_env_test(env: Environment, steps: int = QUICK_TEST_STEPS) -> bool:
    """Run a quick environment functionality test."""
    try:
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        
        for i in range(steps):
            key = jax.random.PRNGKey(i + 1)
            action = env.action_space.sample(key)
            timestep, state = env.step(key, state, action)
            
            # Basic sanity checks
            assert timestep.reward is not None
            assert timestep.observation is not None
            assert isinstance(timestep.terminated, (bool, jnp.ndarray))
            assert isinstance(timestep.truncated, (bool, jnp.ndarray))
            
        return True
    except Exception:
        return False


def quick_agent_test(agent_cls: type, env: Environment, config: Dict[str, Any]) -> bool:
    """Run a quick agent training test without full convergence."""
    try:
        agent = agent_cls(**config)
        agent = agent.train(jax.random.PRNGKey(0), env)
        
        # Quick evaluation
        rewards = agent.evaluate(jax.random.PRNGKey(1), env, num_eval_episodes=QUICK_EVAL_EPISODES)
        
        # Basic sanity checks
        assert len(rewards) == QUICK_EVAL_EPISODES
        assert all(isinstance(r, (int, float, jnp.ndarray)) for r in rewards)
        
        return True
    except Exception:
        return False


def assert_spaces_compatible(env: Environment) -> None:
    """Assert that environment spaces are properly defined."""
    obs_space = env.observation_space
    action_space = env.action_space
    
    assert obs_space is not None, "Observation space is None"
    assert action_space is not None, "Action space is None"
    
    # Test sampling
    key = jax.random.PRNGKey(42)
    try:
        action = action_space.sample(key)
        assert action is not None, "Action sampling failed"
    except Exception as e:
        pytest.fail(f"Action space sampling failed: {e}")


def create_minimal_test_env() -> Environment:
    """Create a minimal test environment for unit tests."""
    from jymkit.envs.cartpole import CartPole
    return CartPole()


class MockEnvironment(Environment):
    """Minimal mock environment for isolated unit tests."""
    
    def __init__(self, obs_dim: int = 4, action_dim: int = 2):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
    def reset_env(self, key: PRNGKeyArray) -> Tuple[jnp.ndarray, Any]:
        obs = jax.random.normal(key, (self.obs_dim,))
        state = {"step": 0}
        return obs, state
        
    def step_env(self, key: PRNGKeyArray, state: Any, action: jnp.ndarray) -> Tuple[Any, Any]:
        obs = jax.random.normal(key, (self.obs_dim,))
        reward = jax.random.uniform(key)
        new_state = {"step": state["step"] + 1}
        
        terminated = new_state["step"] >= 10
        truncated = False
        
        from jymkit import TimeStep
        timestep = TimeStep(obs, reward, terminated, truncated, {})
        return timestep, new_state
    
    @property
    def observation_space(self):
        from jymkit import Box
        return Box(low=-jnp.inf, high=jnp.inf, shape=(self.obs_dim,))
    
    @property 
    def action_space(self):
        from jymkit import Discrete
        return Discrete(self.action_dim)


def get_test_marks(test_type: str) -> List[str]:
    """Get appropriate pytest marks for test type."""
    marks = {
        "unit": ["unit", "fast"],
        "integration": ["integration", "fast"], 
        "performance": ["performance", "slow"],
        "external": ["external", "integration"]
    }
    return marks.get(test_type, [])