from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jaxtyping import Array, PRNGKeyArray

from jymkit import Discrete, Environment, MultiDiscrete, Space, TimeStep
from jymkit.algorithms import PPO


class SimpleEnvState(eqx.Module):
    last_actions: Array
    current_step: int = 0


class SimpleMultiAgentEnv(Environment):
    num_agents: int = 3
    episode_length: int = 100
    current_step: int = 0

    @property
    def multi_agent(self) -> bool:
        return True

    def reset_env(self, key: PRNGKeyArray):
        state = SimpleEnvState(
            last_actions=jnp.ones((self.num_agents,), dtype=jnp.int32) * -1,
            current_step=0,
        )
        return self.get_observation(state), state

    def step_env(self, key: PRNGKeyArray, state: SimpleEnvState, action: Array):
        action_array = jnp.array(action, dtype=jnp.int32)
        # Reward is 1 for each agent if they all take the same action
        reward = jnp.sum(jnp.all(action_array == action_array[0]))
        reward = [reward for _ in range(self.num_agents)]

        new_state = SimpleEnvState(
            last_actions=action_array,
            current_step=state.current_step + 1,
        )

        truncated = new_state.current_step >= self.episode_length

        return TimeStep(
            self.get_observation(new_state),
            reward,
            False,  # terminated
            truncated,  # truncated
            {},
        ), new_state

    def get_observation(self, state: SimpleEnvState):
        # Each agent observes the last actions of all agents
        actions = state.last_actions
        obs = [actions for _ in range(self.num_agents)]

        return obs

    @property
    def action_space(self) -> List[Space]:
        num_actions = 3
        return [Discrete(num_actions) for _ in range(self.num_agents)]

    @property
    def observation_space(self) -> List[MultiDiscrete]:
        num_actions = 3
        nvec = np.array([num_actions] * self.num_agents)
        # Each agent observes the last action of the other two agents (vector of length 2)
        return [MultiDiscrete(nvec) for _ in range(self.num_agents)]


class HeterogeneousMultiAgentEnvState(eqx.Module):
    agent0_state: Array  # Discrete state for agent 0
    agent1_state: Array  # Discrete state for agent 1
    current_step: int = 0


class HeterogeneousMultiAgentSimpleEnv(Environment):
    num_agents: int = 2
    episode_length: int = 50

    @property
    def multi_agent(self) -> bool:
        return True

    def reset_env(self, key: PRNGKeyArray):
        # Use dict keys to access spaces for sampling
        state = HeterogeneousMultiAgentEnvState(
            agent0_state=jnp.array(-1, dtype=jnp.int32),
            agent1_state=jnp.array(-1, dtype=jnp.int32),
            current_step=0,
        )
        return self.get_observation(state), state

    def step_env(
        self,
        key: PRNGKeyArray,
        state: HeterogeneousMultiAgentEnvState,
        action: dict[str, Array],
    ):
        action0 = action["agent0"]
        action1 = action["agent1"]

        # Agent 0's state changes based on its action (e.g., action 0 -> state 0, action 1 -> state 1)
        new_agent0_state = (action0).astype(jnp.int32)

        # Agent 1's state changes based on its action (e.g., sum of its multi-discrete actions modulo 3)
        new_agent1_state = jnp.sum(action1) % 3

        new_state = HeterogeneousMultiAgentEnvState(
            agent0_state=new_agent0_state,
            agent1_state=new_agent1_state,
            current_step=state.current_step + 1,
        )

        # Simple reward: +1 if both agents are in state 1
        reward_val = jnp.where(
            (new_agent0_state == 1) & (new_agent1_state == 1), 1.0, 0.0
        )
        reward = {"agent0": reward_val, "agent1": reward_val}

        truncated = new_state.current_step >= self.episode_length
        terminated = False

        return TimeStep(
            self.get_observation(new_state),
            reward,
            terminated,
            truncated,
            {},
        ), new_state

    def get_observation(self, state: HeterogeneousMultiAgentEnvState):
        # Agent 0 observes its own state
        obs0 = state.agent0_state
        # Agent 1 observes both states
        obs1 = jnp.array([state.agent0_state, state.agent1_state])
        return {"agent0": obs0, "agent1": obs1}

    @property
    def action_space(self) -> dict[str, Space]:
        # Agent 0: Discrete action (0 or 1)
        space0 = Discrete(2)
        # Agent 1: MultiDiscrete action (two actions, each with 2 choices)
        space1 = MultiDiscrete(np.array([2, 2]))
        return {"agent0": space0, "agent1": space1}

    @property
    def observation_space(self) -> dict[str, Space]:
        # Agent 0: Discrete observation (state 0 or 1)
        space0 = Discrete(2)
        # Agent 1: MultiDiscrete observation (observes agent 0 state [0,1] and its own state [0,1,2])
        space1 = MultiDiscrete(np.array([2, 3]))
        return {"agent0": space0, "agent1": space1}


def test_simple_multi_agent_env_runs():
    env = SimpleMultiAgentEnv(num_agents=3, episode_length=100)
    obs, state = env.reset(jax.random.PRNGKey(0))
    for i in range(100):
        key = jax.random.PRNGKey(i)

        keys = optax.tree_utils.tree_split_key_like(key, env.action_space)  # type: ignore
        actions = jax.tree.map(lambda space, k: space.sample(k), env.action_space, keys)

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

        keys = optax.tree_utils.tree_split_key_like(key, env.action_space)  # type: ignore
        actions = jax.tree.map(lambda space, k: space.sample(k), env.action_space, keys)

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
