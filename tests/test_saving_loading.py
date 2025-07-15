import os

import cloudpickle
import jax
import jax.numpy as jnp

import jymkit
import jymkit.algorithms


def test_saving_loading(tmp_path):
    # Create a simple environment
    env = jymkit.make("CartPole-v1")

    # Initialize the agent
    agent = jymkit.algorithms.PPO(
        num_envs=2, total_timesteps=1000, num_epochs=1, log_function=None
    )

    # Train the agent
    agent = agent.train(jax.random.PRNGKey(1), env)

    save_path = tmp_path / "test_saving_loading.eqx."
    agent.save_state(save_path)

    # Load the agent
    load_agent = jymkit.algorithms.PPO(
        num_envs=2, total_timesteps=1000, num_epochs=1, log_function=None
    )
    load_agent = load_agent.init(jax.random.PRNGKey(1), env)
    load_agent = load_agent.load_state(save_path)

    # Check if weights match (via some arbitary layer)
    assert jnp.all(
        agent.state.actor.ffn_layers[0].weight
        == load_agent.state.actor.layers[0].weight
    ), "Weights do not match after loading."
    assert jnp.all(
        agent.state.critic.ffn_layers[1].weight
        == load_agent.state.critic.layers[1].weight
    ), "Weights do not match after loading."

    # Check if the loaded agent can still train
    load_agent.train(jax.random.PRNGKey(1), env)
    load_agent.evaluate(jax.random.PRNGKey(1), env, num_eval_episodes=10)

    # Remove the saved file after the test
    os.remove(save_path)


def test_cloudpickle_saving(tmp_path):
    # Create a simple environment
    env = jymkit.make("CartPole-v1")

    # Initialize the agent
    agent = jymkit.algorithms.PPO(
        num_envs=2, total_timesteps=1000, num_epochs=1, log_function=None
    )

    # Train the agent
    agent = agent.train(jax.random.PRNGKey(1), env)

    save_path = tmp_path / "test_cloudpickle_saving.pkl"
    with open(save_path, "wb") as f:
        cloudpickle.dump(agent, f)

    # Load the agent
    with open(save_path, "rb") as f:
        load_agent: jymkit.algorithms.PPO = cloudpickle.load(f)

    # Check if weights match
    assert jnp.all(
        agent.state.actor.layers[0].weight == load_agent.state.actor.layers[0].weight
    ), "Weights do not match after loading."
    assert jnp.all(
        agent.state.critic.layers[1].weight == load_agent.state.critic.layers[1].weight
    ), "Weights do not match after loading."

    # Check if the loaded agent can still train
    load_agent.train(jax.random.PRNGKey(1), env)
    load_agent.evaluate(jax.random.PRNGKey(1), env, num_eval_episodes=10)

    # Remove the saved file after the test
    os.remove(save_path)
