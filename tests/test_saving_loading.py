import os

import cloudpickle
import jax
import jax.numpy as jnp
import pytest
from _test_utils import get_valid_test_algs

import jaxnasium as jym
import jaxnasium.algorithms as jxalgs

TEST_ENV = jym.make("CartPole-v1")


@pytest.mark.parametrize("test_alg_cls", get_valid_test_algs(TEST_ENV))
def test_saving_loading(tmp_path, test_alg_cls: type[jxalgs.RLAlgorithm]):
    # Create a simple environment
    env = TEST_ENV

    agent = test_alg_cls()  # type: ignore
    agent = agent.init_agent(jax.random.PRNGKey(1), env)

    save_path = tmp_path / "test_saving_loading.eqx."
    agent.save_state(save_path)

    # Reload the agent
    load_agent = test_alg_cls()  # type: ignore
    load_agent = load_agent.init_agent(jax.random.PRNGKey(42), env)
    load_agent = load_agent.load_state(save_path)

    agent_weight = jym.tree.get_first(agent, "weight")
    load_agent_weight = jym.tree.get_first(load_agent, "weight")
    assert jnp.all(agent_weight == load_agent_weight), (
        "Weights do not match after loading."
    )

    # Check if the loaded agent can still train
    # load_agent.train(jax.random.PRNGKey(1), env)
    load_agent.evaluate(jax.random.PRNGKey(1), env, num_eval_episodes=2)

    # Remove the saved file after the test
    os.remove(save_path)


@pytest.mark.parametrize("test_alg_cls", get_valid_test_algs(TEST_ENV))
def test_cloudpickle_saving(tmp_path, test_alg_cls: type[jxalgs.RLAlgorithm]):
    # Create a simple environment
    env = TEST_ENV

    agent = test_alg_cls()  # type: ignore
    agent = agent.init_agent(jax.random.PRNGKey(1), env)

    save_path = tmp_path / "test_cloudpickle_saving.pkl"
    with open(save_path, "wb") as f:
        cloudpickle.dump(agent, f)

    # Load the agent
    with open(save_path, "rb") as f:
        load_agent = cloudpickle.load(f)

    agent_weight = jym.tree.get_first(agent, "weight")
    load_agent_weight = jym.tree.get_first(load_agent, "weight")
    assert jnp.all(agent_weight == load_agent_weight), (
        "Weights do not match after loading."
    )

    # Check if the loaded agent can still train
    # load_agent.train(jax.random.PRNGKey(1), env)
    load_agent.evaluate(jax.random.PRNGKey(1), env, num_eval_episodes=2)

    # Remove the saved file after the test
    os.remove(save_path)
