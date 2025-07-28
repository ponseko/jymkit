import importlib.util

import _consts as TEST_CONSTS
import jax
import pytest

import jymkit
import jymkit.algorithms


@pytest.mark.parametrize("env_name", TEST_CONSTS.GYMNAX_TEST_ENVS)
def test_gymnax_envs(env_name):
    if importlib.util.find_spec("gymnax") is None:
        pytest.skip("Gymnax is not installed.")
    env = jymkit.make(env_name)
    agent = jymkit.algorithms.PPO(**TEST_CONSTS.PPO_MIN_CONFIG)

    # Expecting this to fail without flattening
    if env_name == "Breakout-MinAtar":
        with pytest.raises(AssertionError) as excinfo:
            agent = agent.train(jax.random.PRNGKey(1), env)
        assert (
            "Flatten the observations with `jymkit.FlattenObservationWrapper`"
            in str(excinfo.value)
        )

    env = jymkit.FlattenObservationWrapper(env)
    agent = agent.train(jax.random.PRNGKey(1), env)


@pytest.mark.parametrize("env_name", TEST_CONSTS.JUMANJI_TEST_ENVS)
def test_jumanji_envs(env_name):
    if importlib.util.find_spec("jumanji") is None:
        pytest.skip("Jumanji is not installed.")
    env = jymkit.make(env_name)
    agent = jymkit.algorithms.PPO(**TEST_CONSTS.PPO_MIN_CONFIG)
    env = jymkit.FlattenObservationWrapper(env)
    agent = agent.train(jax.random.PRNGKey(1), env)


@pytest.mark.parametrize("env_name", TEST_CONSTS.BRAX_TEST_ENVS)
def test_brax_envs(env_name):
    if importlib.util.find_spec("brax") is None:
        pytest.skip("Brax is not installed.")
    env = jymkit.make(env_name)
    agent = jymkit.algorithms.PPO(**TEST_CONSTS.PPO_MIN_CONFIG)
    env = jymkit.FlattenObservationWrapper(env)
    agent = agent.train(jax.random.PRNGKey(1), env)


def test_econojax_env():
    try:
        from econojax import EconoJax  # pyright: ignore
    except ImportError:
        pytest.skip("Econojax is not installed.")

    env = EconoJax(num_population=4)
    agent = jymkit.algorithms.PPO(**TEST_CONSTS.PPO_MIN_CONFIG)
    agent = agent.train(jax.random.PRNGKey(1), env)


def test_rice_jax_env():
    try:
        from rice_jax import Rice  # pyright: ignore
        from rice_jax.util import load_region_yamls  # pyright: ignore
    except ImportError:
        pytest.skip("rice_jax is not installed.")

    region_yamls = load_region_yamls(3)
    env = Rice(region_yamls)
    agent = jymkit.algorithms.PPO(**TEST_CONSTS.PPO_MIN_CONFIG)
    agent = agent.train(jax.random.PRNGKey(1), env)
