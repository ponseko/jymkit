import _consts as TEST_CONSTS
import jax
import pytest

import jymkit
import jymkit.algorithms


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
