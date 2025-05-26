import importlib.util

import jax
import pytest

import jymkit
import jymkit.algorithms


def test_gymnax_breakout():
    if importlib.util.find_spec("gymnax") is None:
        pytest.skip("Gymnax is not installed.")
    env = jymkit.make("Breakout-MinAtar")
    agent = jymkit.algorithms.PPO(num_envs=2, total_timesteps=1000, num_epochs=1)

    # Expecting this to fail without flattening
    with pytest.raises(AssertionError) as excinfo:
        agent = agent.train(jax.random.PRNGKey(1), env)
    assert "Flatten the observations with `jymkit.FlattenObservationWrapper`" in str(
        excinfo.value
    )

    env = jymkit.FlattenObservationWrapper(env)
    agent = agent.train(jax.random.PRNGKey(1), env)


@pytest.mark.parametrize(
    "env_name",
    ["Snake-v1", "Game2048-v1", "Cleaner-v0", "Maze-v0"],
)
def test_jumanji_envs(env_name):
    if importlib.util.find_spec("jumanji") is None:
        pytest.skip("Jumanji is not installed.")
    env = jymkit.make(env_name)
    agent = jymkit.algorithms.PPO(num_envs=2, total_timesteps=1000, num_epochs=1)
    env = jymkit.FlattenObservationWrapper(env)
    agent = agent.train(jax.random.PRNGKey(1), env)


@pytest.mark.parametrize(
    "env_name",
    [
        "ant",
        "halfcheetah",
        "humanoid",
        "inverted_double_pendulum",
        "walker2d",
    ],
)
def test_brax_envs(env_name):
    if importlib.util.find_spec("brax") is None:
        pytest.skip("Brax is not installed.")
    env = jymkit.make(env_name)
    agent = jymkit.algorithms.PPO(num_envs=2, total_timesteps=1000, num_epochs=1)
    env = jymkit.FlattenObservationWrapper(env)
    agent = agent.train(jax.random.PRNGKey(1), env)


def test_econojax_env():
    try:
        from econojax import EconoJax  # pyright: ignore
    except ImportError:
        pytest.skip("Econojax is not installed.")

    env = EconoJax(num_population=4)
    agent = jymkit.algorithms.PPO(num_envs=2, total_timesteps=1000, num_epochs=1)
    agent = agent.train(jax.random.PRNGKey(1), env)


def test_rice_jax_env():
    try:
        from rice_jax import Rice  # pyright: ignore
        from rice_jax.util import load_region_yamls
    except ImportError:
        pytest.skip("rice_jax is not installed.")

    region_yamls = load_region_yamls(3)
    env = Rice(region_yamls)
    agent = jymkit.algorithms.PPO(num_envs=2, total_timesteps=1000, num_epochs=1)
    agent = agent.train(jax.random.PRNGKey(1), env)
