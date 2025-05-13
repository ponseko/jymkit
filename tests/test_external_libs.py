import jax
import pytest

import jymkit
import jymkit.algorithms

try:
    import econojax
except ImportError:
    econojax = None

try:
    import rice_jax  # pyright: ignore
except ImportError:
    rice_jax = None

try:
    import gymnax
except ImportError:
    gymnax = None


@pytest.mark.skipif(gymnax is None, reason="gymnax is not installed.")
def test_gymnax_pendulum():
    env = jymkit.make("Pendulum-v1")
    agent = jymkit.algorithms.PPO(num_envs=2, total_timesteps=1000, num_epochs=1)
    agent = agent.train(jax.random.PRNGKey(1), env)


@pytest.mark.skipif(gymnax is None, reason="gymnax is not installed.")
def test_gymnax_breakout():
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


@pytest.mark.skipif(econojax is None, reason="econojax is not installed.")
def test_econojax_env():
    from econojax import EconoJax  # pyright: ignore

    env = EconoJax(num_population=4)
    agent = jymkit.algorithms.PPO(num_envs=2, total_timesteps=1000, num_epochs=1)
    agent = agent.train(jax.random.PRNGKey(1), env)


@pytest.mark.skipif(rice_jax is None, reason="rice_jax is not installed.")
def test_rice_jax_env():
    from rice_jax import Rice  # pyright: ignore
    from rice_jax.util import load_region_yamls

    region_yamls = load_region_yamls(3)
    env = Rice(region_yamls)
    agent = jymkit.algorithms.PPO(num_envs=2, total_timesteps=1000, num_epochs=1)
    agent = agent.train(jax.random.PRNGKey(1), env)
