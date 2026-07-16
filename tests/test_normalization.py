import jax
import jax.numpy as jnp
import pytest
from _proxy_test_envs import REPRESENTATIVE_ENVS, observation_arrays, rollout

from jaxnasium import AgentObservation
from jaxnasium.algorithms.core import Normalizer
from jaxnasium.algorithms.core._normalization import RunningStatisticsState

SEED = jax.random.PRNGKey(0)
BATCH_SIZE = 512


@pytest.mark.parametrize(
    "env", REPRESENTATIVE_ENVS.values(), ids=REPRESENTATIVE_ENVS.keys()
)
def test_obs_normalization_standardizes_and_preserves_structure(env):
    obs_batch = jax.vmap(env.sample_observation)(jax.random.split(SEED, BATCH_SIZE))

    normalizer = Normalizer(
        obs_space=env.observation_space, normalize_obs=True, normalize_rew=False
    )
    normalizer = normalizer.update_obs(obs_batch)
    normalized = normalizer.normalize_obs(obs_batch)

    assert jax.tree.structure(normalized) == jax.tree.structure(obs_batch)

    # Each observation leaf is standardized to ~zero mean / unit std over the batch.
    for leaf in observation_arrays(normalized):
        leaf = leaf.astype(jnp.float32)
        assert jnp.allclose(jnp.mean(leaf, axis=0), 0.0, atol=1e-3)
        assert jnp.allclose(jnp.std(leaf, axis=0), 1.0, atol=1e-2)


def test_obs_normalization_preserves_action_mask():
    env = REPRESENTATIVE_ENVS["masked_discrete"]
    obs_batch = jax.vmap(env.sample_observation)(jax.random.split(SEED, BATCH_SIZE))

    normalizer = Normalizer(
        obs_space=env.observation_space, normalize_obs=True, normalize_rew=False
    )
    normalizer = normalizer.update_obs(obs_batch)
    normalized = normalizer.normalize_obs(obs_batch)

    assert isinstance(normalized, AgentObservation)
    assert normalized.action_mask is not None
    assert jnp.array_equal(normalized.action_mask, obs_batch.action_mask)
    assert not jnp.allclose(normalized.observation, obs_batch.observation)


def test_reward_normalization_runs_and_scales():
    env = REPRESENTATIVE_ENVS["vector_discrete"]
    num_steps, num_envs = 16, 4
    transition = rollout(env, SEED, num_steps=num_steps, num_envs=num_envs)

    normalizer = Normalizer(
        normalize_obs=False,
        normalize_rew=True,
        gamma=0.99,
        rew_shape=(num_steps, num_envs),
    )
    normalizer = normalizer.update(transition)
    normalized = normalizer.normalize_reward(transition.reward)

    assert normalized.shape == transition.reward.shape
    assert jnp.all(jnp.isfinite(normalized))
    assert not jnp.allclose(normalized, transition.reward)


def test_normalizer_is_noop_when_disabled():
    env = REPRESENTATIVE_ENVS["vector_discrete"]
    transition = rollout(env, SEED, num_steps=8, num_envs=2)

    normalizer = Normalizer(normalize_obs=False, normalize_rew=False)
    normalizer = normalizer.update(transition)

    assert jnp.array_equal(
        normalizer.normalize_obs(transition.observation), transition.observation
    )
    assert jnp.array_equal(
        normalizer.normalize_reward(transition.reward), transition.reward
    )


def test_running_statistics_matches_batch_moments():
    data = jax.random.normal(SEED, (1000, 4)) * 3.0 + 5.0

    stats = RunningStatisticsState(jnp.zeros(4))
    stats = stats.update(data)

    assert jnp.allclose(stats.mean, jnp.mean(data, axis=0), atol=1e-3)
    assert jnp.allclose(stats.std, jnp.std(data, axis=0), atol=1e-3)

    normalized = stats.normalize(data)
    assert jnp.allclose(jnp.mean(normalized, axis=0), 0.0, atol=1e-3)
    assert jnp.allclose(jnp.std(normalized, axis=0), 1.0, atol=1e-2)


def test_running_statistics_incremental_matches_single_batch():
    data = jax.random.normal(SEED, (1000, 3))

    single = RunningStatisticsState(jnp.zeros(3)).update(data)
    incremental = RunningStatisticsState(jnp.zeros(3))
    for chunk in data.reshape(10, 100, 3):
        incremental = incremental.update(chunk)

    assert jnp.allclose(single.mean, incremental.mean, atol=1e-4)
    assert jnp.allclose(single.std, incremental.std, atol=1e-4)
