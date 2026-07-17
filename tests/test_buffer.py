import jax
import jax.numpy as jnp
import pytest

from jaxnasium.algorithms.core import (
    PrioritizedTransitionBuffer,
    Transition,
    TransitionBuffer,
)

AGENTS = ("agent0", "agent1")
SHORT_LEN = 2
LONG_LEN = 4


def _make_single_and_multi_agent_transition(
    length: int,
    *,
    num_envs: int = 2,
    feature_dim: int = 3,
):
    time = jnp.arange(length)

    single_agent_transition = Transition(
        observation=jnp.ones((length, num_envs, feature_dim)) * time[:, None, None],
        action=jnp.ones((length, num_envs)) * time[:, None],
        reward=jnp.ones((length, num_envs)) * time[:, None],
        terminated=jnp.zeros((length, num_envs), dtype=bool),
        truncated=jnp.zeros((length, num_envs), dtype=bool),
    )

    multi_agent_transition = jax.tree.map(  # a copy per agent
        lambda x: single_agent_transition, {"agent0": 0, "agent1": 0}
    )

    return single_agent_transition, multi_agent_transition


@pytest.mark.parametrize("n_steps", [1, 3])
@pytest.mark.parametrize("with_replacement", [True, False])
def test_buffer_incremental_fill_samples_valid_region_sa(
    with_replacement: bool, n_steps: int
) -> None:
    """Test if samples stay within valid buffer range (single agent)."""
    max_size = 100
    chunk_size = max_size // 10
    transition, _ = _make_single_and_multi_agent_transition(max_size)

    buffer = TransitionBuffer(
        max_size=max_size,
        sample_batch_size=chunk_size,
        data_sample=transition,
        n_steps=n_steps,
    )
    for i in range(4):
        start_index = i * chunk_size
        chunk_transition = jax.tree.map(
            lambda x: x[start_index : start_index + chunk_size], transition
        )
        buffer = buffer.insert(chunk_transition)

        for k in range(10):
            samples = buffer.sample(
                jax.random.PRNGKey(k), with_replacement=with_replacement
            )
            assert jnp.all(jnp.ravel(samples.reward) < start_index + chunk_size)


@pytest.mark.parametrize("n_steps", [1, 3])
@pytest.mark.parametrize("with_replacement", [True, False])
def test_buffer_incremental_fill_samples_valid_region_ma(
    with_replacement: bool, n_steps: int
) -> None:
    """Test if samples stay within valid buffer range (multi agent)."""
    max_size = 100
    chunk_size = max_size // 10
    _, transition = _make_single_and_multi_agent_transition(max_size)

    buffer = TransitionBuffer(
        max_size=max_size,
        sample_batch_size=chunk_size,
        data_sample=transition,
        n_steps=n_steps,
    )
    for i in range(4):
        start_index = i * chunk_size
        chunk_transition = jax.tree.map(
            lambda x: x[start_index : start_index + chunk_size], transition
        )
        buffer = buffer.insert(chunk_transition)

        for k in range(10):
            samples = buffer.sample(
                jax.random.PRNGKey(k), with_replacement=with_replacement
            )
            assert jnp.all(
                jnp.ravel(samples["agent0"].reward) < start_index + chunk_size  # type: ignore
            )


def test_per_update_priorities_sa() -> None:
    transition, _ = _make_single_and_multi_agent_transition(4, num_envs=2)
    buffer = PrioritizedTransitionBuffer(
        max_size=8,
        sample_batch_size=3,
        data_sample=transition,
    )
    buffer = buffer.insert(transition)

    batch = buffer.sample(jax.random.PRNGKey(0))
    assert batch.PER_index is not None
    td_errors = jnp.array([0.5, 1.0, 2.0], dtype=jnp.float32)
    buffer = buffer.update_priorities(batch.PER_index, td_errors)

    eps = 1e-6
    for flat_idx, td_error in zip(batch.PER_index, td_errors, strict=True):
        step = flat_idx // 2
        env = flat_idx % 2
        assert buffer.priorities[step, env] == pytest.approx(abs(td_error) + eps)


def test_per_update_priorities_ma() -> None:
    _, transition = _make_single_and_multi_agent_transition(4, num_envs=2)
    buffer = PrioritizedTransitionBuffer(
        max_size=8,
        sample_batch_size=3,
        data_sample=transition,
    )
    buffer = buffer.insert(transition)

    batch = buffer.sample(jax.random.PRNGKey(0))
    per_index = batch["agent0"].PER_index  # type: ignore
    assert per_index is not None
    assert jnp.array_equal(per_index, batch["agent1"].PER_index)  # type: ignore

    td_errors = jnp.array([0.5, 1.0, 2.0], dtype=jnp.float32)
    buffer = buffer.update_priorities(per_index, td_errors)

    eps = 1e-6
    for flat_idx, td_error in zip(per_index, td_errors, strict=True):
        step = flat_idx // 2
        env = flat_idx % 2
        assert buffer.priorities[step, env] == pytest.approx(abs(td_error) + eps)
