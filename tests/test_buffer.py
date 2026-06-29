import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from jaxnasium.algorithms.utils import Transition, TransitionBuffer
from jaxnasium.algorithms.utils._buffer import PrioritizedTransitionBuffer

AGENTS = ("agent0", "agent1")


def _make_single_agent_transition(
    length: int,
    num_envs: int = 1,
    *,
    value: float = 0.0,
    feature_dim: int = 3,
) -> Transition:
    return Transition(
        observation=jnp.full((length, num_envs, feature_dim), value),
        action=jnp.full((length, num_envs), value),
        reward=jnp.full((length, num_envs), value),
        terminated=jnp.zeros((length, num_envs), dtype=bool),
        truncated=jnp.zeros((length, num_envs), dtype=bool),
    )


def _make_multi_agent_transition(
    length: int,
    num_envs: int = 1,
    *,
    feature_dim: int = 2,
    encode: bool = False,
) -> Transition:
    """Multi-agent transition with per-agent PyTree fields and shared done flags."""

    def _field(agent_idx: int) -> jnp.ndarray:
        if encode:
            # Scalar code per (time, env): agent*10_000 + env*1_000 + time*10
            time = jnp.arange(length)[:, None]
            env = jnp.arange(num_envs)[None, :]
            encoded = agent_idx * 10_000 + env * 1_000 + time * 10
            return jnp.broadcast_to(encoded[..., None], (length, num_envs, feature_dim))
        return jnp.full((length, num_envs, feature_dim), float(agent_idx))

    return Transition(
        observation={agent: _field(i) for i, agent in enumerate(AGENTS)},  # type: ignore[arg-type]
        action={
            agent: jnp.full((length, num_envs), float(i))
            for i, agent in enumerate(AGENTS)
        },  # type: ignore[arg-type]
        reward={
            agent: jnp.full((length, num_envs), float(i))
            for i, agent in enumerate(AGENTS)
        },  # type: ignore[arg-type]
        terminated=jnp.zeros((length, num_envs), dtype=bool),
        truncated=jnp.zeros((length, num_envs), dtype=bool),
    )


def _assert_sample_batch_shapes(
    batch: Transition,
    *,
    sample_batch_size: int,
    n_steps: int,
    feature_dim: int = 3,
    multi_agent: bool = False,
) -> None:
    if multi_agent:
        assert batch.observation["agent0"].shape == (
            sample_batch_size,
            *([n_steps] if n_steps > 1 else []),
            feature_dim,
        )
        assert batch.action["agent1"].shape == (
            sample_batch_size,
            *([n_steps] if n_steps > 1 else []),
        )
    else:
        assert batch.observation.shape == (
            sample_batch_size,
            *([n_steps] if n_steps > 1 else []),
            feature_dim,
        )
        assert batch.reward.shape == (
            sample_batch_size,
            *([n_steps] if n_steps > 1 else []),
        )


def _assert_encoded_windows_are_contiguous(
    encoded_obs: jnp.ndarray,
    num_envs: int,
    n_steps: int,
) -> None:
    """Check n-step windows stay within one env and advance by one timestep."""
    env_id = encoded_obs // 1_000
    time_id = (encoded_obs % 1_000) // 10
    assert bool(jnp.all(env_id == env_id[:, :1]))
    if n_steps > 1:
        assert bool(jnp.all(jnp.diff(time_id, axis=1) == 1))


@pytest.mark.parametrize("num_envs", [1, 3])
@pytest.mark.parametrize("n_steps", [1, 2, 3])
@pytest.mark.parametrize("with_replacement", [False, True])
def test_sample_shapes_single_agent(num_envs, n_steps, with_replacement):
    rollout_len = 8
    max_size = 24 if num_envs > 1 else 8
    sample_batch_size = min(4, num_envs * (max_size // num_envs - n_steps + 1))

    data_sample = _make_single_agent_transition(1, num_envs)
    buffer = TransitionBuffer(
        max_size=max_size,
        sample_batch_size=sample_batch_size,
        data_sample=data_sample,
        n_steps=n_steps,
    )
    buffer = buffer.insert(
        _make_single_agent_transition(rollout_len, num_envs, value=1.0)
    )

    batch = buffer.sample(jax.random.PRNGKey(0), with_replacement=with_replacement)
    _assert_sample_batch_shapes(
        batch, sample_batch_size=sample_batch_size, n_steps=n_steps
    )


def test_max_size_rounds_down_for_vectorized_envs():
    buffer = TransitionBuffer(
        max_size=10,
        sample_batch_size=2,
        data_sample=_make_single_agent_transition(1, num_envs=3),
        n_steps=1,
    )
    assert buffer.max_size == 9
    assert buffer.max_size_per_env == 3
    assert buffer.num_vec_envs == 3
    assert buffer.data.observation.shape == (3, 3, 3)


def test_storage_keeps_env_axis_for_vectorized_rollouts():
    buffer = TransitionBuffer(
        max_size=12,
        sample_batch_size=2,
        data_sample=_make_single_agent_transition(1, num_envs=3),
        n_steps=1,
    )
    assert buffer.data.observation.shape == (4, 3, 3)
    buffer = buffer.insert(_make_single_agent_transition(4, num_envs=3, value=7.0))
    assert int(buffer.size) == 4
    assert jnp.allclose(buffer.data.observation[3, 2, 0], 7.0)


def test_multi_env_n_step_windows_stay_within_env():
    num_envs = 3
    time_len = 10
    transitions = Transition(
        observation=jnp.array(
            [[[e * 1_000 + t] for e in range(num_envs)] for t in range(time_len)],
            dtype=jnp.float32,
        ),
        action=jnp.zeros((time_len, num_envs)),
        reward=jnp.zeros((time_len, num_envs)),
        terminated=jnp.zeros((time_len, num_envs), dtype=bool),
        truncated=jnp.zeros((time_len, num_envs), dtype=bool),
    )
    buffer = TransitionBuffer(
        max_size=30,
        sample_batch_size=8,
        data_sample=_make_single_agent_transition(1, num_envs=num_envs),
        n_steps=3,
    )
    buffer = buffer.insert(transitions)

    for key in jax.random.split(jax.random.PRNGKey(1), 32):
        for with_replacement in (True, False):
            batch = buffer.sample(key, with_replacement=with_replacement)
            env_id = batch.observation[..., 0] // 1_000
            time_id = batch.observation[..., 0] % 1_000
            assert bool(jnp.all(env_id == env_id[:, :1]))
            assert bool(jnp.all(jnp.diff(time_id, axis=1) == 1))


def test_sequence_sample_is_contiguous_single_env():
    buffer = TransitionBuffer(
        max_size=6,
        sample_batch_size=1,
        data_sample=_make_single_agent_transition(1),
        n_steps=3,
    )
    transitions = Transition(
        observation=jnp.arange(6 * 3, dtype=jnp.float32).reshape(6, 1, 3),
        action=jnp.arange(6, dtype=jnp.float32).reshape(6, 1),
        reward=jnp.arange(6, dtype=jnp.float32).reshape(6, 1),
        terminated=jnp.zeros((6, 1), dtype=bool),
        truncated=jnp.zeros((6, 1), dtype=bool),
    )
    buffer = buffer.insert(transitions)
    batch = buffer.sample(jax.random.PRNGKey(2), with_replacement=True)

    obs = batch.observation[0, :, 0]
    assert jnp.allclose(obs, jnp.array([0.0, 3.0, 6.0])) or jnp.allclose(
        obs, jnp.array([3.0, 6.0, 9.0])
    )


def test_sequence_sample_does_not_cross_write_seam():
    buffer = TransitionBuffer(
        max_size=4,
        sample_batch_size=1,
        data_sample=_make_single_agent_transition(1),
        n_steps=2,
    )
    first = Transition(
        observation=jnp.array([[[0.0]], [[1.0]], [[2.0]], [[3.0]]]),
        action=jnp.array([[0.0], [1.0], [2.0], [3.0]]),
        reward=jnp.array([[0.0], [1.0], [2.0], [3.0]]),
        terminated=jnp.zeros((4, 1), dtype=bool),
        truncated=jnp.zeros((4, 1), dtype=bool),
    )
    second = Transition(
        observation=jnp.array([[[10.0]], [[11.0]]]),
        action=jnp.array([[10.0], [11.0]]),
        reward=jnp.array([[10.0], [11.0]]),
        terminated=jnp.zeros((2, 1), dtype=bool),
        truncated=jnp.zeros((2, 1), dtype=bool),
    )
    buffer = buffer.insert(first)
    buffer = buffer.insert(second)

    valid = buffer._get_flat_valid_start_indices()
    assert valid.shape == (buffer.max_size,)
    assert valid.tolist() == [True, False, True, True]

    for subkey in jax.random.split(jax.random.PRNGKey(3), 16):
        batch = buffer.sample(subkey, with_replacement=True)
        pairs = batch.observation[:, :, 0]
        invalid_pair = jnp.logical_and(
            jnp.abs(pairs[:, 0] - 11.0) < 1e-6,
            jnp.abs(pairs[:, 1] - 2.0) < 1e-6,
        )
        assert not jnp.any(invalid_pair)


def test_non_vectorized_env_buffer():
    data_sample = Transition(
        observation=jnp.zeros((1, 3)),
        action=jnp.zeros((1,)),
        reward=jnp.zeros((1,)),
        terminated=jnp.zeros((1,), dtype=bool),
        truncated=jnp.zeros((1,), dtype=bool),
    )
    buffer = TransitionBuffer(
        max_size=6,
        sample_batch_size=2,
        data_sample=data_sample,
        vectorized_env=False,
        n_steps=2,
    )
    assert buffer.num_vec_envs is None
    assert buffer.data.observation.shape == (6, 3)

    rollout = Transition(
        observation=jnp.arange(6 * 3, dtype=jnp.float32).reshape(6, 3),
        action=jnp.arange(6.0),
        reward=jnp.arange(6.0),
        terminated=jnp.zeros((6,), dtype=bool),
        truncated=jnp.zeros((6,), dtype=bool),
    )
    buffer = buffer.insert(rollout)
    batch = buffer.sample(jax.random.PRNGKey(4), with_replacement=True)
    assert batch.observation.shape == (2, 2, 3)


@pytest.mark.parametrize("n_steps", [1, 3])
@pytest.mark.parametrize("num_envs", [1, 2])
def test_multi_agent_pytree_insert_and_sample(n_steps, num_envs):
    sample_batch_size = 4
    max_size = 12 if num_envs == 1 else 16
    data_sample = _make_multi_agent_transition(1, num_envs)
    buffer = TransitionBuffer(
        max_size=max_size,
        sample_batch_size=sample_batch_size,
        data_sample=data_sample,
        n_steps=n_steps,
    )

    # Per-agent PyTree fields and shared done flags should share the same storage layout.
    assert buffer.data.observation["agent0"].shape == (
        buffer.max_size_per_env,
        num_envs,
        2,
    )
    assert buffer.data.terminated.shape == (buffer.max_size_per_env, num_envs)

    insert_len = min(8, buffer.max_size_per_env)
    buffer = buffer.insert(
        _make_multi_agent_transition(insert_len, num_envs, encode=True)
    )
    batch = buffer.sample(jax.random.PRNGKey(5), with_replacement=True)

    _assert_sample_batch_shapes(
        batch,
        sample_batch_size=sample_batch_size,
        n_steps=n_steps,
        feature_dim=2,
        multi_agent=True,
    )
    # Shared fields lose the env axis in the sample batch (collapsed into flat batch).
    assert batch.terminated.shape == (
        sample_batch_size,
        *([n_steps] if n_steps > 1 else []),
    )

    if n_steps > 1:
        _assert_encoded_windows_are_contiguous(
            batch.observation["agent0"][..., 0], num_envs, n_steps
        )


def test_multi_agent_and_multi_env_windows_do_not_mix_streams():
    num_envs = 3
    buffer = TransitionBuffer(
        max_size=30,
        sample_batch_size=6,
        data_sample=_make_multi_agent_transition(1, num_envs),
        n_steps=3,
    )
    buffer = buffer.insert(_make_multi_agent_transition(10, num_envs, encode=True))

    for key in jax.random.split(jax.random.PRNGKey(6), 24):
        batch = buffer.sample(key, with_replacement=False)
        for agent in AGENTS:
            _assert_encoded_windows_are_contiguous(
                batch.observation[agent][..., 0], num_envs, n_steps=3
            )


def test_multi_agent_per_agent_done_flags():
    """Per-agent terminated/truncated PyTrees (same leading shape as other fields)."""
    length, num_envs = 6, 2
    data_sample = Transition(
        observation={agent: jnp.zeros((1, num_envs, 2)) for agent in AGENTS},  # type: ignore[arg-type]
        action={agent: jnp.zeros((1, num_envs)) for agent in AGENTS},  # type: ignore[arg-type]
        reward={agent: jnp.zeros((1, num_envs)) for agent in AGENTS},  # type: ignore[arg-type]
        terminated={agent: jnp.zeros((1, num_envs), dtype=bool) for agent in AGENTS},  # type: ignore[arg-type]
        truncated={agent: jnp.zeros((1, num_envs), dtype=bool) for agent in AGENTS},  # type: ignore[arg-type]
    )
    transition = Transition(
        observation={agent: jnp.ones((length, num_envs, 2)) for agent in AGENTS},  # type: ignore[arg-type]
        action={agent: jnp.zeros((length, num_envs)) for agent in AGENTS},  # type: ignore[arg-type]
        reward={agent: jnp.zeros((length, num_envs)) for agent in AGENTS},  # type: ignore[arg-type]
        terminated={
            agent: jnp.zeros((length, num_envs), dtype=bool) for agent in AGENTS
        },  # type: ignore[arg-type]
        truncated={
            agent: jnp.zeros((length, num_envs), dtype=bool) for agent in AGENTS
        },  # type: ignore[arg-type]
    )
    buffer = TransitionBuffer(
        max_size=12,
        sample_batch_size=3,
        data_sample=data_sample,
        n_steps=2,
    )
    buffer = buffer.insert(transition)
    batch = buffer.sample(jax.random.PRNGKey(7), with_replacement=True)

    assert batch.terminated["agent0"].shape == (3, 2)
    assert batch.truncated["agent1"].shape == (3, 2)


@pytest.mark.parametrize("multi_agent", [False, True])
def test_insert_and_sample_are_jittable(multi_agent):
    if multi_agent:
        data_sample = _make_multi_agent_transition(1, num_envs=2)
        make_rollout = lambda: _make_multi_agent_transition(4, num_envs=2)
        expected_shape = (2, 2, 2)  # batch, n_steps, feature
        leaf = lambda b: b.observation["agent0"]
    else:
        data_sample = _make_single_agent_transition(1, num_envs=2)
        make_rollout = lambda: _make_single_agent_transition(4, num_envs=2)
        expected_shape = (2, 2, 3)
        leaf = lambda b: b.observation

    buffer = TransitionBuffer(
        max_size=12,
        sample_batch_size=2,
        data_sample=data_sample,
        n_steps=2,
    )

    @eqx.filter_jit
    def step(buf, transition, key):
        buf = buf.insert(transition)
        return buf.sample(key)

    batch = step(buffer, make_rollout(), jax.random.PRNGKey(8))
    assert leaf(batch).shape == expected_shape


def test_per_priority_storage_shape():
    vec_data = _make_single_agent_transition(1, num_envs=3)
    vec_buffer = PrioritizedTransitionBuffer(
        max_size=12,
        sample_batch_size=2,
        data_sample=vec_data,
    )
    assert vec_buffer.priorities.shape == (4, 3)

    non_vec_data = Transition(
        observation=jnp.zeros((1, 3)),
        action=jnp.zeros((1,)),
        reward=jnp.zeros((1,)),
        terminated=jnp.zeros((1,), dtype=bool),
        truncated=jnp.zeros((1,), dtype=bool),
    )
    non_vec_buffer = PrioritizedTransitionBuffer(
        max_size=6,
        sample_batch_size=2,
        data_sample=non_vec_data,
        vectorized_env=False,
    )
    assert non_vec_buffer.priorities.shape == (6,)


def test_per_insert_sets_max_priority_on_written_slots():
    num_envs = 3
    buffer = PrioritizedTransitionBuffer(
        max_size=12,
        sample_batch_size=2,
        data_sample=_make_single_agent_transition(1, num_envs=num_envs),
    )
    buffer = buffer.insert(_make_single_agent_transition(2, num_envs=num_envs))

    assert jnp.allclose(buffer.priorities[0], buffer.max_priority)
    assert jnp.allclose(buffer.priorities[1], buffer.max_priority)
    assert buffer.priorities[2:].sum() == 0.0


@pytest.mark.parametrize("num_envs", [1, 3])
@pytest.mark.parametrize("n_steps", [1, 2])
@pytest.mark.parametrize("with_replacement", [False, True])
def test_per_sample_shapes(num_envs, n_steps, with_replacement):
    max_size = 24 if num_envs > 1 else 8
    sample_batch_size = min(4, num_envs * (max_size // num_envs - n_steps + 1))
    data_sample = _make_single_agent_transition(1, num_envs)

    buffer = PrioritizedTransitionBuffer(
        max_size=max_size,
        sample_batch_size=sample_batch_size,
        data_sample=data_sample,
        n_steps=n_steps,
    )
    buffer = buffer.insert(_make_single_agent_transition(8, num_envs))

    batch, weights, flat_indices = buffer.sample(
        jax.random.PRNGKey(0), with_replacement=with_replacement
    )

    _assert_sample_batch_shapes(
        batch, sample_batch_size=sample_batch_size, n_steps=n_steps
    )
    assert weights.shape == (sample_batch_size,)
    assert flat_indices.shape == (sample_batch_size,)


def test_per_weights_are_max_normalized():
    buffer = PrioritizedTransitionBuffer(
        max_size=8,
        sample_batch_size=4,
        data_sample=_make_single_agent_transition(1),
        alpha=0.6,
        beta=0.4,
    )
    buffer = buffer.insert(_make_single_agent_transition(8))

    _, weights, _ = buffer.sample(jax.random.PRNGKey(1), with_replacement=False)
    assert float(weights.max()) == pytest.approx(1.0)
    assert jnp.all(weights > 0)


def test_per_update_priorities_decodes_flat_indices():
    num_envs = 2
    buffer = PrioritizedTransitionBuffer(
        max_size=8,
        sample_batch_size=1,
        data_sample=_make_single_agent_transition(1, num_envs=num_envs),
        eps=1e-6,
    )
    buffer = buffer.insert(_make_single_agent_transition(4, num_envs))

    flat_index = jnp.array([5])  # step 2, env 1
    td_error = jnp.array([2.5])
    buffer = buffer.update_priorities(flat_index, td_error)

    assert buffer.priorities[2, 1] == pytest.approx(2.5 + 1e-6)
    assert buffer.priorities[2, 0] == pytest.approx(1.0)


def test_per_update_priorities_increases_max_priority():
    buffer = PrioritizedTransitionBuffer(
        max_size=4,
        sample_batch_size=1,
        data_sample=_make_single_agent_transition(1),
    )
    buffer = buffer.insert(_make_single_agent_transition(4))
    assert float(buffer.max_priority) == pytest.approx(1.0)

    buffer = buffer.update_priorities(jnp.array([0]), jnp.array([10.0]))
    assert float(buffer.max_priority) == pytest.approx(10.0)


def test_per_high_priority_indices_are_sampled_more():
    buffer = PrioritizedTransitionBuffer(
        max_size=8,
        sample_batch_size=1,
        data_sample=_make_single_agent_transition(1),
        alpha=1.0,
        beta=0.0,
    )
    buffer = buffer.insert(_make_single_agent_transition(8))

    flat_indices = jnp.arange(8)
    td_errors = jnp.zeros(8).at[3].set(100.0)
    buffer = buffer.update_priorities(flat_indices, td_errors)

    keys = jax.random.split(jax.random.PRNGKey(2), 64)
    sampled = []
    for key in keys:
        _, _, idx = buffer.sample(key, with_replacement=True)
        sampled.append(int(idx[0]))

    assert all(i == 3 for i in sampled)


def test_per_n_step_multi_env_windows_stay_within_env():
    num_envs = 3
    buffer = PrioritizedTransitionBuffer(
        max_size=30,
        sample_batch_size=6,
        data_sample=_make_multi_agent_transition(1, num_envs),
        n_steps=3,
        alpha=0.6,
        beta=0.4,
    )
    buffer = buffer.insert(_make_multi_agent_transition(10, num_envs, encode=True))

    for key in jax.random.split(jax.random.PRNGKey(3), 16):
        batch, _, _ = buffer.sample(key, with_replacement=False)
        for agent in ("agent0", "agent1"):
            _assert_encoded_windows_are_contiguous(
                batch.observation[agent][..., 0], num_envs, n_steps=3
            )


def test_per_insert_sample_update_are_jittable():
    buffer = PrioritizedTransitionBuffer(
        max_size=8,
        sample_batch_size=2,
        data_sample=_make_single_agent_transition(1, num_envs=2),
        n_steps=1,
    )

    @eqx.filter_jit
    def step(buf, transition, key):
        buf = buf.insert(transition)
        batch, weights, indices = buf.sample(key, with_replacement=True)
        buf = buf.update_priorities(indices, jnp.ones(indices.shape[0]))
        return batch, weights

    batch, weights = step(
        buffer,
        _make_single_agent_transition(4, num_envs=2),
        jax.random.PRNGKey(4),
    )
    assert batch.observation.shape == (2, 3)
    assert weights.shape == (2,)
