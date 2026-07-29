import logging
import warnings
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from ._transition import Transition

logger = logging.getLogger(__name__)


class TransitionBuffer(eqx.Module):
    """
    A buffer for storing transitions and sampling contiguous sequences from them
    (or single transitions in the case of ``n_steps=1``).
    Samples uniformly from valid sequence start positions in the buffer.
    The buffer is implemented as a circular buffer, where the oldest transitions are
    overwritten when the buffer is full.

    **Arguments**:
        `max_size`: The maximum number of transitions the buffer can hold (across all envs).
        `sample_batch_size`: The number of sequences to sample from the buffer.
        `data_sample`: A sample `Transition` to initialize the buffer structure.
        `vectorized_env`: Whether the inserted transitions are vectorized environment rollouts.
            If True, inserted transitions are assumed to be of shape (insert_batch_size, num_envs, ...).
            data is stored as is, but at sampling time, the num_envs axis is collapsed into the batch dimension.
        `n_steps`: Length of each sampled sequence. ``1`` reproduces single-transition
            sampling with output shape ``(sample_batch_size, ...)``. For ``n_steps > 1``,
            output shape is ``(sample_batch_size, n_steps, ...)``.
    """

    data: Transition
    insert_position: int
    size: int
    max_size: int = eqx.field(static=True)
    max_size_per_env: int = eqx.field(static=True)
    sample_batch_size: int = eqx.field(static=True)
    vectorized_env: bool = eqx.field(static=True)
    num_vec_envs: int | None = eqx.field(static=True)
    n_steps: int = eqx.field(static=True)

    def __check_init__(self):
        assert self.sample_batch_size > 0, "sample_batch_size must be greater than 0"
        assert self.max_size > 0, "max_size must be greater than 0"
        assert self.n_steps >= 1, "n_steps must be at least 1"
        assert self.n_steps <= self.max_size_per_env, (
            "n_steps must be less than or equal to max_size_per_env (max_size // num_vec_envs)"
        )
        assert self.max_size_per_env > 0, "max_size_per_env must be greater than 0"
        assert self.sample_batch_size <= (self.num_vec_envs or 1) * (
            self.max_size_per_env - self.n_steps + 1
        ), (
            "sample_batch_size must not exceed the number of valid sequence start "
            "positions: num_vec_envs * (max_size_per_env - n_steps + 1)"
        )

    def __init__(
        self,
        max_size: int,
        sample_batch_size: int,
        data_sample: Transition,
        vectorized_env: bool = True,
        n_steps: int = 1,
        **kwargs,
    ):
        self.insert_position = 0
        self.max_size = max_size
        self.sample_batch_size = sample_batch_size
        self.size = 0
        self.vectorized_env = vectorized_env
        self.n_steps = n_steps
        if kwargs.get("num_batch_axes", False):
            warnings.warn(
                "num_batch_axes no longer has an effect. "
                "Insertions are always assumed to be batches. "
                "control vectorized environment behaviour by setting `vectorized_env`."
            )

        if self.vectorized_env:
            num_vec_envs = jax.tree.leaves(data_sample)[0].shape[1]
            self.num_vec_envs = num_vec_envs
            self.max_size_per_env = self.max_size // num_vec_envs
            effective_max_size = self.max_size_per_env * num_vec_envs
            if effective_max_size != self.max_size:
                logger.warning(
                    f"max_size {self.max_size} is not divisible by the number of vectorized environments {num_vec_envs}. "
                    f"Setting max_size to {effective_max_size}."
                )
                self.max_size = effective_max_size
        else:
            self.num_vec_envs = None
            self.max_size_per_env = self.max_size

        self.data = jax.tree.map(
            lambda x: jnp.zeros((self.max_size_per_env,) + x.shape[1:], dtype=x.dtype),
            data_sample,
        )

    def _destination_indices(self, transition: Transition) -> jnp.ndarray:
        """Time-axis slot indices that `transition` will be written to based on the current insert position."""
        data_len = jax.tree.leaves(transition)[0].shape[0]
        assert data_len <= self.max_size_per_env, (
            "Transition length exceeds per-env buffer size. "
            f"Transition length: {data_len}, Per-env buffer size: {self.max_size_per_env} (max_size // num_vec_envs)"
        )
        return (jnp.arange(data_len) + self.insert_position) % self.max_size_per_env

    @eqx.filter_jit
    def insert(self, transition: Transition) -> Self:
        """
        Insert a batch of transitions into the buffer.

        Possible vectorized environments are stored as is (to preserve sequences)
        but flattened into a single batch dimension at sampling time.
        """
        data = self.data
        idx = self._destination_indices(transition)
        data_len = idx.shape[0]

        data = jax.tree.map(
            lambda x, y: x.at[idx].set(y, unique_indices=True),
            data,
            transition,
        )

        insert_position = (self.insert_position + data_len) % self.max_size_per_env
        size = jnp.minimum(self.size + data_len, self.max_size_per_env)
        buffer = self
        buffer = eqx.tree_at(lambda x: x.data, buffer, data)
        buffer = eqx.tree_at(lambda x: x.insert_position, buffer, insert_position)
        buffer = eqx.tree_at(lambda x: x.size, buffer, size)
        return buffer

    def sample(self, key: PRNGKeyArray, with_replacement: bool = True) -> Transition:
        """
        Sample a batch of transitions from the buffer. Samples a batch of sequences
        of length `n_steps` when `n_steps > 1`, otherwise a batch of single transitions.

        When `n_steps=1`, returns shape `(sample_batch_size, ...)`.
        When `n_steps > 1`, returns shape `(sample_batch_size, n_steps, ...)`.

        When `vectorized_env` is True, the vectorized environment axis is collapsed into
        the batch dimension of the returned transitions.
        """

        flat_valid_start_indices = self._get_flat_valid_start_indices()

        if with_replacement:
            probs = flat_valid_start_indices.astype(jnp.float32) / jnp.maximum(
                flat_valid_start_indices.sum(), 1
            )
            flat_indices = jax.random.choice(
                key, self.max_size, (self.sample_batch_size,), p=probs, replace=True
            )
        else:
            id_probs = jax.random.uniform(key, (self.max_size,))
            id_probs = jnp.where(flat_valid_start_indices, id_probs, -jnp.inf)
            flat_indices = jax.lax.top_k(id_probs, self.sample_batch_size)[1]

        batch = self._gather_batch(flat_indices)

        return batch

    def _get_flat_valid_start_indices(self) -> jnp.ndarray:
        """Boolean mask over time indices that can start a valid sequence."""
        start_indices = jnp.arange(self.max_size_per_env)
        if self.n_steps == 1:
            # All sequences are valid as long as the data is written to (index < size)
            valid_indices = start_indices < self.size

        else:  # Else, we need to check for valid start indices
            # While the buffer is not full, data is valid until the end of the so far written buffer.
            not_full = self.size < self.max_size_per_env
            valid_not_full = start_indices + self.n_steps <= self.size

            # When full, exclude windows that cross the circular write seam.
            # This is the data that we are about to write to (and is therefor from another rollout)
            window_idx = (
                start_indices[:, None] + jnp.arange(self.n_steps)
            ) % self.max_size_per_env
            prev_idx = (self.insert_position - 1) % self.max_size_per_env
            crosses_seam = jnp.any(window_idx == prev_idx, axis=1) & jnp.any(
                window_idx == self.insert_position, axis=1
            )
            valid_full = ~crosses_seam

            valid_indices = jax.lax.select(not_full, valid_not_full, valid_full)

        # Broadcast valid indices to each environment stream and flatten
        valid_indices = jnp.broadcast_to(
            valid_indices[:, None], (self.max_size_per_env, self.num_vec_envs or 1)
        ).reshape(-1)

        return valid_indices

    def _gather_batch(self, flat_indices: Array) -> Transition:
        step_indices, env_indices = self._flat_to_step_and_env_indices(flat_indices)

        window_idx = (
            step_indices[:, None] + jnp.arange(self.n_steps)[None, :]
        ) % self.max_size_per_env

        if self.vectorized_env:
            batch: Transition = jax.tree.map(
                lambda x: x[window_idx, env_indices[:, None]],
                self.data,
            )
        else:
            batch = jax.tree.map(lambda x: x[window_idx], self.data)

        if self.n_steps == 1:
            batch = jax.tree.map(lambda x: x[:, 0], batch)

        return batch

    def _flat_to_step_and_env_indices(self, flat_indices: Array) -> tuple[Array, Array]:
        num_vec_envs = self.num_vec_envs or 1
        step_indices = flat_indices // num_vec_envs
        env_indices = flat_indices % num_vec_envs
        return step_indices, env_indices


class PrioritizedTransitionBuffer(TransitionBuffer):
    """
    A circular buffer with Prioritized Experience Replay

    **Additional arguments**:
        `alpha`: Exponent controlling how strongly priorities bias sampling.
            `0` recovers uniform sampling, `1` samples fully proportionally.
        `beta`: Exponent for the importance-sampling correction. May be annealed externally.
        `eps`: Small constant added to `|TD-error|` so no transition ever
            gets a zero probability of being sampled.
    """

    priorities: Array
    max_priority: Float[Array, " "]
    alpha: float
    beta: float
    eps: float = eqx.field(static=True)

    def __init__(
        self,
        max_size: int,
        sample_batch_size: int,
        data_sample: Transition,
        vectorized_env: bool = True,
        n_steps: int = 1,
        alpha: float = 0.6,
        beta: float = 0.4,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(
            max_size=max_size,
            sample_batch_size=sample_batch_size,
            data_sample=data_sample,
            vectorized_env=vectorized_env,
            n_steps=n_steps,
            **kwargs,
        )
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        if self.vectorized_env:
            self.priorities = jnp.zeros(
                (self.max_size_per_env, self.num_vec_envs), dtype=jnp.float32
            )
        else:
            self.priorities = jnp.zeros((self.max_size_per_env,), dtype=jnp.float32)
        self.max_priority = jnp.array(1.0, dtype=jnp.float32)

    def insert(self, transition: Transition) -> Self:
        """
        Insert a transition into the buffer.
        Maximum priority is assigned to the new insertion such that it is sampled at least once.
        """
        idx = self._destination_indices(transition)
        buffer = super().insert(transition)

        new_priorities = buffer.priorities.at[idx].set(self.max_priority)
        buffer = eqx.tree_at(lambda b: b.priorities, buffer, new_priorities)

        return buffer

    def sample(self, key: PRNGKeyArray, with_replacement: bool = False) -> Transition:
        """
        Sample a batch of transitions from the buffer. Samples a batch of sequences
        of length ``n_steps`` when ``n_steps > 1``, otherwise a batch of single transitions.

        When `n_steps=1`, returns shape `(sample_batch_size, ...)`.
        When `n_steps > 1`, returns shape `(sample_batch_size, n_steps, ...)`.

        When `vectorized_env` is True, the vectorized environment axis is collapsed into
        the batch dimension of the returned transitions.

        Alongside the Transition batch, this PER adds the PER_weights
        to the Transition batch and returns the indices of the sampled sequence starts as
        ``(Transition, flat_indices)``.
        """
        flat_valid_start_indices = self._get_flat_valid_start_indices()
        flat_priorities = self.priorities.reshape(-1)
        scaled_priorities = jnp.where(
            flat_valid_start_indices, flat_priorities**self.alpha, 0.0
        )
        probs = scaled_priorities / jnp.maximum(scaled_priorities.sum(), 1e-8)

        if with_replacement:
            flat_indices = jax.random.choice(
                key, self.max_size, (self.sample_batch_size,), p=probs, replace=True
            )
        else:
            gumbel = jax.random.gumbel(key, (self.max_size,))
            keys_ = jnp.where(
                flat_valid_start_indices, jnp.log(probs + 1e-12) + gumbel, -jnp.inf
            )
            flat_indices = jax.lax.top_k(keys_, self.sample_batch_size)[1]

        num_valid = flat_valid_start_indices.sum()
        sample_probs = probs[flat_indices]
        weights = (num_valid * sample_probs) ** (-self.beta)
        weights = weights / jnp.maximum(weights.max(), 1e-8)

        batch = self._gather_batch(flat_indices)

        if isinstance(batch, Transition):
            batch = batch.replace(PER_weight=weights, PER_index=flat_indices)
        else:
            batch = jax.tree.map(
                lambda transition: transition.replace(
                    PER_weight=weights, PER_index=flat_indices
                ),
                batch,
                is_leaf=lambda x: isinstance(x, Transition),
            )

        return batch

    def update_priorities(self, indices: Array, td_errors: Array) -> Self:
        """
        Refresh the priorities of the given buffer indices from new TD errors.

        **Arguments**:
            `indices`: Flat buffer start indices to update, i.e. the `indices` returned by `sample`.
            `td_errors`: TD errors (or any priority signal) for those indices.
        """
        priorities = jnp.abs(td_errors) + self.eps
        step_indices, env_indices = self._flat_to_step_and_env_indices(indices)
        if self.vectorized_env:
            new_priorities = self.priorities.at[step_indices, env_indices].set(
                priorities
            )
        else:
            new_priorities = self.priorities.at[step_indices].set(priorities)
        max_priority = jnp.maximum(self.max_priority, priorities.max())

        buffer = eqx.tree_at(lambda b: b.priorities, self, new_priorities)
        buffer = eqx.tree_at(lambda b: b.max_priority, buffer, max_priority)
        return buffer

    def update_beta(self, beta: float) -> Self:
        """Update the beta parameter of the buffer."""
        buffer = eqx.tree_at(lambda b: b.beta, self, beta)
        return buffer
