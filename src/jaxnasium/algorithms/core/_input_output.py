import logging
from functools import partial
from typing import Any, Callable, List, Literal, Protocol, Sequence

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, PRNGKeyArray, PyTree

import jaxnasium as jym

from ..architectures import CNN, Identity
from ._distributions import TanhNormalFactory

logger = logging.getLogger(__name__)


class SpaceLike(Protocol):
    shape: tuple[int, ...]
    sample: Callable[[PRNGKeyArray], Array]
    dtype: jnp.dtype


class DiscreteSpaceLike(SpaceLike, Protocol):
    n: int | None = None
    nvec: Sequence[int] | None = None


class ContinuousSpaceLike(SpaceLike, Protocol):
    low: Array
    high: Array


class Network(Protocol):
    """Any module with a __call__ defined"""

    def __call__(self, *args, **kwargs) -> Any: ...


def _is_space_discrete(space: SpaceLike) -> bool:
    return hasattr(space, "n") or hasattr(space, "nvec")


def _is_space_continuous(space: SpaceLike) -> bool:
    return hasattr(space, "low") and hasattr(space, "high")


def _is_callable_module(x) -> bool:
    """Check if x is a callable eqx.Module."""
    return isinstance(x, eqx.Module) and callable(x)


def _is_distribution(x: Any) -> bool:
    return isinstance(x, distrax.Distribution)


def _make_independent(dist: distrax.Distribution) -> distrax.Distribution:
    """Wraps a distrax distribution in an Independent distribution if the
    output space is multi-dimensional and sets the event shape accordingly."""
    ndims = len(dist.batch_shape)
    if ndims == 0:
        return dist  # Discrete, MultiDiscrete([n]), scalar Box
    return distrax.Independent(dist, reinterpreted_batch_ndims=ndims)


def _assert_homogeneous_output_space(num_outputs: List[int]):
    assert len(set(num_outputs)) == 1, (
        "Only homogeneous multi-dimensional output spaces supported to be supported for vmap."
        f" (all nvec elements must be the same, got {num_outputs}) "
        "For heterogeneous spaces, use a composite of spaces instead. "
        "(E.g. {'action1': (Discrete(n), 'action2': Discrete(m), ...}) "
        "This can be done through `jym.MultiDiscreteToListDiscreteWrapper` "
        "or by flattening the action space using `jym.FlattenActionWrapper`."
    )


def _apply_action_mask(logits: Array, action_mask: Array) -> Array:
    """Mask out invalid actions by pushing their logits/values to -inf.

    NOTE: This requires a (multi-)discrete output space.
    NOTE: The mask is assumed to be a PyTree of the same structure as the
        output space. Masking dependent on another action is not supported.
    """
    try:
        BIG_NEGATIVE = -1e9
        return jax.tree.map(
            lambda a, mask: a + (BIG_NEGATIVE * (1 - mask)),
            logits,
            action_mask,
        )
    except Exception as e:
        logger.error(f"Failed to apply action mask: {e}")
        raise ValueError(
            "Failed to apply action mask with the above error. "
            "logits and action_mask must have the same pytree structure and shapes. "
            f"logits: {jax.tree.structure(logits)} with shapes {jax.tree.map(lambda x: x.shape, logits)} "
            f"action_mask: {jax.tree.structure(action_mask)} with shapes {jax.tree.map(lambda x: x.shape, action_mask)}."
            "Further note that action masking is not supported for actions which are conditionally dependent. "
            "In this case, flatten the action space (e.g. using a `jym.FlattenActionWrapper`) or use a custom model."
        )


def _resolve_discrete_distribution(
    distribution: Literal["categorical"],
    dtype: jnp.dtype = jnp.int32,
) -> Callable[..., distrax.Distribution]:
    if distribution == "categorical":
        return partial(distrax.Categorical, dtype=dtype)
    raise ValueError(f"Unsupported discrete distribution: {distribution}")


def _resolve_continuous_distribution(
    distribution: Literal["normal", "tanhnormal"], low: np.ndarray, high: np.ndarray
) -> Callable[..., distrax.Distribution]:
    if distribution == "normal":
        return distrax.Normal
    if distribution == "tanhnormal":
        return TanhNormalFactory(low=low, high=high)
    raise ValueError(f"Unsupported continuous distribution: {distribution}")


class FlattenLayer(eqx.Module):
    """Flattens the input array to a 1D vector."""

    def __call__(self, x: Array, *args, **kwargs) -> Array:
        return x.reshape(-1)


class PyTreeFlattenLayer(eqx.Module):
    """Flattens every leaf of a pytree into a single 1D vector."""

    def __call__(self, x: Any, *args, **kwargs) -> Array:
        leaves = jax.tree.leaves(x)
        flat = [jnp.ravel(jnp.asarray(leaf, dtype=jnp.float32)) for leaf in leaves]
        return jnp.concatenate(flat)


class PyTreeObsSpaceNetwork(eqx.Module):
    """Builds a separate observation network for each observation space.

    Automatically builds a given 1d architecture for 1d observation spaces
    and a given 2d architecture for 2d observation spaces.

    During a forward call this network simply returns a jax.tree.map over all observation spaces
    and concatenates the outputs of all observation networks as a single 1d vector.
    """

    networks: PyTree[Network]

    num_observation_spaces: int = eqx.field(static=True)
    input_structure: Any = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    def __init__(
        self,
        obs_space: PyTree[SpaceLike],
        *,
        key: PRNGKeyArray,
        architecture_1d: Callable[..., Network] = Identity,
        architecture_2d: Callable[..., Network] = CNN.with_params(
            out_channels=(32, 64, 64),
            kernel_sizes=(3, 3, 2),
            strides=(1, 1, 1),
            padding=(0, 0, 0),
        ),
        network_kwargs_1d: dict[str, Any] | None = {},
        network_kwargs_2d: dict[str, Any] | None = {},
    ):
        kwargs_1d = network_kwargs_1d or {}
        kwargs_2d = network_kwargs_2d or {}

        def create_obs_processor(key: PRNGKeyArray, obs_space: SpaceLike):
            if obs_space.shape == () or len(obs_space.shape) == 1:
                return self._create_1d_obs_processor(
                    key, obs_space, architecture_1d, kwargs_1d
                )
            elif len(obs_space.shape) == 3:
                return self._create_2d_obs_processor(
                    key, obs_space, architecture_2d, kwargs_2d
                )
            elif len(obs_space.shape) == 2:
                raise ValueError(
                    f"2D observation space shape without a channel axis ({obs_space.shape}) detected. "
                    "Either add a channel axis or use a FlattenObservationWrapper."
                )
            raise ValueError(f"Unsupported observation space shape: {obs_space.shape}")

        # Exclude action mask from the observation space if present
        obs_space = jax.tree.map(
            lambda o: o.observation if isinstance(o, jym.AgentObservation) else o,
            obs_space,
            is_leaf=lambda o: isinstance(o, jym.AgentObservation),
        )

        # For continuous action space where a Q network is used, the action is included
        # in the observation space. We process it seperately by flattening only.
        action_input = None
        original_obs_space = obs_space
        if isinstance(obs_space, dict):
            original_obs_space = obs_space.copy()
            action_input = obs_space.pop("_ACTION", None)

        self.num_observation_spaces = len(jax.tree.leaves(obs_space))
        self.input_structure = jax.tree.structure(obs_space)

        keys = optax.tree.split_key_like(key, obs_space)
        self.networks = jax.tree.map(
            lambda o, k: create_obs_processor(k, o),
            obs_space,
            keys,
        )

        # Add back the action input if present
        if action_input is not None:
            action_input_layer = PyTreeFlattenLayer()
            self.networks["_ACTION"] = action_input_layer

        # Infer the output feature size
        dummy_obs = jax.tree.map(lambda o: jnp.zeros(o.shape), original_obs_space)
        f = lambda obs: jnp.atleast_1d(self(obs))
        self.out_features = jax.eval_shape(f, dummy_obs).shape[0]

    def __call__(self, x, *, key: PRNGKeyArray | None = None):
        x = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), x)

        outputs = jax.tree.map(
            lambda layer, x: layer(x, key=key),
            self.networks,
            x,
            is_leaf=_is_callable_module,
        )
        return jym.tree.concatenate(outputs)

    def _create_1d_obs_processor(
        self,
        key: PRNGKeyArray,
        obs_space: SpaceLike,
        architecture: Callable[..., Network],
        architecture_kwargs: dict[str, Any],
    ):
        try:
            if obs_space.shape == ():
                in_features = 1
            elif len(obs_space.shape) == 1:
                in_features = obs_space.shape[0]
            else:
                raise ValueError(
                    f"Unsupported observation space shape: {obs_space.shape}"
                )
            return architecture(in_features, key=key, **architecture_kwargs)
        except AttributeError:
            raise ValueError(f"Unsupported observation space {obs_space}")

    def _create_2d_obs_processor(
        self,
        key: PRNGKeyArray,
        obs_space: SpaceLike,
        architecture: Callable[..., Network],
        architecture_kwargs: dict[str, Any],
    ):
        try:
            if len(obs_space.shape) == 3:
                channels_axis = self._infer_channels_axis(obs_space)
                return architecture(
                    obs_space.shape,
                    key=key,
                    channels_axis=channels_axis,
                    **architecture_kwargs,
                )
            raise ValueError(f"Unsupported observation space shape: {obs_space.shape}")
        except AttributeError:
            raise ValueError(f"Unsupported observation space {obs_space}")

    def _infer_channels_axis(self, obs_space: SpaceLike):
        """
        Attempts to infer the channels axis from the observation space shape.
        By checking for the smallest dimension and assuming it to be the channels dimension.
        """
        if len(obs_space.shape) != 3:
            raise ValueError(
                "`infer_channels_axis` requires a (C, H, W) or (H, W, C) observation space."
            )
        c0, c1, c2 = obs_space.shape
        is_chw = c0 < min(c1, c2)
        is_hwc = c2 < min(c0, c1)
        if is_chw and not is_hwc:
            return "first"
        if is_hwc and not is_chw:
            return "last"
        raise ValueError(
            f"Cannot infer channel axis from shape {obs_space.shape}. "
            "Pass `channels_axis` explicitly  ('first' or 'last')."
        )


class DiscreteHead(eqx.Module):
    """Latent -> Categorical distribution or raw Q-values over a (multi-)discrete space.

    Produces one set of logits per output dimension and optionally applies an
    action mask.

    `distribution`="categorical": Produces a categorical distribution.
    `distribution=None`: Produces raw logits (e.g. for Q values).
    """

    layers: List[Network]
    distribution: Callable[..., distrax.Distribution] | None = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        output_space: DiscreteSpaceLike,
        *,
        key: PRNGKeyArray,
        distribution: Literal["categorical"] | None = "categorical",
        layer_type: Callable[..., Network] = eqx.nn.Linear,
    ):
        # Obtain the number of outputs per dimension: [n] (Discrete) or [n, n, ...] (MultiDiscrete)
        num_outputs = getattr(output_space, "n", getattr(output_space, "nvec", None))
        if num_outputs is None:
            raise ValueError(f"Unsupported discrete output space: {output_space}")
        num_outputs = np.atleast_1d(num_outputs).tolist()
        _assert_homogeneous_output_space(num_outputs)

        keys = optax.tree.split_key_like(key, num_outputs)
        self.layers = jax.tree.map(  # Create a (homegenuous) head per output dimension
            lambda o, k: layer_type(in_features, o, key=k), num_outputs, keys
        )

        self.distribution = (
            None
            if distribution is None
            else _resolve_discrete_distribution(distribution, dtype=output_space.dtype)
        )

    def __call__(self, x, action_mask=None, *, key: PRNGKeyArray | None = None):
        if len(self.layers) == 1:  # single-dimensional output space
            logits = self.layers[0](x, key=key)
        else:
            stacked_layers = jym.tree.stack(self.layers)
            logits = jax.vmap(lambda layer: layer(x, key=key))(stacked_layers)

        if action_mask is not None:
            logits = _apply_action_mask(logits, action_mask)
        if self.distribution is None:
            return logits  # raw Q-values
        return self.distribution(logits=logits)


class ContinuousHead(eqx.Module):
    """Latent -> Distribution over a continuous (`Box`-like) space.

    Per dimension, produces a mean and (log) std and returns a continuous
    distribution (`normal` or `tanhnormal`).
    """

    layers: List[Network]
    distribution: Callable[..., distrax.Distribution] = eqx.field(static=True)
    log_std_min: float = eqx.field(static=True)
    log_std_max: float = eqx.field(static=True)
    output_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        output_space: ContinuousSpaceLike,
        *,
        key: PRNGKeyArray,
        distribution: Literal["normal", "tanhnormal"] = "normal",
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        layer_type: Callable[..., Network] = eqx.nn.Linear,
    ):
        low = np.array(output_space.low, dtype=float)
        high = np.array(output_space.high, dtype=float)
        self.distribution = _resolve_continuous_distribution(distribution, low, high)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.output_shape = output_space.shape

        if self.output_shape == ():
            num_action_dims = 1
        else:
            num_action_dims = int(np.prod(self.output_shape))

        num_outputs = [2] * num_action_dims  # [mean, std] per dimension

        # Create a (homegenuous) head per output dimension
        keys = optax.tree.split_key_like(key, num_outputs)
        self.layers = jax.tree.map(
            lambda o, k: layer_type(in_features, o, key=k), num_outputs, keys
        )

    def __call__(self, x, action_mask=None, *, key: PRNGKeyArray | None = None):
        if action_mask is not None:
            logger.debug("Action mask provided for continuous space, ignoring.")

        if self.output_shape == ():
            out = self.layers[0](x, key=key)  # scalar output
        else:
            stacked_layers = jym.tree.stack(self.layers)
            out = jax.vmap(lambda layer: layer(x, key=key))(stacked_layers)

        mean = out[..., 0].reshape(self.output_shape)
        log_std = jnp.clip(
            out[..., 1].reshape(self.output_shape), self.log_std_min, self.log_std_max
        )
        std = jnp.exp(log_std)

        return self.distribution(mean, std)


class QHead(eqx.Module):
    """Convenience wrapper for Latent -> Q-values

    - `mode="discrete"`: wraps a `DiscreteHead` with `distribution=None`
      (one Q-value per action).
    - `mode="continuous"`: maps the input to a single scalar Q(s, a).
    Mode is inferred from the output space.
    """

    layer: Network
    mode: Literal["discrete", "continuous"] = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        output_space: SpaceLike,
        *,
        key: PRNGKeyArray,
        layer_type: Callable[..., Network] = eqx.nn.Linear,
    ):
        if _is_space_discrete(output_space):
            self.mode = "discrete"
            self.layer = DiscreteHead(
                in_features,
                output_space,  # type: ignore[arg-type]
                key=key,
                distribution=None,
                layer_type=layer_type,
            )
        elif _is_space_continuous(output_space):
            self.mode = "continuous"
            self.layer = layer_type(in_features, 1, key=key)  # Scalar output layer
        else:
            raise ValueError(f"Unsupported output space: {output_space}")

    def __call__(self, x, action_mask=None, *, key: PRNGKeyArray | None = None):
        if self.mode == "discrete":
            return self.layer(x, action_mask=action_mask, key=key)
        if action_mask is not None:
            logger.debug("Action mask provided for continuous space, ignoring.")
        return self.layer(x, key=key).squeeze()


class PyTreeOutputNetwork(eqx.Module):
    """Output network for a (single or PyTree of) output space(s).

    Builds one head per output-space leaf, selecting a discrete or continuous
    head automatically from the space. Heads either output a distribution or
    raw values (e.g. for Q-values) when `distribution` is None.

    A single head may itself be multi-dimensional (homogeneous),
    in which case its sub-layers are stacked and vmapped in the forward call.

    During a forward call this maps over all output spaces. When every head
    returns a distribution and `assume_independent` is set, per-space
    distributions are wrapped in `Independent` and, for multiple spaces,
    combined into a `distrax.Joint`. Otherwise the raw PyTree of outputs is
    returned unchanged.
    """

    heads: PyTree[Network]

    num_output_spaces: int = eqx.field(static=True)
    output_structure: Any = eqx.field(static=True)
    assume_independent: bool = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        output_space: PyTree[SpaceLike],
        *,
        key: PRNGKeyArray,
        discrete_distribution: Literal["categorical"] | None = "categorical",
        continuous_distribution: Literal["normal", "tanhnormal"] | None = "normal",
        layer_type: Callable[..., Network] = eqx.nn.Linear,
        assume_independent: bool = True,
    ):
        def create_head(key: PRNGKeyArray, space: SpaceLike):
            if _is_space_discrete(space):
                if discrete_distribution is None:
                    return QHead(
                        in_features,
                        space,
                        key=key,
                        layer_type=layer_type,
                    )
                return DiscreteHead(
                    in_features,
                    space,  # type: ignore[arg-type]
                    key=key,
                    distribution=discrete_distribution,
                    layer_type=layer_type,
                )
            elif _is_space_continuous(space):
                if continuous_distribution is None:
                    return QHead(
                        in_features,
                        space,
                        key=key,
                        layer_type=layer_type,
                    )
                return ContinuousHead(
                    in_features,
                    space,  # type: ignore[arg-type]
                    key=key,
                    distribution=continuous_distribution,
                    layer_type=layer_type,
                )
            raise ValueError(f"Unsupported output space: {space}")

        self.num_output_spaces = len(jax.tree.leaves(output_space))
        self.output_structure = jax.tree.structure(output_space)

        keys = optax.tree.split_key_like(key, output_space)
        self.heads = jax.tree.map(lambda o, k: create_head(k, o), output_space, keys)
        self.assume_independent = assume_independent

    @property
    def include_action_in_input(self) -> bool:
        """In case a we output a continuous Q-value, the action
        is required to be fed into the network as input."""

        output_heads = jax.tree.leaves(self.heads, is_leaf=_is_callable_module)
        has_continuous_q_head = any(
            isinstance(head, QHead) and head.mode == "continuous"
            for head in output_heads
        )
        only_continuous_q_heads = all(
            isinstance(head, QHead) and head.mode == "continuous"
            for head in output_heads
        )
        if has_continuous_q_head and not only_continuous_q_heads:
            logger.warning(
                "Mixed continuous and discrete Q-heads. This may have adverse training effects."
            )
        return has_continuous_q_head

    def __call__(self, x, action_mask=None, *, key: PRNGKeyArray | None = None):
        if action_mask is None:  # Dummy action mask if not provided
            action_mask = jax.tree.map(
                lambda _: None, self.heads, is_leaf=_is_callable_module
            )

        outputs = jax.tree.map(
            lambda head, mask: head(x, action_mask=mask, key=key),
            self.heads,
            action_mask,
            is_leaf=_is_callable_module,
        )

        # Policy heads return distributions; Q/value heads return raw arrays.
        dist_list = jax.tree.leaves(outputs, is_leaf=_is_distribution)
        if dist_list and all(_is_distribution(o) for o in dist_list):
            if self.assume_independent:
                outputs = jax.tree.map(
                    _make_independent, outputs, is_leaf=_is_distribution
                )
            if len(dist_list) > 1:
                return distrax.Joint(outputs)

        return outputs
