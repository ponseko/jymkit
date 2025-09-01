import logging
from typing import Callable, List, Literal, Tuple

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import PRNGKeyArray, PyTree, PyTreeDef

import jymkit as jym
import jymkit.tree
from jymkit.algorithms.utils import TanhNormalFactory

from ._architectures import CNN, Identity

logger = logging.getLogger(__name__)


class MultiInputNetwork(eqx.Module):
    """Input network for **single and multi** observation space environments.

    Builds a separate input network for each observation space. Automatically
    builds a 1d architecture for 1d observation spaces (default=None/Identity) and
    a 2d architecture for 2d observation spaces (default=NatureCNN-like).

    During a forward call this network simply returns a jax.tree.map over all input spaces.
    optionally concatenates the outputs of all input networks as a single 1d vector.
    """

    networks: PyTree[eqx.Module]
    num_observation_spaces: int = eqx.field(static=True)
    input_structure: PyTreeDef = eqx.field(static=True)
    output_structure: PyTreeDef = eqx.field(static=True)
    out_features: int | None = eqx.field(static=True, default=None)
    """ Only set if `concatenate_outputs_to_1d=True`. """
    concatenate_outputs_to_1d: bool = eqx.field(static=True, default=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        obs_space: PyTree[jym.Space],
        *,
        concatenate_outputs_to_1d: bool = True,
        architecture_1d: Literal["identity"] = "identity",
        architecture_2d: Literal["cnn"] = "cnn",
        cnn_hidden_sizes: Tuple[int, ...] = (32, 64, 64),
        cnn_kernel_sizes: Tuple[int, ...] = (3, 3, 2),
        cnn_strides: Tuple[int, ...] = (1, 1, 1),
        cnn_padding: Tuple[int, ...] = (0, 0, 0),
        **kwargs,
    ):
        def _create_input_network(key: PRNGKeyArray, obs_space: jym.Space):
            if len(obs_space.shape) > 3:
                raise NotImplementedError(
                    f"Observation space shape: {obs_space.shape} not supported. "
                    f"Only 1D and 2D (excluding channels) observations are supported."
                )

            if obs_space.shape == () or len(obs_space.shape) == 1:  # 0d or 1d input
                if architecture_1d.lower() in ["identity"]:
                    return Identity()

                # elif architecture == "broNet":
                #     depth = 3
                #     width_size = 128
                #     return BroNet(key, obs_space.shape[0], depth, width_size)

                else:
                    raise ValueError(f"Unsupported 1d architecture: {architecture_1d}")

            elif len(obs_space.shape) == 3 or len(obs_space.shape) == 2:
                if architecture_2d.lower() in ["cnn", "naturecnn"]:
                    # in_channels = obs_space.shape[0]  # Assume channel-first
                    return CNN(
                        key,
                        obs_space,
                        cnn_hidden_sizes,
                        cnn_kernel_sizes,
                        cnn_strides,
                        cnn_padding,
                    )
                # elif architecture == "hadamax":
                #     pass

                else:
                    raise ValueError(f"Unsupported 2d architecture: {architecture_2d}")
            else:
                raise NotImplementedError(
                    f"Unsupported observation space shape: {obs_space.shape}"
                )

        self.num_observation_spaces = len(jax.tree.leaves(obs_space))
        self.input_structure = jax.tree.structure(obs_space)
        self.concatenate_outputs_to_1d = concatenate_outputs_to_1d
        if concatenate_outputs_to_1d:
            self.output_structure = jax.tree.structure(0)
        else:
            self.output_structure = self.input_structure

        keys = optax.tree.split_key_like(key, obs_space)
        self.networks = jax.tree.map(
            lambda o, k: _create_input_network(k, o), obs_space, keys
        )

        # Infer the output feature size if concatenating to 1d
        if concatenate_outputs_to_1d:
            dummy_obs = jax.tree.map(
                lambda o: jnp.zeros(o.shape, dtype=jnp.float32), obs_space
            )
            f = lambda obs: jnp.atleast_1d(self(obs))  # type: ignore
            self.out_features = jax.eval_shape(f, dummy_obs).shape[0]

    def __call__(self, x):
        # Convert non-float inputs to float32
        x = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), x)

        if callable(self.networks) and self.num_observation_spaces == 1:
            return self.networks(x)  # More efficient compared to map (?)

        outputs = jax.tree.map(
            lambda layer, x: layer(x),
            self.networks,
            x,
            is_leaf=lambda x: isinstance(x, eqx.Module),
        )
        if self.concatenate_outputs_to_1d:
            return jymkit.tree.concatenate(outputs)
        return outputs


class MultiOutputNetwork(eqx.Module):
    """Output network for **single and multi** action space environments.

    Will build an individual output head for each action space and inner individual
    output heads in multidimensional spaces. During a forward call this network
    simply returns a jax.tree.map over all output spaces.
    Inner multidimensional spaces are vmapped, and are expected to be 1d and homogeneous.
    """

    networks: PyTree[eqx.Module]
    num_action_spaces: int = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        in_features: int,
        output_space: PyTree[jym.Space],
        *,
        discrete_output_dist: Literal["categorical",] | None = "categorical",
        continuous_output_dist: Literal["normal", "tanhnormal"] | None = "normal",
        **kwargs,
    ):
        def _create_output_network(key: PRNGKeyArray, output_space: jym.Space):
            is_discrete = hasattr(output_space, "n") or hasattr(output_space, "nvec")
            if is_discrete:
                return DiscreteOutputNetwork(
                    key, in_features, output_space, distribution=discrete_output_dist
                )

            is_continu = hasattr(output_space, "low") and hasattr(output_space, "high")
            if is_continu:
                return ContinuousOutputNetwork(
                    key,
                    in_features,
                    output_space,  # type: ignore
                    distribution=continuous_output_dist,
                )

            else:
                raise ValueError(f"Unsupported action space: {output_space}")

        self.num_action_spaces = len(jax.tree.leaves(output_space))
        keys = optax.tree.split_key_like(key, output_space)
        self.networks = jax.tree.map(
            lambda a, k: _create_output_network(k, a), output_space, keys
        )

    def __call__(self, x, action_mask):
        if self.num_action_spaces == 1:
            return self.networks(x, action_mask)  # More efficient compared to map (?)

        if action_mask is None:  # Dummy action mask if not provided
            action_mask = jax.tree.map(
                lambda _: None,
                self.networks,
                is_leaf=lambda x: isinstance(x, eqx.Module),
            )
        return jax.tree.map(
            lambda layer, mask: layer(x, mask),
            self.networks,
            action_mask,
            is_leaf=lambda x: isinstance(x, eqx.Module),
        )


class DiscreteOutputNetwork(eqx.Module):
    layers: List[eqx.nn.Linear]
    distribution: Callable[..., distrax.Distribution] | None = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        in_features: int,
        output_space: jym.Space,
        distribution: Literal["categorical"] | None,
    ):
        if distribution is None:
            self.distribution = None
        elif distribution == "categorical":
            self.distribution = distrax.Categorical
        else:
            raise ValueError(f"Unsupported discrete distribution: {distribution}")

        # Get n (Discrete) or nvec (MultiDiscrete)
        num_outputs = getattr(output_space, "n", getattr(output_space, "nvec", None))
        if num_outputs is None:
            raise ValueError(f"Unsupported action space: {output_space}")

        # We create a list of outputs: [n] for Discrete, [n,n,...] for MultiDiscrete
        num_outputs = np.atleast_1d(num_outputs).tolist()
        assert len(set(num_outputs)) == 1, (
            "Only homogeneous MultiDiscrete spaces supported to be supported for vmap."
            f" (all nvec elements must be the same, got {num_outputs}) "
            "For heterogeneous spaces, use a composite of spaces instead."
        )

        # Then we create an output head per element (1 for Discrete)
        # These heads can be stacked + vmapped in the forward call
        keys = optax.tree.split_key_like(key, num_outputs)
        self.layers = jax.tree.map(
            lambda o, k: eqx.nn.Linear(in_features, o, key=k), num_outputs, keys
        )

    def __call__(self, x, action_mask):
        if len(self.layers) == 1:  # single dimensional output
            logits = self.layers[0](x)
        else:
            stacked_layers = jymkit.tree.stack(self.layers)
            logits = jax.vmap(lambda layer: layer(x))(stacked_layers)

        if action_mask is not None:
            logits = self._apply_action_mask(logits, action_mask)

        if self.distribution is not None:
            return self.distribution(logits=logits)
        return logits

    def _apply_action_mask(self, logits, action_mask):
        """Apply the action mask to the output of the network.

        NOTE: This requires a (multi-)discrete action space.
        NOTE: Currently, action mask is assumed to be a PyTree of the same structure as the action space.
            Therefore, masking is not supported when the mask is dependent on another action.
        """

        BIG_NEGATIVE = -1e9
        masked_logits = jax.tree.map(
            lambda a, mask: ((jnp.ones_like(a) * BIG_NEGATIVE) * (1 - mask)) + a,
            logits,
            action_mask,
        )
        return masked_logits


class ContinuousOutputNetwork(eqx.Module):
    layers: List[eqx.nn.Linear]
    distribution: Callable[..., distrax.Distribution] | None = eqx.field(static=True)
    low: np.ndarray
    high: np.ndarray

    def __init__(
        self,
        key: PRNGKeyArray,
        in_features: int,
        output_space: jym.Box,
        distribution: Literal["normal", "tanhnormal"] | None,
    ):
        assert hasattr(output_space, "low") and hasattr(output_space, "high"), (
            "Continuous action space is assumed to be a `Box`-like and "
            "must have 'low' and 'high' and `shape` attributes."
        )

        self.low = np.array(output_space.low, dtype=float)
        self.high = np.array(output_space.high, dtype=float)
        if distribution is None:
            self.distribution = None
        elif distribution.lower() == "normal":
            self.distribution = distrax.Normal
        elif distribution.lower() == "tanhnormal":
            self.distribution = TanhNormalFactory(low=self.low, high=self.high)
        else:
            raise ValueError(f"Unsupported continuous distribution: {distribution}")

        num_outputs = np.ones(output_space.shape, dtype=int)
        if distribution is not None:
            num_outputs = num_outputs * 2  # mean, std
            # NOTE: this assumes Normal / TanhNormal distribution with 2 outputs

        # We create a list of output heads per dimension
        # These heads can be stacked + vmapped in the forward call
        num_outputs = num_outputs.tolist()

        assert len(set(num_outputs)) == 1, (
            "Only homogeneous Box spaces supported to be supported for vmap."
            f" (all shape elements must be the same, got {num_outputs}) "
            "For heterogeneous spaces, use a composite of spaces instead."
        )

        keys = optax.tree.split_key_like(key, num_outputs)
        self.layers = jax.tree.map(
            lambda o, k: eqx.nn.Linear(in_features, o, key=k), num_outputs, keys
        )

    def __call__(self, x, action_mask):
        if action_mask is not None:
            logging.debug("Action mask provided for continuous action space, ignoring.")

        if len(self.layers) == 1:  # single dimensional output
            logits = self.layers[0](x)
        else:
            stacked_layers = jymkit.tree.stack(self.layers)
            logits = jax.vmap(lambda layer: layer(x))(stacked_layers)

        if self.distribution is None:
            return logits

        mean = logits[..., 0]
        log_std = logits[..., 1]
        log_std = jnp.clip(log_std, -20, 2)
        std = jnp.exp(log_std)

        return self.distribution(mean, std)
