import warnings
from abc import abstractmethod
from typing import Any, Callable, List, Literal

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

import jymkit as jym
from jymkit.algorithms.utils import DistraxContainer


def _get_input_dim_of_flat_obs(obs_space: jym.Space | PyTree[jym.Space]) -> int:
    """
    Get the flattened input dimension of the observation space.
    """
    # Check if each obs_space is a 0D or 1D space
    below_2d = jax.tree.leaves(jax.tree.map(lambda x: len(x.shape) < 2, obs_space))
    assert all(below_2d), (
        "This model requires all observations to be 0D or 1D spaces."
        "Flatten the observations with `jymkit.FlattenObservationWrapper` or "
        "use a custom network.",
        f"spaces={obs_space}",
    )
    input_shape = jax.tree.map(
        lambda x: int(np.array(x.shape).prod()),
        obs_space,
    )
    input_dim = int(np.sum(np.array(jax.tree.leaves(input_shape))))
    return input_dim


def create_ffn_networks(
    key: PRNGKeyArray, obs_space: jym.Space, hidden_dims: List[int]
):
    """
    Create a feedforward neural network with the given hidden dimensions and output space.
    """
    layers = []
    keys = jax.random.split(key, len(hidden_dims))

    input_dim = _get_input_dim_of_flat_obs(obs_space)
    for i, hidden_dim in enumerate(hidden_dims):
        layers.append(
            eqx.nn.Linear(in_features=input_dim, out_features=hidden_dim, key=keys[i])
        )
        input_dim = hidden_dim

    return layers


def create_bronet_networks(
    key: PRNGKeyArray, obs_space: jym.Space, hidden_dims: List[int]
):
    """
    Create a BroNet neural network with the given hidden dimensions and output space.
    https://arxiv.org/html/2405.16158v1
    """

    class BroNetBlock(eqx.Module):
        layers: list
        in_features: int = eqx.field(static=True)
        out_features: int = eqx.field(static=True)

        def __init__(self, key: PRNGKeyArray, shape: int):
            key1, key2 = jax.random.split(key)
            self.layers = [
                eqx.nn.Linear(in_features=shape, out_features=shape, key=key1),
                eqx.nn.LayerNorm(shape),
                eqx.nn.Linear(in_features=shape, out_features=shape, key=key2),
                eqx.nn.LayerNorm(shape),
            ]
            self.in_features = shape
            self.out_features = shape

        def __call__(self, x):
            _x = self.layers[0](x)
            _x = self.layers[1](_x)
            _x = jax.nn.relu(_x)
            _x = self.layers[2](_x)
            _x = self.layers[3](_x)
            return x + _x

    keys = jax.random.split(key, len(hidden_dims))

    input_dim = _get_input_dim_of_flat_obs(obs_space)
    layers = [
        eqx.nn.Linear(in_features=input_dim, out_features=hidden_dims[0], key=keys[0]),
        eqx.nn.LayerNorm(hidden_dims[0]),
    ]
    for i, hidden_dim in enumerate(hidden_dims):
        layers.append(BroNetBlock(keys[i], hidden_dim))

    return layers


class AgentOutputLinear(eqx.Module):
    layers: List[eqx.nn.Linear]

    @abstractmethod
    def __init__(self, key: PRNGKeyArray, in_features: int, space: Any):
        pass

    def __call__(self, x, action_mask):
        if len(self.layers) == 1:
            # Single output layer (Discrete action space with one dimension)
            logits = self.layers[0](x)  # single-dimensional output
        else:
            stacked_layers = jax.tree.map(lambda *v: jnp.stack(v), *self.layers)
            logits = jax.vmap(lambda layer: layer(x))(stacked_layers)

        if action_mask is not None:
            logits = self._apply_action_mask(logits, action_mask)

        return logits

    def _create_pytree_of_output_heads(self, key, in_features, num_outputs):
        """Create a PyTree of output heads based on the number of outputs."""
        keys = optax.tree_utils.tree_split_key_like(key, num_outputs)
        return jax.tree.map(
            lambda o, k: eqx.nn.Linear(in_features, o, key=k), num_outputs, keys
        )

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


class DiscreteActionLinear(AgentOutputLinear):
    def __init__(self, key, in_features: int, space: Any):
        if hasattr(space, "n"):
            num_outputs = np.array([int(space.n)])
        elif hasattr(space, "nvec"):
            num_outputs = np.array(space.nvec)
        else:
            raise ValueError(
                "Missing attributes 'n' or 'nvec' in the action space for DiscreteLinear."
            )

        # Convert to PyTree (List) to create a PyTree of output heads
        num_outputs = num_outputs.tolist()
        self.layers = self._create_pytree_of_output_heads(key, in_features, num_outputs)

    def __call__(self, x, action_mask):
        logits = super().__call__(x, action_mask)
        return distrax.Categorical(logits=logits)


class ContinuousActionLinear(AgentOutputLinear):
    low: Float[Array, " action_dim"]
    high: Float[Array, " action_dim"]

    def __init__(self, key, in_features: int, space: Any):
        assert (
            hasattr(space, "low") and hasattr(space, "high") and hasattr(space, "shape")
        ), (
            "Continuous action space is assumed to be a `Box`-like and "
            "must have 'low' and 'high' and `shape` attributes."
        )
        self.low = jnp.array(space.low, dtype=jnp.float32)
        self.high = jnp.array(space.high, dtype=jnp.float32)
        num_outputs = np.ones(space.shape, dtype=int) * 2  # mean, std

        # Convert to PyTree (List) to create a PyTree of output heads
        num_outputs = num_outputs.tolist()
        self.layers = self._create_pytree_of_output_heads(key, in_features, num_outputs)

    def __call__(self, x, action_mask=None):
        LOG_STD_MIN = -20
        LOG_STD_MAX = 2

        logits = super().__call__(x, action_mask)
        mean = logits[..., 0]
        log_std = logits[..., 1]
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)
        return distrax.Normal(loc=mean, scale=std)

    def _apply_action_mask(self, logits, action_mask):
        warnings.warn(
            "Action mask provided for a unsupported continuous action space. Ignoring ..."
        )


class DiscreteQLinear(AgentOutputLinear):
    def __init__(self, key, in_features: int, space: Any):
        if hasattr(space, "n"):
            num_outputs = np.array([int(space.n)])
        elif hasattr(space, "nvec"):
            num_outputs = np.array(space.nvec)
        else:
            raise ValueError(
                "Missing attributes 'n' or 'nvec' in the action space for DiscreteLinear."
            )

        # Convert to PyTree (List) to create a PyTree of output heads
        num_outputs = num_outputs.tolist()
        self.layers = self._create_pytree_of_output_heads(key, in_features, num_outputs)

    def __call__(self, x, action_mask, action):  # type: ignore[override]
        logits = super().__call__(x, action_mask)
        return logits


class ContinuousQLinear(AgentOutputLinear):
    def __init__(self, key, in_features: int, space: Any):
        assert (
            hasattr(space, "low") and hasattr(space, "high") and hasattr(space, "shape")
        ), (
            "Continuous action space is assumed to be a `Box`-like and "
            "must have 'low' and 'high' and `shape` attributes."
        )
        num_outputs = np.ones(space.shape, dtype=int)

        # Continuous Q networks receive action as input, so increase in_features:
        in_features += num_outputs.size  # Add action dimension

        # Convert to PyTree (List) to create a PyTree of output heads
        num_outputs = num_outputs.tolist()
        self.layers = self._create_pytree_of_output_heads(key, in_features, num_outputs)

    def __call__(self, x, action_mask, action):  # type: ignore[override]
        # concat action to the input
        action = jnp.atleast_1d(action)
        x = jnp.concatenate([x, action], axis=-1)
        logits = super().__call__(x, action_mask)
        return logits.squeeze()

    def _apply_action_mask(self, logits, action_mask):
        warnings.warn(
            "Action mask provided for a unsupported continuous action space. Ignoring ..."
        )


def create_agent_output_layers(
    key,
    space: jym.Space | Any,
    in_features: int,
    network_type: Literal["actor", "critic"],
):
    def _check_valid_action_space(space: jym.Space):
        assert len(space.shape) <= 1, (
            f"Currently, only 0D or 1D spaces are supported. Got {space.shape}. "
            "For higher dimensions, use a composite of spaces or a custom network.",
        )
        # If multiple dimensions, assert that each dimension is of equal size
        assert len(space.shape) == 0 or len(set(space.shape)) == 1, (
            f"Action space dimensions must be of equal size. "
            f"Got {space.shape}. For varying dimensions, use a composite of spaces or a custom network.",
        )

    _check_valid_action_space(space)

    if network_type == "actor":
        if hasattr(space, "n") or hasattr(space, "nvec"):
            return DiscreteActionLinear(key, in_features, space)
        return ContinuousActionLinear(key, in_features, space)

    elif network_type == "critic":
        if hasattr(space, "n") or hasattr(space, "nvec"):
            return DiscreteQLinear(key, in_features, space)  # type: ignore[return]
        return ContinuousQLinear(key, in_features, space)


class AbstractAgentNetwork(eqx.Module):
    output_layers: PyTree[eqx.nn.Linear]
    ffn_layers: list
    feature_extractor: Callable = eqx.nn.Identity()

    def init_backbone(self, key: PRNGKeyArray, obs_space, hidden_dims: List[int]):
        """
        Initialize the backbone of the network based on the observation space.
        """
        # TODO: Feature extractor
        # self.feature_extractor = Identity()

        self.ffn_layers = create_ffn_networks(key, obs_space, hidden_dims)
        # Possibly use BroNet

    @staticmethod
    def concat_all_spaces(x):
        """
        Concatenates all observation spaces into a single vector.
        This assumes that all spaces are 0D or 1D or are already flattened,
        e.g. using `jymkit.FlattenObservationWrapper` or a feature extractor.
        """
        x = jax.tree.leaves(x)
        x = jax.tree.map(jnp.atleast_1d, x)
        return jnp.concatenate(x)


class ActorNetwork(AbstractAgentNetwork):
    def __init__(
        self,
        key: PRNGKeyArray,
        obs_space: PyTree[jym.Space],
        hidden_dims: List[int],
        output_space: PyTree[jym.Space],
    ):
        self.init_backbone(key, obs_space, hidden_dims)

        keys = optax.tree_utils.tree_split_key_like(key, output_space)
        self.output_layers = jax.tree.map(
            lambda o, k: create_agent_output_layers(
                k, o, self.ffn_layers[-1].out_features, "actor"
            ),
            output_space,
            keys,
        )

    def __call__(self, x):
        if isinstance(x, jym.AgentObservation):
            action_mask = x.action_mask
            x = x.observation
        else:
            action_mask = jax.tree.map(
                lambda _: None,
                self.output_layers,
                is_leaf=lambda x: isinstance(x, AgentOutputLinear),
            )

        x = jax.tree.map(self.feature_extractor, x)
        x = self.concat_all_spaces(x)
        for layer in self.ffn_layers:
            x = jax.nn.gelu(layer(x))

        action_dists = jax.tree.map(
            lambda action_layer, mask: action_layer(x, mask),
            self.output_layers,
            action_mask,
            is_leaf=lambda x: isinstance(x, AgentOutputLinear),
        )
        return DistraxContainer(action_dists)


class ValueNetwork(AbstractAgentNetwork):
    def __init__(
        self,
        key: PRNGKeyArray,
        obs_space: PyTree[jym.Space],
        hidden_dims: List[int],
    ):
        self.init_backbone(key, obs_space, hidden_dims)
        self.output_layers = eqx.nn.Linear(self.ffn_layers[-1].out_features, 1, key=key)

    def __call__(self, x):
        if isinstance(x, jym.AgentObservation):
            x = x.observation
        x = jax.tree.map(self.feature_extractor, x)
        x = self.concat_all_spaces(x)

        for layer in self.ffn_layers:
            x = jax.nn.gelu(layer(x))
        output = self.output_layers(x)
        return jnp.squeeze(output, axis=-1)


class QValueNetwork(AbstractAgentNetwork):
    def __init__(
        self,
        key: PRNGKeyArray,
        obs_space: PyTree[jym.Space],
        output_space: PyTree[jym.Space],
        hidden_dims: List[int],
    ):
        self.init_backbone(key, obs_space, hidden_dims)
        keys = optax.tree_utils.tree_split_key_like(key, output_space)
        self.output_layers = jax.tree.map(
            lambda o, k: create_agent_output_layers(
                k, o, self.ffn_layers[-1].out_features, "critic"
            ),
            output_space,
            keys,
        )

    def __call__(self, x, action=None):
        if isinstance(x, jym.AgentObservation):
            action_mask = x.action_mask
            x = x.observation
        else:
            action_mask = jax.tree.map(
                lambda _: None,
                self.output_layers,
                is_leaf=lambda x: isinstance(x, AgentOutputLinear),
            )
        if action is None:
            action = jax.tree.map(
                lambda _: None,
                self.output_layers,
                is_leaf=lambda x: isinstance(x, AgentOutputLinear),
            )

        x = jax.tree.map(self.feature_extractor, x)
        x = self.concat_all_spaces(x)
        for layer in self.ffn_layers:
            x = jax.nn.gelu(layer(x))

        q_values = jax.tree.map(
            lambda action_layer, mask, a: action_layer(x, mask, a),
            self.output_layers,
            action_mask,
            action,
            is_leaf=lambda x: isinstance(x, AgentOutputLinear),
        )
        return q_values
