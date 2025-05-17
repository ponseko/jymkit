from typing import List, Literal

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import PRNGKeyArray, PyTree

import jymkit as jym


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


class ActionLinear(eqx.Module):
    layers: list
    space_type: Literal["Discrete", "MultiDiscrete", "Continuous"] = eqx.field(
        static=True
    )

    def __init__(self, key: PRNGKeyArray, space: jym.Space, in_features: int):
        assert len(space.shape) <= 1, (
            f"Currently, only 0D or 1D spaces are supported. Got {space.shape}. ",
            "For higher dimensions, use a composite of spaces or a custom network.",
        )

        # Determine the type of space and get the number and dimension of outputs
        if hasattr(space, "nvec"):
            self.space_type = "MultiDiscrete"
            num_outputs: list = np.array(space.nvec).tolist()  # pyright: ignore[reportAttributeAccessIssue]
        elif hasattr(space, "n"):
            self.space_type = "Discrete"
            num_outputs = [int(space.n)]  # pyright: ignore[reportAttributeAccessIssue]
        else:  # Box (Continuous)
            self.space_type = "Continuous"
            assert hasattr(space, "low") and hasattr(space, "high")
            num_outputs = (np.ones(space.shape, dtype=int) * 2).tolist()  # mean, std
        keys = optax.tree_utils.tree_split_key_like(key, num_outputs)
        self.layers = jax.tree.map(
            lambda o, k: eqx.nn.Linear(in_features, o, key=k), num_outputs, keys
        )
        if len(self.layers) == 1:
            self.layers = self.layers[0]

    def __call__(self, x, action_mask):
        if isinstance(self.layers, eqx.nn.Linear):
            logits = self.layers(x)  # single-dimensional output
        else:
            try:  # If actions are homogeneous, we can stack the outputs and use vmap
                stacked_layers = jax.tree.map(lambda *v: jnp.stack(v), *self.layers)
                logits = jax.vmap(lambda layer: layer(x))(stacked_layers)
            except ValueError:  # Else just map
                logits = jax.tree.map(
                    lambda layer: layer(x),
                    self.layers,
                    is_leaf=lambda x: isinstance(x, eqx.nn.Linear),
                )

        if action_mask is not None:
            logits = self._apply_action_mask(logits, action_mask)

        if self.space_type == "Continuous":
            return distrax.Normal(
                loc=logits[..., 0], scale=jax.nn.softplus(logits[..., 1])
            )

        return distrax.Categorical(logits=logits)

    def _apply_action_mask(self, logits, action_mask):
        """Apply the action mask to the output of the network.

        NOTE: This requires a (multi-)discrete action space.
        NOTE: Currently, action mask is assumed to be a PyTree of the same structure as the action space.
            Therefore, masking is not supported when the mask is dependent on another action.
        """
        if self.space_type == "Continuous":
            raise ValueError("Action masks provided for a continuous action space.")
        BIG_NEGATIVE = -1e9
        masked_logits = jax.tree.map(
            lambda a, mask: ((jnp.ones_like(a) * BIG_NEGATIVE) * (1 - mask)) + a,
            logits,
            action_mask,
        )
        return masked_logits


class ActorNetwork(eqx.Module):
    """
    A Basic class for RL agents that can be used to create actor and critic networks
    with different architectures.
    This agent will flatten all observations and treat it as a single vector.
    """

    layers: list

    def __init__(
        self,
        key: PRNGKeyArray,
        obs_space: PyTree[jym.Space],
        hidden_dims: List[int],
        output_space: PyTree[jym.Space],
        use_bronet: bool = False,
    ):
        if use_bronet:
            self.layers = create_bronet_networks(key, obs_space, hidden_dims)
        else:
            self.layers = create_ffn_networks(key, obs_space, hidden_dims)

        keys = optax.tree_utils.tree_split_key_like(key, output_space)
        output_nets = jax.tree.map(
            lambda o, k: ActionLinear(k, o, self.layers[-1].out_features),
            output_space,
            keys,
        )
        self.layers.append(output_nets)

    def __call__(self, x):
        action_mask = None
        if isinstance(x, jym.AgentObservation):
            action_mask = x.action_mask
            x = x.observation

        # If multiple spaces of observations, concat them (assuming they are all 1D)
        # This should have been enforced in the creation of the networks
        x = jax.tree.leaves(x)
        x = jax.tree.map(jnp.atleast_1d, x)
        x = jnp.concatenate(x)

        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))

        if action_mask is None:  # Create dummy PyTree for action mask
            action_mask = jax.tree.map(
                lambda _: None,
                self.layers[-1],
                is_leaf=lambda x: isinstance(x, ActionLinear),
            )
        action_dists = jax.tree.map(
            lambda action_layer, mask: action_layer(x, mask),
            self.layers[-1],
            action_mask,
            is_leaf=lambda x: isinstance(x, ActionLinear),
        )
        return action_dists


class CriticNetwork(eqx.Module):
    """
    A Basic class for RL agents that can be used to create actor and critic networks
    with different architectures.
    This agent will flatten all observations and treat it as a single vector.
    """

    layers: list

    def __init__(
        self,
        key: PRNGKeyArray,
        obs_space: PyTree[jym.Space],
        hidden_dims: List[int],
        use_bronet: bool = False,
    ):
        if use_bronet:
            self.layers = create_bronet_networks(key, obs_space, hidden_dims)
        else:
            self.layers = create_ffn_networks(key, obs_space, hidden_dims)
        self.layers.append(eqx.nn.Linear(self.layers[-1].out_features, 1, key=key))

    def __call__(self, x):
        if isinstance(x, jym.AgentObservation):
            x = x.observation

        # If multiple spaces of observations, concat them (assuming they are all 1D)
        # This should have been enforced in the creation of the networks
        x = jax.tree.leaves(x)
        x = jax.tree.map(jnp.atleast_1d, x)
        x = jnp.concatenate(x)

        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return jnp.squeeze(self.layers[-1](x), axis=-1)
