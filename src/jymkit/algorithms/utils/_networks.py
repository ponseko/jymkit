from typing import List

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PRNGKeyArray, PyTree, PyTreeDef

import jymkit as jym


def create_ffn_networks(key: PRNGKeyArray, obs_space, hidden_dims):
    """
    Create a feedforward neural network with the given hidden dimensions and output space.
    """
    layers = []
    keys = jax.random.split(key, len(hidden_dims))

    # Flatten the input space
    input_shape = jax.tree.map(
        lambda x: np.array(x.shape).prod(),
        obs_space,
        # is_leaf=lambda x: isinstance(x, jym.Space),
    )
    input_dim = int(np.sum(np.array(input_shape)))
    for i, hidden_dim in enumerate(hidden_dims):
        layers.append(
            eqx.nn.Linear(in_features=input_dim, out_features=hidden_dim, key=keys[i])
        )
        input_dim = hidden_dim

    return layers


def create_bronet_networks(key: PRNGKeyArray, obs_space, hidden_dims):
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

    # Flatten the input space
    input_shape = jax.tree.map(
        lambda x: np.array(x.shape).prod(),
        obs_space,
        # is_leaf=lambda x: isinstance(x, jym.Space),
    )
    input_dim = int(np.sum(np.array(input_shape)))
    layers = [
        eqx.nn.Linear(in_features=input_dim, out_features=hidden_dims[0], key=keys[0]),
        eqx.nn.LayerNorm(hidden_dims[0]),
    ]
    for i, hidden_dim in enumerate(hidden_dims):
        layers.append(BroNetBlock(keys[i], hidden_dim))

    return layers


class ActorNetwork(eqx.Module):
    """
    A Basic class for RL agents that can be used to create actor and critic networks
    with different architectures.
    This agent will flatten all observations and treat it as a single vector.
    """

    layers: list
    output_structure: PyTreeDef = eqx.field(static=True)
    use_bronet: bool = eqx.field(static=True, default=False)

    def __init__(
        self,
        key: PRNGKeyArray,
        obs_space: PyTree[jym.Space],
        hidden_dims: List[int],
        output_space: int | PyTree[jym.Space],
    ):
        if self.use_bronet:
            self.layers = create_bronet_networks(key, obs_space, hidden_dims)
        else:
            self.layers = create_ffn_networks(key, obs_space, hidden_dims)

        self.output_structure = jax.tree.structure(output_space)
        # TODO: Continuous action space
        num_outputs = jax.tree.map(
            lambda o: np.array(o.high) - np.array(o.low),
            output_space,
            # is_leaf=lambda x: isinstance(x, jym.Space),
        )
        num_outputs = jax.tree.map(
            lambda o: o.tolist() if eqx.is_array(o) else o,
            num_outputs,
        )
        output_nets = jax.tree.map(
            lambda x: eqx.nn.Linear(self.layers[-1].out_features, x, key=key),
            num_outputs,
        )
        self.layers.append(output_nets)

    def __call__(self, x):
        action_mask = None
        if isinstance(x, jym.AgentObservation):
            action_mask = x.action_mask
            x = x.observation

        x = jax.tree.map(lambda x: jnp.reshape(x, -1), x)  # flatten the input
        if not isinstance(x, jnp.ndarray):
            x = jnp.concatenate(x)
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))

        if not isinstance(self.layers[-1], list):  # single-dimensional output
            logits = self.layers[-1](x)
        else:  # multi-dimensional output
            try:
                # TODO: maybe move the map.stack to the init
                # If homogeneous output, we can stack the outputs and use vmap
                final_layers = jax.tree.map(lambda *v: jnp.stack(v), *self.layers[-1])
                outputs = jax.vmap(lambda layer: layer(x))(final_layers)
                if self.output_structure.num_leaves == 1:
                    outputs = [outputs]
                else:
                    outputs = outputs.tolist()  # TODO: test
            except ValueError:
                outputs = jax.tree.map(lambda x: x(x), self.layers[-1])

            logits = jax.tree.unflatten(self.output_structure, outputs)
        if action_mask is not None:
            logits = self._apply_action_mask(logits, action_mask)

        return distrax.Categorical(logits=logits)

    def _apply_action_mask(self, logits, action_mask):
        """
        Apply the action mask to the output of the network.
        """
        BIG_NEGATIVE = -1e9
        masked_logits = jax.tree.map(
            lambda a, mask: ((jnp.ones_like(a) * BIG_NEGATIVE) * (1 - mask)) + a,
            logits,
            action_mask,
        )
        return masked_logits


class CriticNetwork(eqx.Module):
    """
    A Basic class for RL agents that can be used to create actor and critic networks
    with different architectures.
    This agent will flatten all observations and treat it as a single vector.
    """

    layers: list
    use_bronet: bool = eqx.field(static=True, default=False)

    def __init__(
        self,
        key: PRNGKeyArray,
        obs_space: PyTree[jym.Space],
        hidden_dims: List[int],
    ):
        if self.use_bronet:
            self.layers = create_bronet_networks(key, obs_space, hidden_dims)
        else:
            self.layers = create_ffn_networks(key, obs_space, hidden_dims)
        self.layers.append(eqx.nn.Linear(self.layers[-1].out_features, 1, key=key))

    def __call__(self, x):
        if isinstance(x, jym.AgentObservation):
            x = x.observation

        x = jax.tree.map(lambda x: jnp.reshape(x, -1), x)  # flatten the input
        if not isinstance(x, jnp.ndarray):
            x = jnp.concatenate(x)
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return jnp.squeeze(self.layers[-1](x), axis=-1)
