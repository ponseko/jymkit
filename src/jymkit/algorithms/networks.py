from typing import List

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PRNGKeyArray, PyTree, PyTreeDef

import jymkit as jym


class AgentNetwork(eqx.Module):
    """
    A Basic class for RL agents that can be used to create actor and critic networks
    with different architectures.
    This agent will flatten all observations and treat it as a single vector.
    """

    layers: list
    output_structure: PyTreeDef = eqx.field(static=True)
    # use_bronet: bool = False  # TODO

    def __init__(
        self,
        key: PRNGKeyArray,
        obs_space: PyTree[jym.Space],
        hidden_dims: List[int],
        output_space: int | PyTree[jym.Space],
    ):
        self.layers = []
        # TODO: structure test for output_space = SpaceContainer
        self.output_structure = jax.tree.structure(output_space)
        keys = jax.random.split(key, len(hidden_dims))

        # Observation space may be an arbitrary tree structure: we flatten it
        input_shape = jax.tree.map(
            lambda x: np.array(x.shape).prod(),
            obs_space,
            # is_leaf=lambda x: isinstance(x, jym.Space),
        )
        input_dim = int(np.sum(np.array(input_shape)))

        for i, hidden_dim in enumerate(hidden_dims):
            self.layers.append(
                eqx.nn.Linear(
                    in_features=input_dim, out_features=hidden_dim, key=keys[i]
                )
            )
            input_dim = hidden_dim

        # TODO: Continuous action space
        num_outputs = output_space
        if not isinstance(output_space, int):  # output is int for value network
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
            lambda x: eqx.nn.Linear(input_dim, x, key=keys[-1]), num_outputs
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
            return self.layers[-1](x)

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

        return logits

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


class ActorNetwork(AgentNetwork):
    def __call__(self, x):
        output = super().__call__(x)
        return distrax.Categorical(logits=output)


class CriticNetwork(AgentNetwork):
    def __call__(self, x):
        output = super().__call__(x)
        return jnp.squeeze(output, axis=-1)
