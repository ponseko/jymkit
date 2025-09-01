import logging

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, PyTree

import jymkit as jym
from jymkit.algorithms.utils import DistraxContainer

from ._architectures import MLP
from ._input_output import MultiInputNetwork, MultiOutputNetwork

logger = logging.getLogger(__name__)


class ActorNetwork(eqx.Module):
    input_layers: MultiInputNetwork
    mlp: MLP
    output_layers: MultiOutputNetwork

    def __init__(
        self,
        key: PRNGKeyArray,
        obs_space: PyTree[jym.Space],
        output_space: PyTree[jym.Space],
        **network_kwargs,
    ):
        key_in, key_mlp, key_out = jax.random.split(key, 3)
        self.input_layers = MultiInputNetwork(
            key_in, obs_space, concatenate_outputs_to_1d=True, **network_kwargs
        )

        assert self.input_layers.out_features
        self.mlp = MLP(key_mlp, self.input_layers.out_features, **network_kwargs)

        self.output_layers = MultiOutputNetwork(
            key_out, self.mlp.out_features, output_space, **network_kwargs
        )

    def __call__(self, x):
        action_mask = None
        if isinstance(x, jym.AgentObservation):
            action_mask = x.action_mask
            x = x.observation

        x = self.input_layers(x)
        x = self.mlp(x)
        action_dists = self.output_layers(x, action_mask)
        if isinstance(action_dists, distrax.Distribution):
            return action_dists  # Single distribution

        # Else return a grouped container of distributions
        return DistraxContainer(action_dists)


class ValueNetwork(eqx.Module):
    input_layers: MultiInputNetwork
    mlp: MLP
    output_layers: eqx.nn.Linear

    def __init__(
        self,
        key: PRNGKeyArray,
        obs_space: PyTree[jym.Space],
        **network_kwargs,
    ):
        key_in, key_mlp, key_out = jax.random.split(key, 3)
        self.input_layers = MultiInputNetwork(
            key_in, obs_space, concatenate_outputs_to_1d=True, **network_kwargs
        )

        assert self.input_layers.out_features
        self.mlp = MLP(key_mlp, self.input_layers.out_features, **network_kwargs)

        self.output_layers = eqx.nn.Linear(self.mlp.out_features, 1, key=key_out)

    def __call__(self, x):
        if isinstance(x, jym.AgentObservation):
            x = x.observation

        x = self.input_layers(x)
        x = self.mlp(x)
        out = self.output_layers(x)
        return jnp.squeeze(out, axis=-1)


class QValueNetwork(eqx.Module):
    input_layers: MultiInputNetwork
    mlp: MLP
    output_layers: MultiOutputNetwork

    include_action_in_input: bool = eqx.field(static=True, default=False)

    def __init__(
        self,
        key: PRNGKeyArray,
        obs_space: PyTree[jym.Space],
        output_space: PyTree[jym.Space],
        **network_kwargs,
    ):
        is_continuous = [isinstance(s, jym.Box) for s in jax.tree.leaves(output_space)]
        if any(is_continuous):
            self.include_action_in_input = True
            if not all(is_continuous):
                logging.warning(
                    "Mixed action spaces with continuous QNetwork may have adverse training effects"
                )
            obs_space = {"_OBSERVATION": obs_space, "_ACTION": output_space}

        key_in, key_mlp, key_out = jax.random.split(key, 3)
        self.input_layers = MultiInputNetwork(
            key_in, obs_space, concatenate_outputs_to_1d=True, **network_kwargs
        )

        assert self.input_layers.out_features
        self.mlp = MLP(key_mlp, self.input_layers.out_features, **network_kwargs)

        self.output_layers = MultiOutputNetwork(
            key_out,
            self.mlp.out_features,
            output_space,
            discrete_output_dist=None,
            continuous_output_dist=None,
            **network_kwargs,
        )

    def __call__(self, x, action=None):
        action_mask = None
        if isinstance(x, jym.AgentObservation):
            action_mask = x.action_mask
            x = x.observation

        if self.include_action_in_input:
            assert action is not None, "Action not provided in continuous Q network."
            x = {"_OBSERVATION": x, "_ACTION": action}

        x = self.input_layers(x)
        x = self.mlp(x)
        q_values = self.output_layers(x, action_mask)
        return q_values


AdvantageCriticNetwork = QValueNetwork
