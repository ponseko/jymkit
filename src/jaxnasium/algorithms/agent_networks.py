import logging
from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray, PyTree

import jaxnasium as jym
from jaxnasium.algorithms.architectures import CNN, MLP
from jaxnasium.algorithms.core import (
    PyTreeObsSpaceNetwork,
    PyTreeOutputNetwork,
    set_weight_bias,
)
from jaxnasium.algorithms.types import Network, OutSizedNetwork

logger = logging.getLogger(__name__)

_DEFAULT_ARCHITECTURE_2D = CNN.with_params(
    out_channels=(32, 64, 64),
    kernel_sizes=(3, 3, 2),
    strides=(1, 1, 1),
    padding=(0, 0, 0),
)


"""
The base Reinforcement Learning Network classes (actor, V-network, Q-network).
Each of these consist of three components:
    - An observation processor (PyTreeObsSpaceNetwork)
        This accepts a PyTree of observation spaces and builds a network per observation space.
        1d observation spaces are processed via the configured ``architecture_1d`` network.
        2d observation spaces are processed via the configured ``architecture_2d`` network.
        The output of each observation processor is concatenated into a single 1d vector.
    - A body (default MLP) = A shared body network that takes the output of the observation processor and processes it jointly.
    - An output processor (PyTreeOutputNetwork)
        This accepts a PyTree of output spaces and builds a network per output space.
        Automtically builds a discrete or continuous output network based on the output space.
        Returns the output of each output network in the same PyTree structure as the action space.
"""


class ActorNetwork(eqx.Module):
    obs_processor: OutSizedNetwork
    body: OutSizedNetwork
    output_layers: PyTreeOutputNetwork

    def __init__(
        self,
        obs_space: PyTree[jym.Space],
        output_space: PyTree[jym.Space],
        *,
        key: PRNGKeyArray,
        body: Callable[..., OutSizedNetwork] = MLP,
        obs_architecture_1d: Callable[..., Network] = eqx.nn.Identity,
        obs_architecture_2d: Callable[..., Network] = _DEFAULT_ARCHITECTURE_2D,
        discrete_distribution: Literal["categorical"] | None = "categorical",
        continuous_distribution: Literal["normal", "tanhnormal"] | None = "normal",
        output_layer_type: Callable[..., Network] = eqx.nn.Linear,
        assume_independent_output: bool = True,
        weights_init: Callable[
            ..., jax.nn.initializers.Initializer
        ] = jax.nn.initializers.orthogonal,
        bias_init: float = 0.0,
    ):
        obs_key, body_key, output_key, wb_key = jax.random.split(key, 4)

        self.obs_processor = PyTreeObsSpaceNetwork(
            obs_space,
            key=obs_key,
            architecture_1d=obs_architecture_1d,
            architecture_2d=obs_architecture_2d,
        )

        self.body = body(self.obs_processor.out_features, key=body_key)

        self.output_layers = PyTreeOutputNetwork(
            self.body.out_features,
            output_space,
            key=output_key,
            discrete_distribution=discrete_distribution,
            continuous_distribution=continuous_distribution,
            layer_type=output_layer_type,
            assume_independent=assume_independent_output,
        )

        (self.obs_processor, self.body, self.output_layers) = set_weight_bias(
            key=wb_key,
            network=(self.obs_processor, self.body, self.output_layers),
            weight_init=weights_init,
            bias_init=bias_init,
        )

    def __call__(self, x, *, key: PRNGKeyArray | None = None):
        action_mask = None
        if isinstance(x, jym.AgentObservation):
            action_mask = x.action_mask
            x = x.observation

        x = self.obs_processor(x, key=key)
        x = self.body(x, key=key)
        return self.output_layers(x, action_mask=action_mask, key=key)


class ValueNetwork(eqx.Module):
    obs_processor: OutSizedNetwork
    body: OutSizedNetwork
    output_layers: Network

    def __init__(
        self,
        obs_space: PyTree[jym.Space],
        *,
        key: PRNGKeyArray,
        body: Callable[..., OutSizedNetwork] = MLP,
        obs_architecture_1d: Callable[..., Network] = eqx.nn.Identity,
        obs_architecture_2d: Callable[..., Network] = _DEFAULT_ARCHITECTURE_2D,
        output_layer_type: Callable[..., Network] = eqx.nn.Linear,
        weights_init: Callable[
            ..., jax.nn.initializers.Initializer
        ] = jax.nn.initializers.orthogonal,
        bias_init: float = 0.0,
    ):
        obs_key, body_key, output_key, wb_key = jax.random.split(key, 4)

        self.obs_processor = PyTreeObsSpaceNetwork(
            obs_space,
            key=obs_key,
            architecture_1d=obs_architecture_1d,
            architecture_2d=obs_architecture_2d,
        )

        self.body = body(self.obs_processor.out_features, key=body_key)

        self.output_layers = output_layer_type(
            in_features=self.body.out_features,
            out_features=1,
            key=output_key,
        )

        (self.obs_processor, self.body, self.output_layers) = set_weight_bias(
            key=wb_key,
            network=(self.obs_processor, self.body, self.output_layers),
            weight_init=weights_init,
            bias_init=bias_init,
        )

    def __call__(self, x, *, key: PRNGKeyArray | None = None):
        if isinstance(x, jym.AgentObservation):
            x = x.observation

        x = self.obs_processor(x, key=key)
        x = self.body(x, key=key)
        return self.output_layers(x, key=key).squeeze(-1)


class QValueNetwork(eqx.Module):
    obs_processor: OutSizedNetwork
    body: OutSizedNetwork
    output_layers: PyTreeOutputNetwork

    include_action_in_input: bool = eqx.field(static=True)

    def __init__(
        self,
        obs_space: PyTree[jym.Space],
        output_space: PyTree[jym.Space],
        *,
        key: PRNGKeyArray,
        body: Callable[..., OutSizedNetwork] = MLP,
        obs_architecture_1d: Callable[..., Network] = eqx.nn.Identity,
        obs_architecture_2d: Callable[..., Network] = _DEFAULT_ARCHITECTURE_2D,
        output_layer_type: Callable[..., Network] = eqx.nn.Linear,
        weights_init: Callable[
            ..., jax.nn.initializers.Initializer
        ] = jax.nn.initializers.orthogonal,
        bias_init: float = 0.0,
    ):
        is_continuous = [isinstance(s, jym.Box) for s in jax.tree.leaves(output_space)]
        if any(is_continuous):
            self.include_action_in_input = True
            obs_space = {"_OBSERVATION": obs_space, "_ACTION": output_space}
        else:
            self.include_action_in_input = False

        obs_key, body_key, output_key, wb_key = jax.random.split(key, 4)

        self.obs_processor = PyTreeObsSpaceNetwork(
            obs_space,
            key=obs_key,
            architecture_1d=obs_architecture_1d,
            architecture_2d=obs_architecture_2d,
        )

        self.body = body(self.obs_processor.out_features, key=body_key)

        self.output_layers = PyTreeOutputNetwork.with_raw_outputs()(
            self.body.out_features,
            output_space,
            key=output_key,
            layer_type=output_layer_type,
        )

        (self.obs_processor, self.body, self.output_layers) = set_weight_bias(
            key=wb_key,
            network=(self.obs_processor, self.body, self.output_layers),
            weight_init=weights_init,
            bias_init=bias_init,
        )

    def __call__(
        self, x, action=None, *, key: PRNGKeyArray | None = None
    ) -> Array | PyTree[Array]:
        action_mask = None
        if isinstance(x, jym.AgentObservation):
            action_mask = x.action_mask
            x = x.observation

        if self.include_action_in_input:
            assert action is not None, "Action not provided in continuous Q network."
            x = {"_OBSERVATION": x, "_ACTION": action}

        x = self.obs_processor(x, key=key)
        x = self.body(x, key=key)
        return self.output_layers(x, action_mask=action_mask, key=key)


AdvantageNetwork = QValueNetwork
