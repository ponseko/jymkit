import logging
from functools import partial
from typing import Any, Callable, Protocol

import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray, PyTree

import jaxnasium as jym
from jaxnasium.algorithms.architectures import MLP
from jaxnasium.algorithms.core import (
    PyTreeObsSpaceNetwork,
    PyTreeOutputNetwork,
    set_weight_bias,
)

logger = logging.getLogger(__name__)

_QVALUE_OUTPUT_LAYERS = partial(
    PyTreeOutputNetwork,
    discrete_distribution=None,
    continuous_distribution=None,
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


class Network(Protocol):
    """Any module with a __call__ defined"""

    def __call__(self, *args, **kwargs) -> Any: ...


class OutSizedNetwork(Network, Protocol):
    """Any module with a __call__ defined and an out_features attribute"""

    out_features: int


def _split_network_kwargs(
    network_kwargs: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    network_kwargs = network_kwargs or {}
    return (
        network_kwargs.get("obs", {}),
        network_kwargs.get("shared", {}),
        network_kwargs.get("output", {}),
    )


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
        obs_processor: Callable[..., OutSizedNetwork] = PyTreeObsSpaceNetwork,
        body: Callable[..., OutSizedNetwork] = MLP,
        output_layers: Callable[..., PyTreeOutputNetwork] = PyTreeOutputNetwork,
        weights_init: jax.nn.initializers.Initializer = jax.nn.initializers.orthogonal(),
        bias_init: float = 0.0,
        **network_kwargs: dict[str, Any],
    ):
        obs_kwargs, body_kwargs, output_kwargs = _split_network_kwargs(network_kwargs)

        obs_key, body_key, output_key, wb_key = jax.random.split(key, 4)
        self.obs_processor = obs_processor(obs_space, key=obs_key, **obs_kwargs)
        self.body = body(self.obs_processor.out_features, key=body_key, **body_kwargs)
        self.output_layers = output_layers(
            self.body.out_features, output_space, key=output_key, **output_kwargs
        )

        (self.obs_processor, self.body, self.output_layers) = set_weight_bias(
            key=wb_key,
            network=(self.obs_processor, self.body, self.output_layers),
            weight_init=weights_init,
            bias_init=bias_init,
        )

    def __call__(self, x):
        action_mask = None
        if isinstance(x, jym.AgentObservation):
            action_mask = x.action_mask
            x = x.observation

        x = self.obs_processor(x)
        x = self.body(x)
        return self.output_layers(x, action_mask=action_mask)


class ValueNetwork(eqx.Module):
    obs_processor: OutSizedNetwork
    body: OutSizedNetwork
    output_layers: Network

    def __init__(
        self,
        obs_space: PyTree[jym.Space],
        *,
        key: PRNGKeyArray,
        obs_processor: Callable[..., OutSizedNetwork] = PyTreeObsSpaceNetwork,
        body: Callable[..., OutSizedNetwork] = MLP,
        output_layers: Callable[..., Network] = eqx.nn.Linear,
        weights_init: jax.nn.initializers.Initializer = jax.nn.initializers.orthogonal(),
        bias_init: float = 0.0,
        **network_kwargs: dict[str, Any],
    ):
        obs_kwargs, body_kwargs, output_kwargs = _split_network_kwargs(network_kwargs)

        obs_key, body_key, output_key, wb_key = jax.random.split(key, 4)
        self.obs_processor = obs_processor(obs_space, key=obs_key, **obs_kwargs)
        self.body = body(self.obs_processor.out_features, key=body_key, **body_kwargs)
        self.output_layers = output_layers(
            in_features=self.body.out_features,
            out_features=1,
            key=output_key,
            **output_kwargs,
        )

        (self.obs_processor, self.body, self.output_layers) = set_weight_bias(
            key=wb_key,
            network=(self.obs_processor, self.body, self.output_layers),
            weight_init=weights_init,
            bias_init=bias_init,
        )

    def __call__(self, x):
        if isinstance(x, jym.AgentObservation):
            x = x.observation

        x = self.obs_processor(x)
        x = self.body(x)
        return self.output_layers(x).squeeze(-1)


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
        obs_processor: Callable[..., OutSizedNetwork] = PyTreeObsSpaceNetwork,
        body: Callable[..., OutSizedNetwork] = MLP,
        output_layers: Callable[..., PyTreeOutputNetwork] = _QVALUE_OUTPUT_LAYERS,
        weights_init: jax.nn.initializers.Initializer = jax.nn.initializers.orthogonal(),
        bias_init: float = 0.0,
        network_kwargs: dict[str, Any] | None = None,
    ):
        is_continuous = [isinstance(s, jym.Box) for s in jax.tree.leaves(output_space)]
        if any(is_continuous):
            self.include_action_in_input = True
            obs_space = {"_OBSERVATION": obs_space, "_ACTION": output_space}
        else:
            self.include_action_in_input = False

        obs_kwargs, body_kwargs, output_kwargs = _split_network_kwargs(network_kwargs)

        obs_key, body_key, output_key, wb_key = jax.random.split(key, 4)
        self.obs_processor = obs_processor(obs_space, key=obs_key, **obs_kwargs)
        self.body = body(self.obs_processor.out_features, key=body_key, **body_kwargs)
        self.output_layers = output_layers(
            self.body.out_features,
            output_space,
            key=output_key,
            **output_kwargs,
        )

        (self.obs_processor, self.body, self.output_layers) = set_weight_bias(
            key=wb_key,
            network=(self.obs_processor, self.body, self.output_layers),
            weight_init=weights_init,
            bias_init=bias_init,
        )

    def __call__(self, x, action=None) -> Array | PyTree[Array]:
        action_mask = None
        if isinstance(x, jym.AgentObservation):
            action_mask = x.action_mask
            x = x.observation

        if self.include_action_in_input:
            assert action is not None, "Action not provided in continuous Q network."
            x = {"_OBSERVATION": x, "_ACTION": action}

        x = self.obs_processor(x)
        x = self.body(x)
        return self.output_layers(x, action_mask=action_mask)


AdvantageNetwork = QValueNetwork
