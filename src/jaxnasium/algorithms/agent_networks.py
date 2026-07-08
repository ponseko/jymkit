import logging
from typing import Any, Callable, Protocol

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray, PyTree

import jaxnasium as jym
from jaxnasium.algorithms import (
    MLP,
    PyTreeObsSpaceNetwork,
    PyTreeOutputNetwork,
    set_weight_bias,
)

logger = logging.getLogger(__name__)


"""
The base Reinforcement Learning Network classes (actor, V-network, Q-network).
Each of these consist of three components:
    - An observation processor (AutoAgentObservationNet)
        This accepts a PyTree of observation spaces and builds a network per observation space.
        In case of a single 1d observation space, this will simply be the Identity network.
        In case of a 2d observation space, this will be a CNN.
        The output of each observation processor is concatenated into a single 1d vector.
    - A MLP =
        This is a simple MLP network that takes the output of the observation processor and passes it through a MLP.
    - An output processor (AutoAgentOutputNet)
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


class ActorNetwork(eqx.Module):
    obs_processor: OutSizedNetwork
    body: OutSizedNetwork
    output_layers: PyTreeOutputNetwork

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        output_space: PyTree[jym.Space],
        obs_space: PyTree[jym.Space],
        obs_processor: Callable[..., OutSizedNetwork] = PyTreeObsSpaceNetwork,
        body: Callable[..., OutSizedNetwork] = MLP,
        output_layers: Callable[..., PyTreeOutputNetwork] = PyTreeOutputNetwork,
        weights_init: jax.nn.initializers.Initializer = jax.nn.initializers.orthogonal(),
        bias_init: float = 0.0,
        **kwargs,
    ):
        obs_key, body_key, output_key, wb_key = jax.random.split(key, 4)
        self.obs_processor = obs_processor(obs_key, obs_space, **kwargs)
        self.body = body(body_key, self.obs_processor.out_features, **kwargs)
        self.output_layers = output_layers(
            output_key, self.body.out_features, output_space, **kwargs
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
        key: PRNGKeyArray,
        *,
        obs_space: PyTree[jym.Space],
        obs_processor: Callable[..., OutSizedNetwork] = PyTreeObsSpaceNetwork,
        body: Callable[..., OutSizedNetwork] = MLP,
        output_layers: Callable[..., Network] = eqx.nn.Linear,
        weights_init: jax.nn.initializers.Initializer = jax.nn.initializers.orthogonal(),
        bias_init: float = 0.0,
        **kwargs,
    ):
        obs_key, body_key, output_key, wb_key = jax.random.split(key, 4)
        self.obs_processor = obs_processor(obs_key, obs_space, **kwargs)
        self.body = body(body_key, self.obs_processor.out_features, **kwargs)
        self.output_layers = output_layers(
            key=output_key, in_features=self.body.out_features, out_features=1, **kwargs
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
        key: PRNGKeyArray,
        *,
        obs_space: PyTree[jym.Space],
        output_space: PyTree[jym.Space],
        obs_processor: Callable[..., OutSizedNetwork] = PyTreeObsSpaceNetwork,
        body: Callable[..., OutSizedNetwork] = MLP,
        output_layers: Callable[..., PyTreeOutputNetwork] = PyTreeOutputNetwork,
        weights_init: jax.nn.initializers.Initializer = jax.nn.initializers.orthogonal(),
        bias_init: float = 0.0,
        **kwargs,
    ):
        is_continuous = [isinstance(s, jym.Box) for s in jax.tree.leaves(output_space)]
        if any(is_continuous):
            self.include_action_in_input = True
            obs_space = {"_OBSERVATION": obs_space, "_ACTION": output_space}

        obs_key, body_key, output_key, wb_key = jax.random.split(key, 4)
        self.obs_processor = obs_processor(obs_key, obs_space, **kwargs)
        self.body = body(body_key, self.obs_processor.out_features, **kwargs)
        self.output_layers = output_layers(output_key, self.body.out_features, **kwargs)

        (self.obs_processor, self.body, self.output_layers) = set_weight_bias(
            key=wb_key,
            network=(self.obs_processor, self.body, self.output_layers),
            weight_init=weights_init,
            bias_init=bias_init,
        )

    def __call__(self, x, action=None):
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
