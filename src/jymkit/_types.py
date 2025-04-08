from typing import NamedTuple, TypeVar

import equinox as eqx
from jaxtyping import Array

Observation = TypeVar("Observation")
Action = TypeVar("Action")
EnvState = TypeVar("EnvState", eqx.Module, dict)


class TimeStep(NamedTuple):
    """
    A container for the output of the step function.
    """

    observation: Observation
    reward: Array
    terminated: Array
    truncated: Array
    info: dict


class AgentObservation(NamedTuple):
    """
    A container for observations from a single agent.
    While this container is not required for most settings, it is useful for environments with action masking.
    jymkit.algorithms expect the output of `get_observation` to be of this type when
    action masking is included in the environment.
    """

    observation: Observation
    action_mask: Array = None
