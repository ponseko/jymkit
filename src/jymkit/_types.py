from typing import NamedTuple

from jaxtyping import Array, PyTree


class TimeStep(NamedTuple):
    """
    A container for the output of the step function.
    """

    observation: Array | PyTree
    reward: float | PyTree[float]
    terminated: bool | Array | PyTree
    truncated: bool | Array | PyTree
    info: dict


class AgentObservation(NamedTuple):
    """
    A container for observations from a single agent.
    While this container is not required for most settings, it is useful for environments with action masking.
    jymkit.algorithms expect the output of `get_observation` to be of this type when
    action masking is included in the environment.
    """

    observation: Array | PyTree
    action_mask: Array | PyTree | None = None
