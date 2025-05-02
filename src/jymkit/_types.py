from typing import NamedTuple

from jaxtyping import Array, Bool, Float, PyTree


class TimeStep(NamedTuple):
    """
    A container for the output of the step function.
    """

    observation: Array | PyTree
    reward: Float[Array, "..."] | PyTree[Float[Array, "..."]]
    terminated: Bool[Array, "..."] | PyTree[Bool[Array, "..."]]
    truncated: Bool[Array, "..."] | PyTree[Bool[Array, "..."]]
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
