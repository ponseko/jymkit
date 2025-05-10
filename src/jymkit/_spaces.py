from abc import ABC, abstractmethod
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Int, PRNGKeyArray

"""
    Space definitions for JymKit. Spaces are purposefully not registered PyTree nodes.
    Composite spaces can be created by simply combining these spaces in a PyTree.
    For example, a tuple of Box spaces can be created as follows:
    ```python
    from jymkit import Box
    
    box1 = Box(low=0, high=1, shape=(3,))
    box2 = Box(low=0, high=1, shape=(4,))
    box3 = Box(low=0, high=1, shape=(5,))
    composite_space = (box1, box2, box3)
    ```

    JymKit algorithms assume multi-agent environments are such a composite space.
    where the first level of the PyTree is the agent dimension.
    For example, a multi-agent observation space can be created as follows:
    ```python
    from jymkit import Box
    from jymkit import MultiDiscrete

    agent1_obs = Box(low=0, high=1, shape=(3,))
    agent2_obs = Box(low=0, high=1, shape=(4,))
    agent3_obs = MultiDiscrete(nvec=[2, 3])
    env_obs_space = {
        "agent1": agent1_obs,
        "agent2": agent2_obs,
        "agent3": agent3_obs,
    }
    ```
    
"""


class Space(ABC):
    shape: eqx.AbstractVar[tuple[int, ...]]

    @abstractmethod
    def sample(self, rng: PRNGKeyArray) -> Array:
        pass

    # @abstractmethod  # NOTE: Do we need this?
    # def contains(self, x: int) -> bool:
    #     pass


@dataclass
class Box(Space):
    """The standard Box space for continuous action/observation spaces."""

    low: float | Array = eqx.field(converter=np.asarray, default=0.0)
    high: float | Array = eqx.field(converter=np.asarray, default=1.0)
    shape: tuple[int, ...] = ()
    dtype: type = jnp.float32

    def sample(self, rng: PRNGKeyArray) -> Array:
        """Sample random action uniformly from set of continuous choices."""
        low = self.low
        high = self.high
        if np.issubdtype(self.dtype, jnp.integer):
            high += 1
            return jax.random.randint(
                rng, shape=self.shape, minval=low, maxval=high, dtype=self.dtype
            )
        return jax.random.uniform(
            rng, shape=self.shape, minval=low, maxval=high, dtype=self.dtype
        )


@dataclass
class Discrete(Space):
    """The standard discrete space for discrete action/observation spaces."""

    n: int
    dtype: type
    shape: tuple[int, ...] = ()

    def __init__(self, n: int, dtype: type = jnp.int16):
        self.n = n
        self.dtype = dtype

    def sample(self, rng: PRNGKeyArray) -> Int[Array, ""]:
        """Sample random action uniformly from set of discrete choices."""
        return jax.random.randint(
            rng, shape=self.shape, minval=0, maxval=self.n, dtype=self.dtype
        )


@dataclass
class MultiDiscrete(Space):
    """
    MultiDiscrete space for discrete action/observation spaces.
    This is a vector of discrete spaces, each with its own number of actions.
    For example, a MultiDiscrete space with nvec=[2, 3] has two discrete actions:
    - The first action has 2 options (0, 1)
    - The second action has 3 options (0, 1, 2)
    """

    nvec: Int[Array | np.ndarray, " num_actions"]
    dtype: type
    shape: tuple[int, ...]

    def __init__(
        self, nvec: Int[Array | np.ndarray, " num_actions"], dtype: type = jnp.int16
    ):
        self.nvec = nvec
        self.dtype = dtype
        self.shape = (len(nvec),)

    def sample(self, rng: PRNGKeyArray) -> Int[Array, ""]:
        """Sample random action uniformly from set of discrete choices."""
        return jax.random.randint(
            rng, shape=self.shape, minval=0, maxval=self.nvec, dtype=self.dtype
        )
