from abc import ABC, abstractmethod
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, DTypeLike, Int, PRNGKeyArray


class Space(ABC):
    """
    The base class for all spaces in Jaxnasium. Instead of using this class directly,
    use the subclasses `Box`, `Discrete`, and `MultiDiscrete`.
    Composite spaces can be created by simply combining spaces in an arbitrary PyTree.
    For example, a tuple of Box spaces can be created as follows:
    ```python
    from jaxnasium import Box

    box1 = Box(low=0, high=1, shape=(3,))
    box2 = Box(low=0, high=1, shape=(4,))
    box3 = Box(low=0, high=1, shape=(5,))
    composite_space = (box1, box2, box3)
    ```

    Jaxnasium algorithms assume multi-agent environments are such a composite space,
    where the first level of the PyTree is the agent dimension.
    For example, a multi-agent environment observation space may look like this:
    ```python
    from jaxnasium import Box
    from jaxnasium import MultiDiscrete

    agent1_obs = Box(low=0, high=1, shape=(3,))
    agent2_obs = Box(low=0, high=1, shape=(4,))
    agent3_obs = MultiDiscrete(nvec=[2, 3])
    env_obs_space = {
        "agent1": agent1_obs,
        "agent2": agent2_obs,
        "agent3": agent3_obs,
    }
    ```

    Spaces are purposefully not registered PyTree nodes.
    """

    shape: eqx.AbstractVar[tuple[int, ...]]
    dtype: eqx.AbstractVar[DTypeLike]

    @abstractmethod
    def sample(self, rng: PRNGKeyArray) -> Array:
        pass

    # @abstractmethod  # NOTE: Do we need this?
    # def contains(self, x: int) -> bool:
    #     pass


@dataclass
class Box(Space):
    """
    The standard Box space for continuous action/observation spaces.

    **Arguments:**

    - `low` (int / Array[int]): The lower bound of the space.
    - `high` (int / Array[int]): The upper bound of the space.
    - `shape`: The shape of the space.
    - `dtype`: The data type of the space. Default is jnp.float32.
    """

    low: float | ArrayLike = eqx.field(converter=np.asarray, default=0.0)
    high: float | ArrayLike = eqx.field(converter=np.asarray, default=1.0)
    shape: tuple[int, ...] = ()
    dtype: DTypeLike = jnp.float32

    def __post_init__(self):
        if not isinstance(self.shape, tuple):
            self.shape = (self.shape,)

    def sample(self, rng: PRNGKeyArray) -> Array:
        """Sample random action uniformly from set of continuous choices."""
        low = self.low
        high = self.high
        if jnp.isdtype(self.dtype, "real floating"):
            return jax.random.uniform(
                rng, shape=self.shape, minval=low, maxval=high, dtype=self.dtype
            )
        if jnp.isdtype(self.dtype, "bool"):
            return jax.random.bernoulli(rng, 0.5, shape=self.shape)
        return jax.random.randint(
            rng, shape=self.shape, minval=low, maxval=high, dtype=self.dtype
        )


@dataclass
class Discrete(Space):
    """
    The standard discrete space for discrete action/observation spaces.

    **Arguments:**

    - `n` (int): The number of discrete actions.
    - `dtype`: The data type of the space. Default is jnp.int16.
    """

    n: int
    dtype: DTypeLike
    shape: tuple[int, ...] = ()

    def __init__(self, n: int, dtype: DTypeLike = jnp.int32):
        self.n = n
        self.dtype = dtype

    def sample(self, rng: PRNGKeyArray) -> Int[Array, ""]:
        """Sample random action uniformly from set of discrete choices."""
        return jax.random.randint(
            rng,
            shape=self.shape,
            minval=0,
            maxval=jnp.array(self.n, dtype=self.dtype),
            dtype=self.dtype,
        )


@dataclass
class MultiDiscrete(Space):
    """
    The standard multi-discrete space for discrete action/observation spaces.

    **Arguments:**

    - `nvec` (Array[int]): The number of discrete actions for each dimension.
    - `dtype`: The data type of the space. Default is jnp.int32.
    """

    nvec: Int[ArrayLike, " num_actions"]
    dtype: DTypeLike
    shape: tuple[int, ...]

    def __init__(
        self,
        nvec: Int[ArrayLike, " num_actions"],
        dtype: DTypeLike = jnp.int32,
    ):
        self.nvec = nvec
        self.dtype = dtype
        self.shape = (len(np.asarray(nvec).tolist()),)

    def sample(self, rng: PRNGKeyArray) -> Int[Array, ""]:
        """Sample random action uniformly from set of discrete choices."""
        return jax.random.randint(
            rng,
            shape=self.shape,
            minval=0,
            maxval=jnp.array(self.nvec, dtype=self.dtype),
            dtype=self.dtype,
        )
