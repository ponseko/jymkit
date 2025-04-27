from abc import ABC, abstractmethod
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PRNGKeyArray


@dataclass
class Space(ABC):
    @abstractmethod
    def sample(self, rng: PRNGKeyArray) -> Array:
        pass

    @abstractmethod
    def contains(self, x: int) -> bool:
        pass


@dataclass
class Box(Space):
    """
    Box space.
    """

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

    def contains(self, x: float) -> bool:
        """Check if x is in the box space."""
        return bool(np.all(np.logical_and(x >= self.low, x <= self.high)))


@dataclass
class BoxWrapper(Space):
    """
    Wrapper for Box spaces. Preferably, any space should subclass this if it is not a Box.
    Alternatively, a Space may be a PyTree of Box / BoxWrapper.
    This is useful for creating spaces that are not Box, but can be represented as a Box.
    """

    @property
    @abstractmethod
    def _box(self) -> Box:
        """Return the Box representation of this space."""
        pass

    def sample(self, rng: PRNGKeyArray) -> Array:
        return self._box.sample(rng)

    def contains(self, x: float) -> bool:
        """Check if x is in the box space."""
        return self._box.contains(x)

    def __getattr__(self, name: str):
        """Get the attribute from the Box space."""
        return getattr(self._box, name)


@dataclass
class Discrete(BoxWrapper):
    """
    Convenience class for discrete spaces.
    Implemented as a Box with low=0 and high=n of shape ().
    """

    n: int

    @property
    def _box(self) -> Box:
        return Box(low=0, high=self.n, shape=(), dtype=jnp.int16)


@dataclass
class MultiDiscrete(BoxWrapper):
    """
    Convenience class for multi-discrete spaces.
    Implemented as a Box with low=0 and high=nvec of shape (len(nvec),).
    """

    nvec: Array

    @property
    def _box(self) -> Box:
        return Box(low=0, high=self.nvec, shape=(len(self.nvec),), dtype=jnp.int16)
