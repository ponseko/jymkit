import logging
from functools import partial
from typing import Callable, List, Sequence

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray
from typing_extensions import Self

logger = logging.getLogger(__name__)


class MLP(eqx.Module):
    """Simple MLP architecture. Final hidden size is the output features."""

    layers: List[eqx.nn.Linear]
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    hidden_sizes: Sequence[int] = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        *,
        key: PRNGKeyArray,
        hidden_sizes: Sequence[int] = (128, 128),
        activation: Callable = jax.nn.relu,
    ):
        depth = len(hidden_sizes) + 1
        keys = jax.random.split(key, depth + 1)
        self.in_features = in_features
        self.hidden_sizes = hidden_sizes
        self.out_features = hidden_sizes[-1]
        self.activation = activation

        self.layers = []
        for i, hidden_dim in enumerate(hidden_sizes):
            self.layers.append(
                eqx.nn.Linear(
                    in_features=in_features, out_features=hidden_dim, key=keys[i]
                )
            )
            in_features = hidden_dim

    def __call__(self, x, *, key: PRNGKeyArray | None = None):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x, key=key))
        return self.layers[-1](x, key=key)

    @classmethod
    def with_params(
        cls,
        *,
        hidden_sizes: Sequence[int] = (128, 128),
        activation: Callable = jax.nn.relu,
    ) -> Callable[..., Self]:
        return partial(cls, hidden_sizes=hidden_sizes, activation=activation)
