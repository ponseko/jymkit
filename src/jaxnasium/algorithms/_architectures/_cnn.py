import logging
from functools import partial
from typing import Callable, List, Literal, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

logger = logging.getLogger(__name__)


class CNN(eqx.Module):
    """Standard CNN architecture similar to the DQN Nature paper.

    Operates on 2D inputs.
    Assumes channels first format (C, H, W).

    Flattens the output of the CNN into a 1d vector.
    """

    layers: List[eqx.nn.Conv2d]
    in_channels: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    channels_axis: Literal["first", "last"] = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        hidden_sizes: Sequence[int],
        kernel_sizes: Sequence[int],
        strides: Sequence[int],
        padding: Sequence[int],
        input_shape: tuple[int, int, int],
        channels_axis: Literal["first", "last"] = "first",
        activation: Callable = jax.nn.relu,
        **kwargs,
    ):
        assert len(hidden_sizes) == len(kernel_sizes) == len(strides) == len(padding)

        if channels_axis == "last":
            logger.warning(
                "2D input is in channels last format, moving channels to first dimension"
                "Prefer providing channels first observations (C, H, W)."
            )

        if self.channels_axis == "first":
            in_channels = input_shape[0]
        elif self.channels_axis == "last":
            in_channels = input_shape[-1]
        else:
            raise ValueError(f"Invalid channels axis: {self.channels_axis}")

        self.activation = activation

        self.layers = []
        keys = jax.random.split(key, len(hidden_sizes))
        for i, hidden_size in enumerate(hidden_sizes):
            self.layers.append(
                eqx.nn.Conv2d(
                    in_channels,
                    hidden_size,
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=padding[i],
                    key=keys[i],
                )
            )
            in_channels = hidden_size

        out_shape = jax.eval_shape(
            lambda x: self(x), jnp.zeros(input_shape, dtype=jnp.float32)
        ).shape
        assert len(out_shape) == 1 and out_shape[0] > 0, (
            f"Invalid CNN output (after flattening): {out_shape}. "
            "Perhaps the observation space shape is too small for the CNN architecture."
        )
        self.out_features = out_shape[0]

    def __call__(self, x):
        if self.channels_axis == "last":
            x = jnp.moveaxis(x, -1, 0)

        for layer in self.layers:
            x = self.activation(layer(x))
        x = jnp.reshape(x, -1)
        return x

    @classmethod
    def with_params(
        cls,
        hidden_sizes=(32, 64, 64),
        kernel_sizes=(3, 3, 2),
        strides=(1, 1, 1),
        padding=(0, 0, 0),
        activation=jax.nn.relu,
        **kwargs,
    ):
        return partial(
            cls,
            hidden_sizes=hidden_sizes,
            kernel_sizes=kernel_sizes,
            strides=strides,
            padding=padding,
            activation=activation,
            **kwargs,
        )
