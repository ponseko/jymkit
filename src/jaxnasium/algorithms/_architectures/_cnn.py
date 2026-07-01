import logging
from typing import List, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from jaxnasium import Space

logger = logging.getLogger(__name__)


class CNN(eqx.Module):
    """Standard CNN architecture similar to the DQN Nature paper.

    Operates on 2D inputs.
    Assumes channels first format (C, H, W).
    """

    layers: List[eqx.nn.Conv2d]
    in_channels: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    channels_axis: int | None = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        obs_space: Space,
        hidden_sizes: Sequence[int],
        kernel_sizes: Sequence[int],
        strides: Sequence[int],
        padding: Sequence[int],
        **kwargs,
    ):
        assert len(hidden_sizes) == len(kernel_sizes) == len(strides) == len(padding)

        if len(obs_space.shape) == 2:
            logger.warning(
                "2D input without channels, adding leading channels in __call__()"
                "In case the observation should be treated as 1d, use a FlattenObservationWrapper."
            )
            self.channels_axis = None
            in_channels = 1
        elif (
            obs_space.shape[0] == obs_space.shape[1]
            and obs_space.shape[2] != obs_space.shape[0]
        ):
            logger.warning(
                "2D input is in channels last format, moving channels to first dimension"
                "Prefer providing channels first observations (C, H, W)."
            )
            self.channels_axis = -1
            in_channels = obs_space.shape[self.channels_axis]
        else:  # channels first
            self.channels_axis = 0
            in_channels = obs_space.shape[self.channels_axis]

        self.in_channels = in_channels

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
            lambda x: self(x), jnp.zeros(obs_space.shape, dtype=jnp.float32)
        ).shape
        assert len(out_shape) == 1 and out_shape[0] > 0, (
            f"Invalid CNN output (after flattening): {out_shape}. "
            "Perhaps the observation space shape is too small for the CNN architecture."
        )
        self.out_features = out_shape[0]

    def __call__(self, x):
        if self.channels_axis is None:
            x = jnp.expand_dims(x, axis=0)
        elif self.channels_axis == -1:
            x = jnp.moveaxis(x, -1, 0)

        for layer in self.layers:
            x = jax.nn.relu(layer(x))
        x = jnp.reshape(x, -1)
        return x
