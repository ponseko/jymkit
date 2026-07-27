from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, PyTree


@eqx.filter_jit
def set_weight_bias(
    key: PRNGKeyArray,
    network: PyTree[eqx.Module],
    weight_init: Callable[..., jax.nn.initializers.Initializer]
    | None = jax.nn.initializers.orthogonal,
    bias_init: float | None = 0.0,
):
    """Sets all `eqx.nn.Linear` and `eqx.nn.Conv` layers in
    a network to a given weight and bias initialization.
    Defaults to orthogonal weight initialization and zero bias initialization.
    Setting weight_init or bias_init to None will leave the weights or biases unchanged.
    """
    is_layer = lambda x: isinstance(x, (eqx.nn.Linear, eqx.nn.Conv))
    layers, network_structure = jax.tree.flatten(network, is_leaf=is_layer)

    new_layers = layers

    # Update bias
    if bias_init is not None:
        new_layers = [
            eqx.tree_at(
                lambda x: x.bias,
                layer,
                replace_fn=lambda x: jnp.ones_like(x) * bias_init,  # type: ignore
            )
            if is_layer(layer) and layer.bias is not None
            else layer
            for layer in layers
        ]

    if weight_init is not None:
        # Update weight
        new_layers = [
            eqx.tree_at(
                lambda x: x.weight,
                layer,
                replace_fn=lambda x: weight_init()(key, x.shape, x.dtype),  # type: ignore
            )
            if is_layer(layer)
            else layer
            for layer in new_layers
        ]

    return jax.tree.unflatten(network_structure, new_layers)


rl_initialization = partial(
    set_weight_bias, weight_init=jax.nn.initializers.orthogonal, bias_init=0.0
)
