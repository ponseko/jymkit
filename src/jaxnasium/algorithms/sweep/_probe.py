import logging
import warnings
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp

logger = logging.getLogger(__name__)


def _is_traceable(fn: Callable[..., Any], kwargs: dict[str, Any], name: str) -> bool:
    """Whether `fn` can be traced with `kwargs[name]` replaced by a JAX tracer.

    Used essentially to check if we can vmap over the parameter.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eqx.filter_make_jaxpr(lambda traced: fn(**{**kwargs, name: traced}))(
                jnp.asarray(kwargs[name])
            )
    except (ValueError, TypeError) as e:
        logger.debug(f"Param {name!r} is static: {type(e).__name__}: {e}")
        return False
    return True


def _get_last(spec: list | tuple) -> Any:
    """Grabs the last value. Tuples may be given as
    (first, last, [space]), hence we check for that."""
    if isinstance(spec, tuple) and len(spec) == 3:
        return spec[1]
    return spec[-1]


def split_static_dynamic_params(
    fn: Callable[..., Any],
    params: dict[str, list | tuple],
) -> tuple[dict[str, list | tuple], dict[str, list | tuple]]:
    """Split sweep params into static and dynamic groups by tracing `fn`.

    *Static* params change the computation graph when varied, so each value needs its own job.
    *Dynamic* params only feed data into a fixed graph and can be batched with `vmap`.

    Note that some params might be dynamic under some other parameters (perhaps because they
    have no effect), but should be static under other parameters. This function attempts to
    identify these cases and return parameters as static whenever *any* combination would
    mark the paramater as static. However, as probing every possible combination could be
    very expensive in large spaces, we only probe a subset.

    **Arguments**:
        `fn`: Function to probe. Traced, never executed for real.
        `params`: param names to a sequence of values / range specs.

    **Returns**:
        The static params and the dynamic params, as two dicts.
    """
    static_params: dict[str, list | tuple] = {}
    dynamic_candidates: dict[str, list | tuple] = {}
    for name, spec in params.items():
        try:
            jnp.asarray(spec[0] if isinstance(spec, tuple) else spec)
        except TypeError:
            static_params[name] = spec
        else:
            dynamic_candidates[name] = spec

    if not dynamic_candidates:
        return static_params, {}

    # Probe fn with each parameter at its first value and each parameter as its last value.
    first_value_baseline = {name: spec[0] for name, spec in params.items()}
    last_value_baseline = {name: _get_last(spec) for name, spec in params.items()}
    contexts = [
        first_value_baseline,
        last_value_baseline,
    ]
    # NOTE: disabled probing more params
    # for name, spec in static_params.items():
    #     for i, value in enumerate(list(spec)):
    #         if i == 0 or i == len(spec) - 1:
    #             continue  # already added
    # contexts.append({**first_value_baseline, name: value})
    # contexts.append({**last_value_baseline, name: value})

    dynamic_params: dict[str, list | tuple] = {}
    for name, spec in dynamic_candidates.items():
        if not all(_is_traceable(fn, ctx, name) for ctx in contexts):
            static_params[name] = spec
        else:
            dynamic_params[name] = spec

    return static_params, dynamic_params
