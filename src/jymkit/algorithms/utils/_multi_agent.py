import functools
import inspect
from typing import Callable, List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, PyTreeDef


def _result_tuple_to_tuple_result(r):
    """
    Some functions may return tuples. Rather than returning
    a pytree of tuples, we convert it to a tuple of pytrees.
    """
    one_level_leaves, structure = eqx.tree_flatten_one_level(r)
    if isinstance(one_level_leaves[0], tuple):
        tupled = tuple([list(x) for x in zip(*one_level_leaves)])
        r = tuple(jax.tree.unflatten(structure, leaves) for leaves in tupled)
    return r


def _is_prng_key(arg):
    """Check if the argument is a JAX PRNGKeyArray."""
    try:
        jax.random.split(arg)
        return True
    except Exception:
        return False


def split_key_over_agents(key: PRNGKeyArray, agent_structure: PyTreeDef):
    """
    Given a key and a pytree structure, split the key into
    as many keys as there are leaves in the pytree.
    Useful when provided with a flat pytree of agents.

    Similar to `optax.tree_utils.tree_split_key_like`, but operates on PyTreeDefs.

    *Arguments*:
        `key`: A PRNGKeyArray to be split.
        `agent_structure`: A pytree structure of agents.
    """
    num_agents = agent_structure.num_leaves
    keys = list(jax.random.split(key, num_agents))
    return jax.tree.unflatten(agent_structure, keys)


def transform_multi_agent(
    func: Optional[Callable] = None,
    *,
    shared_argnames: Optional[List[str]] = None,
) -> Callable:
    """
    Transformation docorator to handle multi-agent settings.
    Essentially, this function either applies a vmap over the agents (if the agent are homogeneous)
    or applies a `jax.tree.map` over the first level of the PyTree structure of the arguments.

    **Arguments**:
        `func`: The function to be transformed. If None, returns a decorator.
        `shared_argnames`: A optional list of argument names that are shared across agents. If None, the first (non-PRNGKey) argument is
        assumed to be a per-agent argument. All arguments with the same first-level PyTree structure are also considered per-agent arguments.
        The rest are considered shared arguments. PRNG keys that are provided as arguments are automatically split over agents.

    """
    assert callable(func) or func is None

    def _treemap_each_agent(agent_func: Callable, agent_args: dict):
        def map_one_level(f, tree, *rest):
            return jax.tree.map(f, tree, *rest, is_leaf=lambda x: x is not tree)

        return map_one_level(agent_func, *agent_args.values())

    def _vmap_each_agent(agent_func: Callable, agent_args: dict):
        def stack_agents(agent_dict):
            return jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *agent_dict.values())

        dummy = agent_args[list(agent_args.keys())[0]]
        agent_structure = jax.tree.structure(dummy, is_leaf=lambda x: x is not dummy)

        stacked = {k: stack_agents(v) for k, v in agent_args.items()}
        result = jax.vmap(agent_func)(*stacked.values())
        leaves = jax.tree.leaves(result)
        leaves = [list(x) for x in zip(*leaves)]
        leaves = [jax.tree.unflatten(jax.tree.structure(result), x) for x in leaves]
        return jax.tree.unflatten(agent_structure, leaves)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            params = bound.arguments

            shared: set[str] = set(shared_argnames or ())
            shared.update(k for k in ("self", "cls") if k in params)

            if (
                shared_argnames is None
            ):  # We infer the shared arguments from the function signature
                ref_key = next(
                    (
                        k
                        for k in params
                        if k not in shared and not _is_prng_key(params[k])
                    ),
                    None,
                )  # Skip 'self', 'cls', or single-prng keys if present
                if ref_key is None:
                    raise ValueError(
                        "Trying to transform a function without any per-agent arguments. "
                    )
                ref_arg = params[ref_key]
                ref_structure = jax.tree.structure(
                    ref_arg, is_leaf=lambda x: x is not ref_arg
                )

                _shared = {
                    k
                    for k, v in params.items()
                    if ref_structure
                    != jax.tree.structure(v, is_leaf=lambda x: x is not v)
                }
                shared.update(_shared)

                for param in params:  # Auto split JAX PRNG keys for each agent
                    if param not in shared or not _is_prng_key(params[param]):
                        continue
                    # If the parameter is a PRNGKeyArray, we split it over agents
                    params[param] = split_key_over_agents(params[param], ref_structure)
                    shared.remove(param)

            shared_params = {k: params[k] for k in shared}
            per_agent_params = {k: params[k] for k in params if k not in shared}

            # Prepare a function that takes only per-agent args
            def agent_func(*agent_args):
                per_agent_kwargs = dict(zip(per_agent_params.keys(), agent_args))
                return func(**per_agent_kwargs, **shared_params)

            try:
                result = _vmap_each_agent(agent_func, per_agent_params)
            except Exception:
                result = _treemap_each_agent(agent_func, per_agent_params)
            return _result_tuple_to_tuple_result(result)

        return wrapper

    return decorator(func) if callable(func) else decorator
