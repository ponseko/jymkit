import functools
from typing import Callable, Optional

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
    multi_agent: bool = False,
    shared_argnames: Optional[list[str]] = None,
) -> Callable:
    """
    Decorator to transform a function to work with multiple agents when `multi_agent` is True.
    If `multi_agent` is False, this decorator returns the identity function.

    **Arguments**:
        `func`: The function to be transformed. If None, returns a decorator.
        `multi_agent`: If True, the function will be transformed to work with multiple agents.
        `shared_argnames`: A optional list of argument names that are shared across agents. If None, the first argument is assumed to be a per-agent argument.
        All arguments with the same first-level PyTree structure are also considered per-agent arguments. The rest are considered shared arguments.

    **Usage**:
    ```python
    @transform_multi_agent(multi_agent=is_multi_agent)
    def my_function(...):
        # Function logic here
    ```
    or
    ```python
    def my_function(...):
        # Function logic here
    my_function = transform_multi_agent(my_function, multi_agent=is_multi_agent)
    ```
    """
    assert callable(func) or func is None

    def _treemap_each_agent(agent_func: Callable, agent_args: dict):
        def map_one_level(f, tree, *rest):
            # NOTE: Immidiately self-referential trees may pose a problem.
            # see eqx.tree_flatten_one_level
            # Likely not a problem here.
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
            if not multi_agent:
                return func(*args, **kwargs)

            # Map positional args to their respective keyword arguments
            kw_args = list(func.__code__.co_varnames[: func.__code__.co_argcount])
            for i, arg in enumerate(args):
                if kw_args[i] in kwargs:
                    raise ValueError(f"Duplicate argument: {kw_args[i]}")
                if i < len(kw_args):
                    kwargs[kw_args[i]] = arg

            def maybe_infer_shared_argnames():
                """
                Infer shared argument names based on the first argument's structure.
                If `shared_argnames` is provided, it will be used directly.
                Otherwise, it will infer from the first argument's structure.
                """
                if shared_argnames is not None:
                    return shared_argnames
                first_arg = kwargs.get(kw_args[0])
                agent_structure = jax.tree.structure(
                    first_arg, is_leaf=lambda x: x is not first_arg
                )
                return [
                    k
                    for k in kwargs.keys()
                    if agent_structure
                    != jax.tree.structure(
                        kwargs[k], is_leaf=lambda x: x is not kwargs[k]
                    )
                ]

            # Separate shared and per-agent args
            shared_args = {
                k: v for k, v in kwargs.items() if k in maybe_infer_shared_argnames()
            }
            per_agent_args = {
                k: v
                for k, v in kwargs.items()
                if k not in maybe_infer_shared_argnames()
            }

            # Prepare a function that takes only per-agent args
            def agent_func(*agent_args):
                per_agent_kwargs = dict(zip(per_agent_args.keys(), agent_args))
                return func(**per_agent_kwargs, **shared_args)

            try:
                result = _vmap_each_agent(agent_func, per_agent_args)
            except Exception:
                result = _treemap_each_agent(agent_func, per_agent_args)
            return _result_tuple_to_tuple_result(result)

        return wrapper

    return decorator(func) if callable(func) else decorator
