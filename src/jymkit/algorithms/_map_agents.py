import functools
from typing import Callable, Literal, Optional

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
        r = tuple(jax.tree_unflatten(structure, leaves) for leaves in tupled)
    return r


def split_key_over_agents(key: PRNGKeyArray, agent_structure: PyTreeDef):
    """
    Given a key and a pytree structure, split the key into
    as many keys as there are leaves in the pytree.
    Useful when provided with a flat pytree of agents.

    *Arguments*:
        `key`: A PRNGKeyArray to be split.
        `agent_structure`: A pytree structure of agents.
    """
    num_agents = agent_structure.num_leaves
    keys = list(jax.random.split(key, num_agents))
    return jax.tree.unflatten(agent_structure, keys)


def map_each_agent(
    func: Optional[Callable] = None,
    shared_argnames: list[str] = [],
    identity: bool = False,
) -> Callable:
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
            if identity:
                return func(*args, **kwargs)

            # Map positional args to their respective keyword arguments
            kw_args = list(func.__code__.co_varnames[: func.__code__.co_argcount])
            for i, arg in enumerate(args):
                if kw_args[i] in kwargs:
                    raise ValueError(f"Duplicate argument: {kw_args[i]}")
                if i < len(kw_args):
                    kwargs[kw_args[i]] = arg

            # Separate shared and per-agent args
            shared_args = {k: v for k, v in kwargs.items() if k in shared_argnames}
            per_agent_args = {
                k: v for k, v in kwargs.items() if k not in shared_argnames
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


def scan_logger(
    func: Optional[Callable] = None,
    log_function: Optional[Callable | Literal["tqdm", "simple"]] = None,
    log_interval: int | float = 20,
    n: Optional[int] = None,
) -> Callable:
    assert callable(func) or func is None

    assert log_interval > 0, "log_interval must be greater than 0"
    if log_interval < 1:
        assert n is not None, "n must be provided if log_interval is less than 1"
        log_interval = int(n * log_interval)

    if log_function == "tqdm":
        import tqdm.auto

        progress_bar = []

        def update_tqdm_bar(_, iteration):
            if iteration == 0:
                progress_bar.append(
                    tqdm.auto.tqdm(
                        total=n, desc="Training Progress", unit=" iterations "
                    )
                )

            progress_bar[0].update(log_interval)

    def simple_reward_logger(data, iteration):
        if (
            "returned_episode_returns" not in data
            or "returned_episode" not in data
            or "timestep" not in data
        ):
            raise ValueError(
                "Missing keys in logging data. Is the environment wrapped with LogWrapper?"
            )

        num_envs = data["timestep"].shape[-1]
        return_values = data["returned_episode_returns"][data["returned_episode"]]
        timesteps = data["timestep"][data["returned_episode"]] * num_envs
        for t in range(len(timesteps)):
            print(f"global step={timesteps[t]}, episodic return={return_values[t]}")

    def maybe_log(iteration: int, data):
        if log_function is not None and log_interval > 0:
            if log_function == "tqdm":
                log_fn = update_tqdm_bar
            elif log_function == "simple":
                log_fn = simple_reward_logger
            else:
                log_fn = log_function

            _ = jax.lax.cond(
                iteration % log_interval == 0,
                lambda: jax.debug.callback(
                    lambda d, i: log_fn(d, i) if callable(log_fn) else None,
                    data,
                    iteration,
                ),
                lambda: None,
            )

    def _scan_logger(func):
        @functools.wraps(func)
        def wrapper(carry, x):
            if type(x) is tuple:
                iter_num, *_ = x
            else:
                iter_num = x

            result = func(carry, x)
            this_iter_metrics = result[1]
            maybe_log(iter_num, this_iter_metrics)
            return result

        return wrapper

    return _scan_logger(func) if callable(func) else _scan_logger
