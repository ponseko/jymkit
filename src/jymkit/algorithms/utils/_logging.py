import functools
from typing import Callable, Literal, Optional

import jax


def scan_callback(
    func: Optional[Callable] = None,
    callback_fn: Optional[Callable | Literal["tqdm", "simple"]] = None,
    callback_interval: int | float = 20,
    n: Optional[int] = None,
) -> Callable:
    assert callable(func) or func is None

    assert callback_interval > 0, "callback_interval must be greater than 0"
    if callback_interval < 1:
        assert n is not None, "n must be provided if callback_interval is less than 1"
        callback_interval = int(n * callback_interval)

    if callback_fn == "tqdm":
        try:
            import tqdm.auto
        except ImportError:
            raise ImportError(
                "Ltqdm is not installed. Please install it with `pip install tqdm`."
            )

        progress_bar = []

        def update_tqdm_bar(_, iteration):
            if iteration == 0:
                progress_bar.append(
                    tqdm.auto.tqdm(
                        total=n, desc="Training Progress", unit=" iterations "
                    )
                )

            progress_bar[0].update(callback_interval)

    def simple_reward_logger(data, iteration):
        assert (
            "returned_episode_returns" in data
            and "returned_episode" in data
            and "timestep" in data
        ), "Missing keys in logging data. Is the environment wrapped with LogWrapper?"

        num_envs = data["timestep"].shape[-1]
        return_values = jax.tree.map(
            lambda x: x[data["returned_episode"]], data["returned_episode_returns"]
        )
        timesteps = data["timestep"][data["returned_episode"]] * num_envs
        for t in range(len(timesteps)):
            return_values_t = jax.tree.map(
                lambda x: x[t].item() if hasattr(x[t], "item") else x[t], return_values
            )
            return_values_t = jax.tree.map(lambda x: round(x, 3), return_values_t)
            print(f"global step={timesteps[t]}, episodic return={return_values_t}")

    def maybe_log(iteration: int, data):
        if callback_fn is not None and callback_interval > 0:
            if callback_fn == "tqdm":
                log_fn = update_tqdm_bar
            elif callback_fn == "simple":
                log_fn = simple_reward_logger
            else:
                log_fn = callback_fn

            _ = jax.lax.cond(
                iteration % callback_interval == 0,
                lambda: jax.debug.callback(
                    lambda d, i: log_fn(d, i) if callable(log_fn) else None,
                    data,
                    iteration,
                ),
                lambda: None,
            )

    def _scan_callback(func):
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

    return _scan_callback(func) if callable(func) else _scan_callback
