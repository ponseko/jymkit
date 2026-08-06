from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
from jaxtyping import PRNGKeyArray
from scipy.stats import qmc

from ._sweep import Sweep


def _scale_to_spec(spec: list | tuple, u: float) -> Any:
    """Maps a ranomly drawn `u` in [0, 1) onto the specified spec.
    A spec is a list of choices, or a tuple of (low, high, ["linear", "log"])

    Examples:
    >>> _scale_to_spec(["a", "b", "c"], 0.5)
    "b"

    >>> _scale_to_spec((1, 5, "linear"), 0.5)
    3 # int input maps to int output

    >>> _scale_to_spec((1.0, 5.0, "linear"), 0.5)
    3.0 # float input maps to float output

    >>> _scale_to_spec((1, 5, "log"), 0.5)
    2.2360680103302

    """
    is_range = isinstance(spec, tuple) and len(spec) in (2, 3)
    is_choice = isinstance(spec, list) and len(spec) > 0
    assert is_range or is_choice, (
        f"Search params must be a non-empty list or a "
        f"(low, high) / (low, high, scale) tuple, got {spec!r}"
    )

    if isinstance(spec, list):
        return spec[int(u * len(spec))]

    if len(spec) == 2:
        low, high = spec
        scale = "linear"
    elif len(spec) == 3:
        low, high, scale = spec
        if scale not in ("linear", "log"):
            raise ValueError(f"Range scale must be 'linear' or 'log', got {scale!r}")
    else:
        raise ValueError(
            f"Range must be (low, high) or (low, high, scale), got {spec!r}"
        )

    if scale == "log":
        if low <= 0 or high <= 0:
            raise ValueError(f"log scale requires positive bounds, got ({low}, {high})")
        log_low, log_high = np.log(low), np.log(high)
        return float(np.exp(log_low + u * (log_high - log_low)))

    if isinstance(low, int) and isinstance(high, int):
        return low + int(u * (high - low))
    return low + u * (high - low)


@dataclass(frozen=True)
class RandomSearch:
    """`num_samples` independent draws from the given parameters.

    Pass as a stage to [`Sweep`][jaxnasium.sweep.Sweep], alone or
    chained with other searches. For only performing this random search,
    `RandomSearch(...).sweep(fn, ...)` is a shorthand for `Sweep(fn, self, ...)`.
    A `Sweep` object allows batching jobs together in various `vmap` calls. See
    [`Sweep`][jaxnasium.sweep.Sweep] for more details.


    **Arguments**:
        `params`: Maps param name to either:
            - a non-empty list to sample uniformly from
              (`{"env": ["env1", "env2", "env3"]}`), or
            - a `(low, high)` or `(low, high, scale)` range, where `scale` is `"linear"` (default) or `"log"`
              (`{"learning_rate": (1e-4, 1e-2, "log")}`).
        `num_samples`: Number of configurations to draw.
        `fixed_seed`: When chaining this search with other search in a `Sweep` object,
        fixing this seed will ensure each configuration of this search uses the same random draws.

    **Examples**:

    ```python
    sweep = RandomSearch(
        {"learning_rate": (1e-4, 1e-2, "log"), "gamma": [0.98, 0.99]},
        num_samples=64,
    ).sweep(train, seed=jax.random.PRNGKey(0), batch_size=16)
    for run in sweep:
        for r in run():
            print(r.arguments, r.result)
    ```

    Combined with a grid search over environments (same learning rates in each env):

    ```python
    sweep = Sweep(
        train,
        GridSearch({"env_name": ["CartPole-v1", "Acrobot-v1"]}),
        RandomSearch(
            {"learning_rate": (1e-4, 1e-2, "log")},
            num_samples=64,
            fixed_seed=jax.random.PRNGKey(1),
        ),
        seed=jax.random.PRNGKey(0),
    )
    for run in sweep:
        for r in run():
            print(r.arguments, r.result)
    ```
    """

    params: dict[str, list | tuple]
    num_samples: int
    fixed_seed: PRNGKeyArray | None = None

    def __post_init__(self):
        assert self.params, "RandomSearch params cannot be empty"
        assert self.num_samples > 0, "num_samples must be positive"
        # Eager validation check of each param:
        [_scale_to_spec(value, 0.0) for value in self.params.values()]

    def configs(self, seed: PRNGKeyArray) -> list[dict[str, Any]]:
        """Draw `num_samples` configurations"""
        random_u_samples = jax.random.uniform(
            seed, (self.num_samples, len(self.params))
        )
        random_u_samples = np.asarray(random_u_samples)
        names = list(self.params)
        return [
            {
                name: _scale_to_spec(self.params[name], float(u))
                for name, u in zip(names, row)
            }
            for row in random_u_samples
        ]

    def sweep(
        self,
        fn: Callable,
        *,
        seed: PRNGKeyArray | None = None,
        batch_size: int | None = None,
        print_cost_estimate: bool = False,
    ) -> Sweep:
        """Create a [`Sweep`][jaxnasium.sweep.Sweep] with this search alone."""
        return Sweep(
            fn,
            self,
            seed=seed,
            batch_size=batch_size,
            print_cost_estimate=print_cost_estimate,
        )


@dataclass(frozen=True)
class SobolSearch:
    """Quasi-random draws that cover the parameter space more evenly than RandomSearch.

    Pass as a stage to [`Sweep`][jaxnasium.sweep.Sweep], alone or
    chained with other searches. For only performing this Sobol search,
    `SobolSearch(...).sweep(fn, ...)` is a shorthand for `Sweep(fn, self, ...)`.
    A `Sweep` object allows batching jobs together in various `vmap` calls. See
    [`Sweep`][jaxnasium.sweep.Sweep] for more details.

    **Arguments**:
        `params`: Maps param name to either:
            - a non-empty list to sample uniformly from
              (`{"env": ["env1", "env2", "env3"]}`), or
            - a `(low, high)` or `(low, high, scale)` range, where `scale` is
              `"linear"` (default) or `"log"`
              (`{"learning_rate": (1e-4, 1e-2, "log")}`).
        `num_samples`: Number of configurations to draw. Prefer a power of two for this value.
        `fixed_seed`: When chaining this search with other search in a `Sweep` object,
        fixing this seed will ensure each configuration of this search uses the same random draws.


    **Examples**:

    ```python
    sweep = SobolSearch(
        {"learning_rate": (1e-4, 1e-2, "log"), "gamma": (0.9, 1.0)},
        num_samples=64,
    ).sweep(train, seed=jax.random.PRNGKey(0), batch_size=16)
    for run in sweep:
        for r in run():
            print(r.arguments, r.result)
    ```

    Combined with a grid search over environments:

    ```python
    sweep = Sweep(
        train,
        GridSearch({"env_name": ["CartPole-v1", "Acrobot-v1"]}),
        SobolSearch(
            {"learning_rate": (1e-4, 1e-2, "log"), "gamma": (0.9, 1.0)},
            num_samples=64,
            fixed_seed=jax.random.PRNGKey(1),
        ),
        seed=jax.random.PRNGKey(0),
        batch_size=16,
    )
    for run in sweep:
        for r in run():
            print(r.arguments, r.result)
    ```
    """

    params: dict[str, list | tuple]
    num_samples: int
    fixed_seed: PRNGKeyArray | None = None

    def __post_init__(self):
        assert self.params, "SobolSearch params cannot be empty"
        assert self.num_samples > 0, "num_samples must be positive"
        # Eager validation check of each param:
        [_scale_to_spec(value, 0.0) for value in self.params.values()]

    def configs(self, seed: PRNGKeyArray) -> list[dict[str, Any]]:
        """`num_samples` configurations spread evenly over the parameter space."""
        # We accept a JAX key, but need to convert it to a regular key
        seed = jax.random.key_data(seed)  # new random.key() api
        sampler = qmc.Sobol(
            len(self.params), rng=np.random.default_rng(np.asarray(seed))
        )
        random_u_samples = sampler.random(self.num_samples)
        names = list(self.params)
        return [
            {
                name: _scale_to_spec(self.params[name], float(u))
                for name, u in zip(names, row)
            }
            for row in random_u_samples
        ]

    def sweep(
        self,
        fn: Callable,
        *,
        seed: PRNGKeyArray | None = None,
        batch_size: int | None = None,
        print_cost_estimate: bool = False,
    ) -> Sweep:
        """Create a [`Sweep`][jaxnasium.sweep.Sweep] with this search alone."""
        return Sweep(
            fn,
            self,
            seed=seed,
            batch_size=batch_size,
            print_cost_estimate=print_cost_estimate,
        )
