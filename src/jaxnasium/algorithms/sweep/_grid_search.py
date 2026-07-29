import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from jaxtyping import PRNGKeyArray

from ._sweep import Sweep


@dataclass(frozen=True)
class GridSearch:
    """Sets up a grid search over the given parameter values. All
    combinations of the parameter values are created in its configs.

    Pass as a stage to [`Sweep`][jaxnasium.algorithms.sweep.Sweep], alone or
    chained with other searches. For only performing this grid search,
    `GridSearch(...).sweep(fn, ...)` is a shorthand for `Sweep(fn, self, ...)`.
    A `Sweep` object allows batching jobs together in various `vmap` calls. See
    [`Sweep`][jaxnasium.algorithms.sweep.Sweep] for more details.

    **Arguments**:
        `params`: A dict of param names mapping to a non-empty list of values.

    **Examples**:

    ```python
    sweep = GridSearch(
        {
            "env_name": ["CartPole-v1", "Acrobot-v1"],
            "gamma": [0.95, 0.99, 0.999],
        }
    ).sweep(train, batch_size=None)
    for run in sweep:
        for r in run():
            print(r.arguments, r.result)
    ```

    Combined with a random search over learning rates:

    ```python
    sweep = Sweep(
        train,
        GridSearch({"env_name": ["CartPole-v1", "Acrobot-v1"]}),
        RandomSearch({"learning_rate": (1e-4, 1e-2, "log")}, num_samples=32),
        seed=jax.random.PRNGKey(0),
        batch_size=8,
    )
    for run in sweep:
        for r in run():
            print(r.arguments, r.result)
    ```
    """

    params: dict[str, list]

    def __post_init__(self):
        assert self.params, "GridSearch params cannot be empty"
        assert all(
            isinstance(values, list) and values for values in self.params.values()
        ), "GridSearch params must be non-empty lists"

    def configs(self, seed: PRNGKeyArray) -> list[dict[str, Any]]:
        """
        Returns every combination of the parameter values.
        `seed` is unused.
        """
        names = list(self.params)
        return [
            dict(zip(names, combination))
            for combination in itertools.product(*self.params.values())
        ]

    def sweep(
        self,
        fn: Callable,
        *,
        seed: PRNGKeyArray | None = None,  # unused for grid search
        batch_size: int | None = None,
    ) -> Sweep:
        """Create a [`Sweep`][jaxnasium.algorithms.sweep.Sweep] object with this grid search"""
        return Sweep(fn, self, seed=seed, batch_size=batch_size)
