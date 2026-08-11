from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from jaxtyping import PRNGKeyArray

from ._sweep import Sweep


@dataclass(frozen=True)
class OneAtATimeSearch:
    """One-factor-at-a-time (OFAT) search over discrete parameter values.

    Sets the first value of each parameter list as the baseline, then varies
    one parameter at a time to each of its other values while holding the rest at
    baseline. Linear in the number of values.

    Pass as a stage to [`Sweep`][jaxnasium.sweep.Sweep], alone or chained with
    other searches. For only performing this search,
    `OneAtATimeSearch(...).sweep(fn, ...)` is a shorthand for
    `Sweep(fn, self, ...)`.

    **Arguments**:
        `params`: A dict of param names mapping to a non-empty list of values.
            The first entry of each list is the baseline value.

    **Examples**:

    ```python
    sweep = OneAtATimeSearch(
        {
            "gamma": [0.99, 0.95, 0.999],
            "lr": [0.1, 0.01],
        }
    ).sweep(train, batch_size=None)
    for run in sweep:
        for r in run():
            print(r.arguments, r.result)
    ```

    **Note**: when batching is requested and the params mix `vmap`-able and
    non-`vmap`-able values, the baseline runs in a job of its own rather than
    joining the batch.
    """

    params: dict[str, list]

    def __post_init__(self):
        assert self.params, "OneAtATimeSearch params cannot be empty"
        assert all(
            isinstance(values, list) and values for values in self.params.values()
        ), "OneAtATimeSearch params must be non-empty lists"

    def configs(self, seed: PRNGKeyArray) -> list[dict[str, Any]]:
        """
        Returns the baseline config, then one config per non-baseline value of each parameter. `seed` is unused.
        """
        baseline = {name: values[0] for name, values in self.params.items()}
        configs = [dict(baseline)]
        for name, values in self.params.items():
            for value in values[1:]:
                configs.append({**baseline, name: value})
        return configs

    def sweep(
        self,
        fn: Callable,
        *,
        seed: PRNGKeyArray | None = None,  # unused for one-at-a-time search
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
