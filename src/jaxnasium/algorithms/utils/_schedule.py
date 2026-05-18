import equinox as eqx
import jax
import optax
from jaxtyping import PyTree


class Schedule(eqx.Module):
    """A simple wrapper around optax schedules to allow for per-agent schedules in multi-agent settings.
    Is either a constant value or a linear schedule from start to end over the course of training."""

    start: float | PyTree
    end: float | PyTree | None
    transition_steps: int

    def __call__(self, count):
        if self.end is None:
            return self.start

        # If start is not per-agent, use end as the first argument of the tree.map:
        if jax.tree.structure(self.start) == jax.tree.structure(0):
            return jax.tree.map(
                lambda end, start: optax.linear_schedule(
                    init_value=start,
                    end_value=end,
                    transition_steps=self.transition_steps,
                )(count),
                self.end,
                self.start,
            )

        return jax.tree.map(
            lambda start, end: optax.linear_schedule(
                init_value=start,
                end_value=end,
                transition_steps=self.transition_steps,
            )(count),
            self.start,
            self.end,
        )
