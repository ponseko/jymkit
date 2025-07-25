import distrax
import equinox as eqx
import jax
from jaxtyping import PyTree


class DistraxContainer(eqx.Module):
    """Container for (possibly nested as PyTrees) distrax distributions."""

    distribution: distrax.Distribution | PyTree[distrax.Distribution]

    def __getattr__(self, name):
        return jax.tree.map(
            lambda x: getattr(x, name),
            self.distribution,
            is_leaf=lambda x: isinstance(x, distrax.Distribution),
        )
