from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PyTree, PyTreeDef


class Transition(eqx.Module):
    observation: Array
    action: Array
    reward: Float[Array, "..."]
    terminated: Bool[Array, "..."]
    log_prob: Array
    info: dict
    value: Array
    next_value: Array
    return_: Optional[Array] = None
    advantage_: Optional[Array] = None

    @property
    def structure(self) -> PyTreeDef:
        """
        Returns the top-level structure of the transition objects (using reward as a reference).
        This is either PyTreeDef(*) for single agents
        or PyTreeDef((*, x num_agents)) for multi-agent environments.
        usefull for unflattening Transition.flat.properties back to the original structure.
        """
        return jax.tree.structure(self.reward)

    @property
    def view_flat(self) -> "Transition":
        """
        Returns a flattened version of the transition.
        Where possible, this is a jnp.stack of the leaves.
        Otherwise, it returns a list of leaves.
        """

        def return_as_stack_or_list(x):
            x = jax.tree.leaves(x)
            try:
                return jnp.stack(x, axis=-1).squeeze()
            except ValueError:
                return x

        return jax.tree.map(
            return_as_stack_or_list,
            self,
            is_leaf=lambda y: y is not self,
        )

    @property
    def view_transposed(self) -> PyTree["Transition"]:
        """
        The original transition is a Transition of PyTrees
            e.g. Transition(observation={a1: ..., a2: ...}, action={a1: ..., a2: ...}, ...)
        The transposed transition is a PyTree of Transitions
            e.g. {a1: Transition(observation=..., action=..., ...), a2: Transition(observation=..., action=..., ...), ...}
        This is useful for multi-agent environments where we want to have a single Transition object per agent.
        In single-agent environments, this will be the same as the original transition.
        """
        if self.structure.num_leaves == 1:  # single agent
            return self

        field_names = list(self.__dataclass_fields__.keys())

        fields = {}
        for f in field_names:
            attr = getattr(self, f)
            fields[f] = jax.tree.leaves(attr, is_leaf=lambda x: x is not attr)

        per_agent_transitions = []
        for i in range(len(fields[field_names[0]])):
            agent_transition = Transition(
                **{
                    field_name: fields[field_name][i]
                    for field_name in field_names
                    if field_name != "info"
                    and (fields[field_name] is not None)
                    # and field_name != "advantage_"
                    and field_name != "terminated"
                },
                terminated=fields["terminated"][0],
                info=fields["info"],
            )
            per_agent_transitions.append(agent_transition)

        return jax.tree.unflatten(self.structure, per_agent_transitions)
