from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray, PyTree, PyTreeDef


class Transition(eqx.Module):
    observation: Array
    action: Array
    reward: Float[Array, " "]
    terminated: Bool[Array, " "]
    log_prob: Optional[Float[Array, "..."]] = None
    info: Optional[dict] = None
    value: Optional[Float[Array, " "]] = None
    next_value: Optional[Float[Array, " "]] = None
    next_observation: Optional[Array] = None
    return_: Optional[Float[Array, " "]] = None
    advantage_: Optional[Float[Array, "..."]] = None

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
            if attr is None:
                fields[f] = None
                continue
            fields[f] = jax.tree.leaves(attr, is_leaf=lambda x: x is not attr)

        per_agent_transitions = []
        for i in range(len(fields[field_names[0]])):
            agent_transition = Transition(
                **{
                    field_name: fields[field_name][i]
                    for field_name in field_names
                    if field_name != "info"
                    and (fields[field_name] is not None)
                    and field_name != "terminated"
                },
                terminated=fields["terminated"][0],
                info=fields["info"],
            )
            per_agent_transitions.append(agent_transition)

        return jax.tree.unflatten(self.structure, per_agent_transitions)

    def make_minibatches(
        self,
        key: PRNGKeyArray,
        n_minibatches: int,
        n_epochs: int = 1,
        n_batch_axis: int = 1,
    ) -> "Transition":
        """
        Creates shuffled minibatches from the transition.
        Returns a copy of the transition with each leaf reshaped to (num_minibatches, ...),

        This function first flattens the transition over the leading n_batch_axis.
        This is useful if your data hasn't been flattened yet and may be structured as
        (rollout_length, num_envs, ...), where num_envs is the number of parallel environments.

        If n_epochs > 1, it will create n_epochs copies of the minibatches. and stack these
        such that there is a single leading axis to scan over for training.

        **Arguments:**
        - `key`: JAX PRNG key for randomization.
        - `num_minibatches`: Number of minibatches to create.
        - `n_epochs`: Number of copies the minibatches should be stacked.
        - `n_batch_axis`: Number of leading batch axes to flatten over. Default is 1 (already flattened).
        """

        def create_minibatch(rng, _):
            rng, key = jax.random.split(rng)

            # Random permutation of the batch indices
            batch_idx = jax.random.permutation(key, batch_size)

            # take from the batch in a new order (the order of the randomized batch_idx)
            shuffled_batch = jax.tree.map(
                lambda x: jnp.take(x, batch_idx, axis=0), batch
            )

            # split in minibatches
            minibatches = jax.tree.map(
                lambda x: x.reshape((n_minibatches, -1) + x.shape[1:]), shuffled_batch
            )
            return rng, minibatches

        # reshape (flatten over all batch axes)
        batch = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[n_batch_axis:]), self)
        batch_size = jax.tree.leaves(batch)[0].shape[0]

        # Create n_epochs of minibatches
        rng, minibatches = jax.lax.scan(create_minibatch, key, None, n_epochs)

        # (n_epochs, n_minibatches, ...) --> (n_epochs * n_minibatches, ...)
        minibatches = jax.tree.map(
            lambda x: x.reshape((-1,) + x.shape[2:]), minibatches
        )

        return minibatches
