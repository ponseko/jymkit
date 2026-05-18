from functools import partial
from typing import Any, Optional

import equinox as eqx
import jax
from jaxtyping import PyTree, PyTreeDef

import jaxnasium as jym

from .._algorithm import RLAgent
from ._transition import Transition


def _is_prng_key(arg):
    """Check if the argument is a JAX PRNGKeyArray."""
    try:
        jax.random.split(arg)
        return True
    except Exception:
        return False


def _result_tuple_to_tuple_result(r, outer_def=None):
    if outer_def is None:
        _, outer_def = eqx.tree_flatten_one_level(r)
    children = outer_def.flatten_up_to(r)
    if type(children[0]) is not tuple:
        return r  # f returned a single value -> r already has outer structure

    transposed = zip(*children)  # tuple-of-(list of children)
    return tuple(
        _result_tuple_to_tuple_result(
            jax.tree.unflatten(outer_def, list(group)), outer_def
        )
        for group in transposed
    )


def _is_pytree_of_agents(x):
    """Check if *x* is a pytree whose first-level leaves are all RLAgents."""
    try:
        leaves, _ = eqx.tree_flatten_one_level(x)
        return all(isinstance(leaf, RLAgent) for leaf in leaves)
    except Exception:
        return False


def _is_pytree_of_transitions(x):
    """Check if *x* is a pytree whose first-level leaves are all Transitions."""
    try:
        leaves, _ = eqx.tree_flatten_one_level(x)
        return all(isinstance(leaf, Transition) for leaf in leaves)
    except Exception:
        return False


def to_per_agent(x, ref_structure):
    _, structure = eqx.tree_flatten_one_level(x)
    if structure == ref_structure:
        return x
    return jax.tree.unflatten(ref_structure, [x] * ref_structure.num_leaves)


def map_multi_agent(
    f,
    tree,
    *rest,
    agent_structure: Optional[PyTreeDef] = None,  # pyright: ignore[reportInvalidTypeForm]
    **kwargs,
) -> Any:
    # from jaxnasium.algorithms import RLAlgorithm
    # check if any element in tree or rest is a "RLAlgorithm"

    # We special case these elements:
    # PRNGKey
    # Transition (Also as output)
    # MultiAgentWrapper (Also as output)

    processed_arguments = []
    arguments = [tree] + list(rest)

    # infer agent structure if not provided
    if agent_structure is None:
        first_arg = next((arg for arg in arguments if not _is_prng_key(arg)), None)
        _, agent_structure = eqx.tree_flatten_one_level(first_arg)

    for item in arguments:
        if _is_prng_key(item):
            processed_arguments.append(
                jym.tree.split_key_like_structure(item, agent_structure)
            )
        elif isinstance(item, Transition):
            processed_arguments.append(item.view_transposed)
        elif isinstance(item, MultiAgentWrapper):
            processed_arguments.append(item.agents)
        else:
            processed_arguments.append(to_per_agent(item, agent_structure))

    # Any kwargs must be duplicated for each agent:

    if kwargs:
        f = partial(f, **kwargs)

    result = jym.tree.map_one_level(f, *processed_arguments)

    # Now the ouput can contain:
    # a PyTree of Agents originally a MultiAgentWrapper → re-wrap in MultiAgentWrapper
    # a PyTree of Transitions originally a Transition → merge back into a single Transition
    # a PyTree of tuples that we want to convert to a tuple of PyTrees (e.g. dist.sample_and_log_prob returns a tuple, which we want to preserve)
    # out = _maybe_untranspose_transitions(out)
    # out = _result_tuple_to_tuple_result(out)
    # out = MultiAgentWrapper(out) if eqx.tree_is_array(out) else out

    out = _result_tuple_to_tuple_result(result)

    def _process_output(o):
        if any(isinstance(x, MultiAgentWrapper) for x in arguments):
            if _is_pytree_of_agents(o):
                return MultiAgentWrapper(o)

        if _is_pytree_of_transitions(o):
            return Transition.from_transposed(o)

        if isinstance(o, tuple):
            return tuple(_process_output(x) for x in o)

        return o

    return _process_output(out)


class MultiAgentWrapper(eqx.Module):
    agents: PyTree

    @property
    def _structure(self) -> PyTreeDef:  # pyright: ignore[reportInvalidTypeForm]
        """First-level pytree structure of the wrapped agents."""
        return eqx.tree_flatten_one_level(self.agents)[1]

    def _matches_structure(self, arg) -> bool:
        """Check whether *arg*'s first-level structure matches the agent structure."""
        try:
            _, s = eqx.tree_flatten_one_level(arg)
            return s == self._structure
        except Exception:
            return False

    def __call__(self, *args, **kwargs):
        return self.__getattr__("__call__")(*args, **kwargs)

    def __getattr__(self, name: str):
        agents = self.agents
        ref_structure = self._structure
        first_agent = eqx.tree_flatten_one_level(agents)[0][0]
        first_attr = getattr(first_agent, name)

        def _is_bound_method(x):
            return hasattr(x, "__self__") and hasattr(x, "__func__")

        if _is_bound_method(first_attr):

            def multi_agent_dispatcher(*args, **kwargs):
                return map_multi_agent(
                    lambda a, *args, **kw: getattr(a, name)(*args, **kw),
                    self,
                    *args,
                    **kwargs,
                    agent_structure=ref_structure,
                )

            return multi_agent_dispatcher

        elif isinstance(first_attr, eqx.Module):
            return MultiAgentWrapper(
                map_multi_agent(
                    lambda a: getattr(a, name), agents, agent_structure=ref_structure
                )
            )

        return map_multi_agent(
            lambda a: getattr(a, name), agents, agent_structure=ref_structure
        )
