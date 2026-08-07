from dataclasses import replace
from typing import Any

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
    agent_structure: PyTreeDef | None = None,  # pyright: ignore[reportInvalidTypeForm]
    **kwargs,
) -> Any:
    # from jaxnasium.algorithms import RLAlgorithm
    # check if any element in tree or rest is a "RLAlgorithm"

    # We special case these elements:
    # PRNGKey
    # Transition (Also as output)
    # MultiAgentWrapper (Also as output)

    arguments = [tree] + list(rest)

    # infer agent structure if not provided
    if agent_structure is None:
        first_arg = next((arg for arg in arguments if not _is_prng_key(arg)), None)
        if isinstance(first_arg, MultiAgentWrapper):
            # a wrapper's own first level is its fields (`agents`, `trainer`), not agents
            first_arg = first_arg.agents
        _, agent_structure = eqx.tree_flatten_one_level(first_arg)

    def _process_item(el):
        if _is_prng_key(el):
            return jym.tree.split_key_like_structure(el, agent_structure)
        elif isinstance(el, Transition):
            return el.view_transposed
        elif isinstance(el, MultiAgentWrapper):
            return el.agents
        else:
            return to_per_agent(el, agent_structure)

    # Process args:
    processed_arguments = [_process_item(item) for item in arguments]

    # Process kwargs:
    if kwargs:
        processed_kwargs = {k: _process_item(v) for k, v in kwargs.items()}
        kw_keys = list(processed_kwargs.keys())
        processed_kwargs = jym.tree.map_one_level(
            lambda *vals: dict(zip(kw_keys, vals)),
            *[processed_kwargs[k] for k in kw_keys],
        )

        def _new_call(*args_w_kw_dict):
            *args, kw_dict = args_w_kw_dict
            return f(*args, **kw_dict)

        result = jym.tree.map_one_level(
            _new_call, *processed_arguments, processed_kwargs
        )

    else:
        result = jym.tree.map_one_level(f, *processed_arguments)

    # Now the ouput can contain:
    # a PyTree of Agents originally a MultiAgentWrapper → re-wrap in MultiAgentWrapper
    # a PyTree of Transitions originally a Transition → merge back into a single Transition
    # a PyTree of tuples that we want to convert to a tuple of PyTrees (e.g. dist.sample_and_log_prob returns a tuple, which we want to preserve)
    # out = _maybe_untranspose_transitions(out)
    # out = _result_tuple_to_tuple_result(out)
    # out = MultiAgentWrapper(out) if eqx.tree_is_array(out) else out

    out = _result_tuple_to_tuple_result(result)

    source_wrapper = next(
        (x for x in arguments if isinstance(x, MultiAgentWrapper)), None
    )

    def _process_output(o):
        if source_wrapper is not None and _is_pytree_of_agents(o):
            # carry the team trainer through, so the re-wrapped agents keep it
            return MultiAgentWrapper(o, trainer=source_wrapper.trainer)

        if _is_pytree_of_transitions(o):
            return Transition.from_transposed(o)

        if isinstance(o, tuple):
            return tuple(_process_output(x) for x in o)

        return o

    return _process_output(out)


class MultiAgentWrapper(eqx.Module):
    agents: PyTree
    trainer: Any = None
    """The team's trainer: the `RLAlgorithm` the agents were built from, unsplit.
     `None` for a wrapper over plain modules rather than agents (e.g. `agent.critic` in multi-agent mode), which has no trainer.
    """

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

    def with_hyperparams(self, **hyperparams) -> "MultiAgentWrapper":
        """Override the hyperparameters of every agent's trainer *and* the wrapper trainer."""
        agents = map_multi_agent(
            lambda agent, **kw: agent.with_hyperparams(**kw),
            self.agents,
            **hyperparams,
            agent_structure=self._structure,
        )
        return MultiAgentWrapper(agents, trainer=replace(self.trainer, **hyperparams))

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
            # Methods on an agent are per-agent by default;
            # methods decorated with `@collective` are opted-out, and are called on the wrapper itself.
            if getattr(first_attr.__func__, "__per_agent__", True) is False:
                return first_attr.__func__.__get__(self)

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
