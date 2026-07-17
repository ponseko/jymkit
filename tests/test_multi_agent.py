"""Direct unit tests for the multi-agent primitives (`map_multi_agent`,
`MultiAgentWrapper`).

The end-to-end multi-agent paths (init_agent / get_action on multi-agent
proxy envs) are already exercised in ``test_multi_agent_input_output.py``.
These tests pin down the lower-level building blocks in isolation.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxnasium.algorithms.core import Transition
from jaxnasium.algorithms.core._multi_agent import MultiAgentWrapper, map_multi_agent

SEED = jax.random.PRNGKey(0)


def _multi_agent_transition(n: int = 5) -> Transition:
    return Transition(
        observation={"a0": jnp.ones((n, 3)), "a1": jnp.zeros((n, 3))},  # type: ignore
        action={"a0": jnp.ones((n, 1)), "a1": jnp.zeros((n, 1))},  # type: ignore
        reward={"a0": jnp.ones((n, 1)), "a1": jnp.full((n, 1), 2.0)},  # type: ignore
        terminated=jnp.zeros((n,), dtype=bool),
        truncated=jnp.zeros((n,), dtype=bool),
    )


def test_map_multi_agent_broadcasts_shared_arg():
    states = {"a0": jnp.array([1.0, 2.0]), "a1": jnp.array([3.0, 4.0])}
    obs = {"a0": jnp.array([0.1, 0.2]), "a1": jnp.array([0.3, 0.4])}

    result = map_multi_agent(lambda s, o, shared: s + o + shared, states, obs, 10.0)

    assert set(result.keys()) == {"a0", "a1"}
    assert jnp.allclose(result["a0"], jnp.array([11.1, 12.2]))
    assert jnp.allclose(result["a1"], jnp.array([13.3, 14.4]))


def test_map_multi_agent_handles_heterogeneous_agent_shapes():
    """Agents differ in what they carry"""
    states = [jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0, 5.0])]
    obs = [jnp.array([0.1, 0.2]), jnp.array([0.3, 0.4, 0.5])]

    result = map_multi_agent(lambda s, o, shared: s + o + shared, states, obs, 10.0)

    assert jnp.allclose(result[0], jnp.array([11.1, 12.2]))
    assert jnp.allclose(result[1], jnp.array([13.3, 14.4, 15.5]))


def test_map_multi_agent_splits_prng_keys_per_agent():
    states = {"a0": 0, "a1": 0}
    result = map_multi_agent(
        lambda key, s: s + jax.random.randint(key, (), 0, 1000), SEED, states
    )

    assert set(result.keys()) == {"a0", "a1"}
    assert result["a0"] != result["a1"]


def test_map_multi_agent_transposes_transition_input():
    transition = _multi_agent_transition()

    result = map_multi_agent(lambda t: t.reward + t.action, transition)

    assert set(result.keys()) == {"a0", "a1"}
    assert jnp.allclose(result["a0"], 1.0 + 1.0)
    assert jnp.allclose(result["a1"], 2.0 + 0.0)


def test_map_multi_agent_merges_transition_output():
    transition = _multi_agent_transition()

    result = map_multi_agent(lambda t: t.replace(reward=t.reward * 2.0), transition)

    assert isinstance(result, Transition)
    assert jnp.allclose(result.reward["a0"], 2.0)
    assert jnp.allclose(result.reward["a1"], 4.0)


def test_map_multi_agent_tuple_return_becomes_tuple_of_trees():
    states = {"a0": jnp.array(1.0), "a1": jnp.array(2.0)}

    result = map_multi_agent(lambda s: (s + 1, s - 1), states)

    assert isinstance(result, tuple) and len(result) == 2
    plus, minus = result
    assert plus == {"a0": 2.0, "a1": 3.0}
    assert minus == {"a0": 0.0, "a1": 1.0}


class _TestAgent(eqx.Module):
    w: jax.Array

    def act(self, x):
        return x * self.w


def _make_wrapper() -> MultiAgentWrapper:
    return MultiAgentWrapper(
        {"a0": _TestAgent(jnp.array(2.0)), "a1": _TestAgent(jnp.array(3.0))}
    )


def test_wrapper_dispatches_method_per_agent():
    wrapper = _make_wrapper()
    out = wrapper.act({"a0": jnp.array(1.0), "a1": jnp.array(1.0)})
    assert out["a0"] == 2.0
    assert out["a1"] == 3.0


def test_wrapper_gathers_leaf_attribute_per_agent():
    wrapper = _make_wrapper()
    out = wrapper.w
    assert out["a0"] == 2.0  # type: ignore
    assert out["a1"] == 3.0  # type: ignore


def test_wrapper_structure_reflects_agents():
    wrapper = _make_wrapper()
    assert wrapper._structure == jax.tree.structure({"a0": 0, "a1": 0})
    assert wrapper._matches_structure({"a0": 99.0, "a1": 999.0})
    assert not wrapper._matches_structure({"a0": 1.0})
