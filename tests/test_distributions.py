import distrax
import jax
import jax.numpy as jnp

from jaxnasium.algorithms.core import TanhNormalFactory
from jaxnasium.algorithms.core._distributions import (
    DistraxContainer,
    TanhNormal,
    _transpose_tree_of_tuples,
)

SEED = jax.random.PRNGKey(0)


def _make_normal(loc=0.0, scale=1.0):
    return distrax.Normal(loc=jnp.array(loc), scale=jnp.array(scale))


# ---------------------------------------------------------------------------
# _transpose_tree_of_tuples unit tests
# ---------------------------------------------------------------------------


class TestTransposeTreeOfTuples:
    def test_flat_dict(self):
        """Dict of tuples -> tuple of dicts."""
        tree = {"a": (1, 10), "b": (2, 20)}
        outer_td = jax.tree.structure({"a": 0, "b": 0})
        result = _transpose_tree_of_tuples(tree, outer_td)
        assert isinstance(result, tuple) and len(result) == 2
        assert result[0] == {"a": 1, "b": 2}
        assert result[1] == {"a": 10, "b": 20}

    def test_nested_dict(self):
        """Nested dict of tuples -> tuple of nested dicts."""
        tree = {"a": {"x": (1, 10), "y": (2, 20)}, "b": (3, 30)}
        outer_td = jax.tree.structure({"a": {"x": 0, "y": 0}, "b": 0})
        result = _transpose_tree_of_tuples(tree, outer_td)
        assert isinstance(result, tuple) and len(result) == 2
        assert result[0] == {"a": {"x": 1, "y": 2}, "b": 3}
        assert result[1] == {"a": {"x": 10, "y": 20}, "b": 30}

    def test_tuple_structure(self):
        """Tuple-of-tuples where outer is structural, inner is result."""
        tree = ((1, 10), (2, 20))
        outer_td = jax.tree.structure((0, 0))
        result = _transpose_tree_of_tuples(tree, outer_td)
        assert isinstance(result, tuple) and len(result) == 2
        assert result[0] == (1, 2)
        assert result[1] == (10, 20)

    def test_list_structure(self):
        """List of tuples -> tuple of lists."""
        tree = [(1, 10), (2, 20)]
        outer_td = jax.tree.structure([0, 0])
        result = _transpose_tree_of_tuples(tree, outer_td)
        assert isinstance(result, tuple) and len(result) == 2
        assert result[0] == [1, 2]
        assert result[1] == [10, 20]

    def test_passthrough_when_no_tuples(self):
        """Non-tuple leaves are returned unchanged."""
        tree = {"a": 1, "b": 2}
        outer_td = jax.tree.structure({"a": 0, "b": 0})
        result = _transpose_tree_of_tuples(tree, outer_td)
        assert result == tree


# ---------------------------------------------------------------------------
# DistraxContainer tests
# ---------------------------------------------------------------------------


class TestDistraxContainerSingle:
    """Single (non-nested) distribution."""

    def test_sample(self):
        c = DistraxContainer(distribution=_make_normal())
        s = c.sample(seed=SEED)
        assert s.shape == ()

    def test_sample_and_log_prob(self):
        c = DistraxContainer(distribution=_make_normal())
        s, lp = c.sample_and_log_prob(seed=SEED)
        assert s.shape == ()
        assert lp.shape == ()

    def test_log_prob(self):
        c = DistraxContainer(distribution=_make_normal())
        lp = c.log_prob(jnp.array(0.5))
        assert lp.shape == ()


class TestDistraxContainerDict:
    """Dict of distributions (flat)."""

    def setup_method(self):
        self.container = DistraxContainer(
            distribution={"a": _make_normal(0.0), "b": _make_normal(1.0)}
        )

    def test_sample(self):
        s = self.container.sample(seed=SEED)
        assert isinstance(s, dict) and set(s.keys()) == {"a", "b"}

    def test_sample_and_log_prob(self):
        s, lp = self.container.sample_and_log_prob(seed=SEED)
        assert isinstance(s, dict) and set(s.keys()) == {"a", "b"}
        assert isinstance(lp, dict) and set(lp.keys()) == {"a", "b"}

    def test_log_prob(self):
        lp = self.container.log_prob({"a": jnp.array(0.0), "b": jnp.array(1.0)})
        assert isinstance(lp, dict) and set(lp.keys()) == {"a", "b"}

    def test_getattr_sample_and_log_prob(self):
        """sample_and_log_prob called via __getattr__ returns (dict, dict)."""
        s, lp = self.container.sample_and_log_prob(seed=SEED)
        assert isinstance(s, dict)
        assert isinstance(lp, dict)


class TestDistraxContainerNestedDict:
    """Nested dict of distributions — the case the old code failed on."""

    def setup_method(self):
        self.container = DistraxContainer(
            distribution={
                "group1": {"x": _make_normal(0.0), "y": _make_normal(1.0)},
                "group2": _make_normal(2.0),
            }
        )

    def test_sample(self):
        s = self.container.sample(seed=SEED)
        assert isinstance(s["group1"], dict)
        assert set(s["group1"].keys()) == {"x", "y"}
        assert s["group2"].shape == ()

    def test_sample_and_log_prob(self):
        s, lp = self.container.sample_and_log_prob(seed=SEED)
        # samples
        assert isinstance(s, dict)
        assert isinstance(s["group1"], dict)
        assert set(s["group1"].keys()) == {"x", "y"}
        assert s["group2"].shape == ()
        # log_probs
        assert isinstance(lp, dict)
        assert isinstance(lp["group1"], dict)
        assert set(lp["group1"].keys()) == {"x", "y"}
        assert lp["group2"].shape == ()

    def test_log_prob(self):
        value = {
            "group1": {"x": jnp.array(0.0), "y": jnp.array(1.0)},
            "group2": jnp.array(2.0),
        }
        lp = self.container.log_prob(value)
        assert isinstance(lp["group1"], dict)  # type: ignore


class TestDistraxContainerTupleStructure:
    """Tuple of distributions — the tricky case with tuple-as-structure."""

    def setup_method(self):
        self.container = DistraxContainer(
            distribution=(_make_normal(0.0), _make_normal(1.0))
        )

    def test_sample(self):
        s = self.container.sample(seed=SEED)
        assert isinstance(s, tuple) and len(s) == 2

    def test_sample_and_log_prob(self):
        s, lp = self.container.sample_and_log_prob(seed=SEED)
        assert isinstance(s, tuple) and len(s) == 2
        assert isinstance(lp, tuple) and len(lp) == 2
        for val in (*s, *lp):
            assert val.shape == ()

    def test_log_prob(self):
        lp = self.container.log_prob((jnp.array(0.0), jnp.array(1.0)))
        assert isinstance(lp, tuple) and len(lp) == 2


def test_samples_within_default_bounds():
    dist = TanhNormal(mean=jnp.zeros((4,)), std=jnp.ones((4,)))
    samples = dist.sample(seed=SEED, sample_shape=(1000,))
    assert samples.shape == (1000, 4)
    # Default shift=0, scale=1 -> support is (-1, 1).
    assert jnp.all(samples > -1.0)
    assert jnp.all(samples < 1.0)


def test_mode_matches_shifted_scaled_tanh():
    mean = jnp.array([0.5, -1.0, 0.0])
    dist = TanhNormal(mean=mean, std=jnp.ones((3,)), shift=1.0, scale=2.0)
    assert jnp.allclose(dist.mode(), 1.0 + 2.0 * jnp.tanh(mean))


def test_log_prob_is_finite_even_at_boundary():
    dist = TanhNormal(mean=jnp.zeros((3,)), std=jnp.ones((3,)))
    # Values exactly at the tanh boundary would blow up without clipping.
    lp = dist.log_prob(jnp.array([-1.0, 0.0, 1.0]))
    assert lp.shape == (3,)
    assert jnp.all(jnp.isfinite(lp))


def test_sample_and_log_prob_consistent_with_log_prob():
    dist = TanhNormal(mean=jnp.zeros((5,)), std=jnp.ones((5,)))
    sample, log_prob = dist.sample_and_log_prob(seed=SEED)
    assert jnp.allclose(log_prob, dist.log_prob(sample), atol=1e-4)


def test_batch_and_event_shape():
    dist = TanhNormal(mean=jnp.zeros((4,)), std=jnp.ones((4,)))
    assert dist.batch_shape == (4,)
    assert dist.event_shape == ()


def test_factory_sets_shift_and_scale_from_bounds():
    low, high = -3.0, 5.0
    factory = TanhNormalFactory(low, high)
    dist = factory(mean=jnp.zeros((2,)), std=jnp.ones((2,)))
    assert isinstance(dist, TanhNormal)
    # scale = (high - low) / 2, shift = (high + low) / 2
    assert jnp.allclose(dist._scale, 4.0)
    assert jnp.allclose(dist._shift, 1.0)


def test_factory_samples_within_bounds():
    low, high = -3.0, 5.0
    factory = TanhNormalFactory(low, high)
    dist = factory(mean=jnp.zeros((3,)), std=jnp.ones((3,)))
    samples = dist.sample(seed=SEED, sample_shape=(1000,))
    assert jnp.all(samples > low)
    assert jnp.all(samples < high)
