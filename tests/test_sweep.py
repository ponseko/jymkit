import json

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import jaxnasium as jym
from jaxnasium.sweep import (
    GridSearch,
    OneAtATimeSearch,
    RandomSearch,
    SobolSearch,
    Sweep,
)
from jaxnasium.sweep._probe import split_static_dynamic_params


def test_split_static_dynamic_params():
    def fn(x, y, mul: bool, z: str):
        if mul:
            return x * y
        return x + y

    static_params, dynamic_params = split_static_dynamic_params(
        fn, {"x": (0.1, 1.0), "y": [1, 2], "mul": [True, False], "z": ["yes"]}
    )
    assert static_params == {"mul": [True, False], "z": ["yes"]}
    assert dynamic_params == {"x": (0.1, 1.0), "y": [1, 2]}


def test_split_static_dynamic_params_with_eqx_module():
    class Model(eqx.Module):
        y: int
        mul: bool

        def __call__(self, x):
            if self.mul:
                return x * self.y
            return x + self.y

    def fn(x, y, mul):
        model = Model(y, mul)
        return model(x)

    static_params, dynamic_params = split_static_dynamic_params(
        fn, {"x": (0.1, 1.0), "y": [1, 2], "mul": [True, False]}
    )
    assert static_params == {"mul": [True, False]}
    assert dynamic_params == {"x": (0.1, 1.0), "y": [1, 2]}


def test_split_static_dynamic_params_with_eqx_module_static_field():
    class Model(eqx.Module):
        y: int
        mul: bool = eqx.field(static=True)

        def __call__(self, x):
            if self.mul:
                return x * self.y
            return x + self.y

    def fn(x, y, mul):
        model = Model(y, mul)
        return model(x)

    static_params, dynamic_params = split_static_dynamic_params(
        fn, {"x": (0.1, 1.0), "y": [1, 2], "mul": [True, False]}
    )
    assert static_params == {"mul": [True, False]}
    assert dynamic_params == {"x": (0.1, 1.0), "y": [1, 2]}


def test_params_differ_static_under_different_values():
    """A param dynamic some values but static under another -> static."""

    def fn(alg, scale):
        if alg == "sac":
            return scale * 2.0  # vmap safe
        # scale defines the shape, so not vmap safe -> static
        return jnp.ones(int(scale) + 1).sum()

    static_params, dynamic_params = split_static_dynamic_params(
        fn, {"alg": ["sac", "ppo"], "scale": (0.5, 2.0)}
    )
    assert static_params == {"alg": ["sac", "ppo"], "scale": (0.5, 2.0)}
    assert dynamic_params == {}


def test_params_are_dynamic_when_safe_under_all():
    def fn(alg, scale):
        if alg == "sac":
            return scale * 2.0
        return scale + 1.0

    static_params, dynamic_params = split_static_dynamic_params(
        fn, {"alg": ["sac", "ppo"], "scale": (0.5, 2.0)}
    )
    assert static_params == {"alg": ["sac", "ppo"]}
    assert dynamic_params == {"scale": (0.5, 2.0)}


def test_grid_search_static_params_get_one_job_each():
    def fn(env_name, task):
        return 0.0

    # NOTE: all are static because not-numeric;
    sweep = GridSearch({"env_name": ["a", "b"], "task": ["x", "y", "z"]}).sweep(fn)

    assert len(sweep) == 6
    assert all(job.dynamic_args == {} for job in sweep.jobs)
    assert all(job.num_runs == 1 for job in sweep.jobs)

    results = sweep[0]()
    assert len(results) == 1
    assert results[0].arguments == {"env_name": "a", "task": "x"}
    assert results[0].num_runs == 1
    assert results[0].duration >= 0.0


def test_grid_search_vmaps_dynamic_params_within_static_jobs():
    def fn(lr, env_name):
        return lr * 2

    sweep = GridSearch({"lr": [0.1, 0.2, 0.3], "env_name": ["a", "b"]}).sweep(
        fn, batch_size=0
    )

    # One job per env, each vmapping all three learning rates.
    assert len(sweep) == 2
    assert {job.static_args["env_name"] for job in sweep.jobs} == {"a", "b"}
    assert all(job.dynamic_args["lr"] == [0.1, 0.2, 0.3] for job in sweep.jobs)

    results = sweep[0]()
    assert len(results) == 3
    assert all(r.num_runs == 3 for r in results)
    assert [r.arguments["lr"] for r in results] == [0.1, 0.2, 0.3]
    assert results[0].result == pytest.approx(0.2)


def test_default_batch_size_does_not_vmap():
    def fn(lr, env_name):
        return lr

    sweep = GridSearch({"env_name": ["a", "b"], "lr": [0.1, 0.2, 0.3]}).sweep(fn)

    assert len(sweep) == 6
    assert all(job.num_runs == 1 for job in sweep.jobs)


def test_batch_size_zero_packs_all_matching_static_args():
    def fn(lr, env_name):
        return lr

    sweep = GridSearch({"env_name": ["a", "b"], "lr": [0.1, 0.2, 0.3, 0.4]}).sweep(
        fn, batch_size=0
    )

    assert len(sweep) == 2
    assert all(job.num_runs == 4 for job in sweep.jobs)


def test_batch_size_splits_a_vmapped_job():
    def fn(lr, env_name):
        return lr

    sweep = GridSearch({"env_name": ["a", "b"], "lr": [0.1, 0.2, 0.3, 0.4]}).sweep(
        fn, batch_size=2
    )

    assert len(sweep) == 4
    assert all(job.num_runs == 2 for job in sweep.jobs)


def test_with_cost_size_estimate(capsys):
    def fn(lr, env_name):
        return lr

    _sweep = GridSearch({"env_name": ["a", "b"], "lr": [0.1, 0.2, 0.3, 0.4]}).sweep(
        fn, batch_size=2, print_cost_estimate=True
    )

    out = capsys.readouterr().out
    assert "largest job runs 2 configuration(s) in parallel" in out
    assert "CostEstimate('fn':" in out


def test_log_cost_estimate_returns_dataclass():
    from jaxnasium.sweep._probe import CostEstimate, log_cost_estimate

    def fn(lr):
        return (jnp.ones(8) * lr).sum()

    estimate = log_cost_estimate(fn, lr=0.1)
    assert isinstance(estimate, CostEstimate)
    assert estimate.error is None
    assert estimate.device_memory_bytes is not None


def test_batch_size_allows_a_smaller_final_batch():
    def fn(lr):
        return lr

    sweep = GridSearch({"lr": [0.1, 0.2, 0.3]}).sweep(fn, batch_size=2)

    assert [job.num_runs for job in sweep.jobs] == [2, 1]


def test_sweep_len_iter_and_slice():
    def fn(lr):
        return lr

    sweep = GridSearch({"lr": [0.1, 0.2, 0.3]}).sweep(fn)
    assert len(sweep) == 3

    # Iteration yields one callable per job; calling it runs that job.
    results = [r for run in sweep for r in run()]
    assert [r.result for r in results] == pytest.approx([0.1, 0.2, 0.3])
    assert [r.arguments["lr"] for r in results] == pytest.approx([0.1, 0.2, 0.3])

    # Integer index: same callable shape as iteration items.
    first = sweep[0]
    assert callable(first)
    assert first()[0].result == pytest.approx(0.1)
    assert sweep[-1]()[0].result == pytest.approx(0.3)

    # Slice: a new Sweep over a subset of jobs; still iterable / indexable.
    sliced = sweep[1:]
    assert isinstance(sliced, Sweep)
    assert len(sliced) == 2
    assert [r.result for run in sliced for r in run()] == pytest.approx([0.2, 0.3])
    assert sliced[0]()[0].result == pytest.approx(0.2)

    mid = sweep[1:2]
    assert len(mid) == 1
    assert mid.jobs[0] is sweep.jobs[1]


def test_sweep_iter_with_vmapped_batch():
    """Each iterated run may return several SweepResults when batch_size > 1."""

    def fn(lr):
        return lr * 2

    sweep = GridSearch({"lr": [0.1, 0.2, 0.3, 0.4]}).sweep(fn, batch_size=2)
    assert len(sweep) == 2

    batches = [run() for run in sweep]
    assert [len(batch) for batch in batches] == [2, 2]
    assert [r.result for batch in batches for r in batch] == pytest.approx(
        [0.2, 0.4, 0.6, 0.8]
    )
    assert all(r.num_runs == 2 for batch in batches for r in batch)

    # Indexing a batched job returns all results for that vmap.
    assert [r.arguments["lr"] for r in sweep[0]()] == pytest.approx([0.1, 0.2])


def test_random_search_vmaps_dynamic_params():
    def fn(scale):
        return scale

    sweep = RandomSearch({"scale": (0.5, 2.0)}, num_samples=6).sweep(
        fn, seed=jax.random.PRNGKey(0), batch_size=2
    )

    assert len(sweep) == 3
    assert all(job.static_args == {} for job in sweep.jobs)
    assert all(job.num_runs == 2 for job in sweep.jobs)

    sampled = [scale for job in sweep.jobs for scale in job.dynamic_args["scale"]]
    assert len(set(sampled)) == 6
    assert all(0.5 <= scale <= 2.0 for scale in sampled)

    results = sweep[0]()
    assert [r.result for r in results] == pytest.approx(
        sweep.jobs[0].dynamic_args["scale"]
    )


def test_random_search_static_param_gets_its_own_job():
    def fn(scale, env_name):
        return scale

    sweep = RandomSearch(
        {"scale": (0.5, 2.0), "env_name": ["a", "b"]}, num_samples=5
    ).sweep(fn, seed=jax.random.PRNGKey(0), batch_size=4)

    # Samples only batch together while the sampled static param stays the same.
    assert sum(job.num_runs for job in sweep.jobs) == 5
    assert all(job.static_args["env_name"] in ("a", "b") for job in sweep.jobs)
    assert all(
        len({*job.dynamic_args["scale"]}) == job.num_runs and job.num_runs <= 4
        for job in sweep.jobs
    )


def test_sampling_without_seed_or_fixed_seed_raises():
    def fn(scale):
        return scale

    with pytest.raises(ValueError, match="seed is required"):
        RandomSearch({"scale": (0.5, 2.0)}, num_samples=2).sweep(fn)


def test_linear_and_log_ranges_sample_differently_and_correctly():
    low, high, mid = 1e-4, 1e-2, 1e-3

    linear = RandomSearch({"lr": (low, high)}, num_samples=20_000).configs(
        jax.random.PRNGKey(0)
    )
    log = RandomSearch({"lr": (low, high, "log")}, num_samples=20_000).configs(
        jax.random.PRNGKey(0)
    )

    linear_above = sum(c["lr"] >= mid for c in linear) / len(linear)
    log_above = sum(c["lr"] >= mid for c in log) / len(log)

    assert linear_above == pytest.approx(0.909, abs=0.02)
    assert log_above == pytest.approx(0.5, abs=0.02)
    assert linear_above > log_above + 0.3


def _grid_then_random(random_search: RandomSearch) -> Sweep:
    def fn(env_name, lr):
        return lr

    return Sweep(
        fn,
        GridSearch({"env_name": ["a", "b"]}),
        random_search,
        seed=jax.random.PRNGKey(0),
        batch_size=2,
    )


def test_grid_then_random_draws_fresh_values_by_default():
    sweep = _grid_then_random(RandomSearch({"lr": (0.0, 1.0)}, num_samples=4))

    assert len(sweep) == 4
    assert [job.static_args["env_name"] for job in sweep.jobs] == ["a", "a", "b", "b"]

    results = [result for job in sweep.jobs for result in sweep.fn(job)]
    assert len(results) == 8
    assert sum(r.arguments["env_name"] == "a" for r in results) == 4

    lrs_a = [r.arguments["lr"] for r in results if r.arguments["env_name"] == "a"]
    lrs_b = [r.arguments["lr"] for r in results if r.arguments["env_name"] == "b"]
    assert not set(lrs_a) & set(lrs_b)


def test_grid_then_random_reuses_draws_with_fixed_seed():
    sweep = _grid_then_random(
        RandomSearch(
            {"lr": (0.0, 1.0)},
            num_samples=4,
            fixed_seed=jax.random.PRNGKey(7),
        )
    )

    results = [result for job in sweep.jobs for result in sweep.fn(job)]
    lrs_a = [r.arguments["lr"] for r in results if r.arguments["env_name"] == "a"]
    lrs_b = [r.arguments["lr"] for r in results if r.arguments["env_name"] == "b"]
    assert lrs_a == lrs_b
    assert len(set(lrs_a)) == 4


def test_mixed_fixed_and_unfixed_sampling_searches():
    def fn(env_name, lr, dropout):
        return lr * dropout

    sweep = Sweep(
        fn,
        GridSearch({"env_name": ["a", "b"]}),
        RandomSearch(
            {"lr": (0.0, 1.0)},
            num_samples=2,
            fixed_seed=jax.random.PRNGKey(1),
        ),
        RandomSearch({"dropout": (0.0, 0.5)}, num_samples=2),
        seed=jax.random.PRNGKey(0),
        batch_size=0,
    )

    # 2 envs * 2 lrs * 2 dropouts = 8 configs, all dynamic except env.
    results_by_env = {"a": [], "b": []}
    for job in sweep.jobs:
        env = job.static_args["env_name"]
        for i in range(job.num_runs):
            results_by_env[env].append(
                (job.dynamic_args["lr"][i], job.dynamic_args["dropout"][i])
            )

    lrs_a = [lr for lr, _ in results_by_env["a"]]
    lrs_b = [lr for lr, _ in results_by_env["b"]]
    drop_a = [d for _, d in results_by_env["a"]]
    drop_b = [d for _, d in results_by_env["b"]]

    # Learning rates are pinned and shared; dropouts are redrawn per env.
    assert lrs_a == lrs_b
    assert drop_a != drop_b


def test_random_then_random_fixed():
    def fn(lr, gamma):
        return lr * gamma

    sweep = Sweep(
        fn,
        RandomSearch({"lr": (0.0, 1.0)}, num_samples=3),
        RandomSearch(
            {"gamma": (0.9, 1.0)},
            num_samples=2,
            fixed_seed=jax.random.PRNGKey(5),
        ),
        seed=jax.random.PRNGKey(0),
        batch_size=0,
    )

    # Both params are dynamic, so all 6 configurations share a single job.
    assert len(sweep) == 1
    assert sweep.jobs[0].num_runs == 6

    # Each of the 3 learning rates is paired with both of the 2 shared gammas.
    lrs = sweep.jobs[0].dynamic_args["lr"]
    gammas = sweep.jobs[0].dynamic_args["gamma"]
    assert len(set(lrs)) == 3
    assert lrs[0] == lrs[1] and lrs[2] == lrs[3] and lrs[4] == lrs[5]
    assert len(set(gammas)) == 2
    assert gammas[0:2] == gammas[2:4] == gammas[4:6]


def test_random_then_grid_chains():
    def fn(lr, env_name):
        return lr

    sweep = Sweep(
        fn,
        RandomSearch({"lr": (0.0, 1.0)}, num_samples=3),
        GridSearch({"env_name": ["a", "b"]}),
        seed=jax.random.PRNGKey(0),
        batch_size=0,
    )

    # env_name is static, so each sampled lr splits into one job per env.
    # Because the grid search comes after the random search, we get a grid for
    # every lr, hence 6 jobs instead of 2 with 3 runs.
    assert len(sweep) == 6
    assert [job.static_args["env_name"] for job in sweep.jobs] == list("ababab")
    assert all(job.num_runs == 1 for job in sweep.jobs)


def test_param_swept_by_two_searches_raises():
    def fn(lr):
        return lr

    with pytest.raises(ValueError, match="more than one stage"):
        Sweep(
            fn,
            GridSearch({"lr": [0.1, 0.2]}),
            RandomSearch({"lr": (0.0, 1.0)}, num_samples=2),
            seed=jax.random.PRNGKey(0),
        )


def test_sobol_search():
    def fn(lr, layers):
        return lr

    def sample(seed):
        sweep = SobolSearch(
            {"lr": (0.0, 1.0), "layers": [16, 32, 64, 128]}, num_samples=4
        ).sweep(fn, seed=seed, batch_size=0)
        return sweep.jobs[0].dynamic_args

    assert sample(jax.random.PRNGKey(0)) == sample(jax.random.PRNGKey(0))
    assert sample(jax.random.PRNGKey(0)) != sample(jax.random.PRNGKey(1))
    assert set(sample(jax.random.PRNGKey(0))["layers"]) <= {16, 32, 64, 128}


def test_result_to_json_handles_arrays_and_pytrees():
    def fn(lr):
        return {"loss": jnp.ones(3) * lr, "steps": jnp.asarray(2)}

    sweep = GridSearch({"lr": [0.5, 1.5]}).sweep(fn, batch_size=0)

    results = sweep[0]()
    payload = json.loads(results[0].to_json())
    assert payload["arguments"] == {"lr": 0.5}
    assert payload["result"]["loss"] == pytest.approx([0.5, 0.5, 0.5])
    assert payload["result"]["steps"] == 2
    assert payload["num_runs"] == 2


def test_real_ppo_env_and_hparams_sweep():
    """A typical use case: jym.make + PPO.train + evaluate, swept over env and hparams."""
    from jaxnasium.algorithms import PPO
    from jaxnasium.algorithms.architectures import MLP

    def trial(env: str, **hparams):
        environment = jym.make(env)
        agent = PPO(
            total_timesteps=512,
            num_envs=1,
            num_steps=128,
            num_minibatches=1,
            num_epochs=1,
            log_function=None,
            normalize_observations=False,
            normalize_rewards=False,
            actor_kwargs={"body": MLP.with_params(hidden_sizes=(8,))},
            critic_kwargs={"body": MLP.with_params(hidden_sizes=(8,))},
            **hparams,
        )
        key_train, key_eval = jax.random.split(jax.random.PRNGKey(0))
        agent, _ = agent.train(key_train, environment)
        return agent.evaluate(key_eval, environment, num_eval_episodes=2).mean()

    sweep = Sweep(
        trial,
        GridSearch({"env": ["CartPole-v1", "Acrobot-v1"]}),
        RandomSearch(
            {
                "learning_rate_start": (1e-4, 1e-2, "log"),
                "gamma": [0.98, 0.99],
            },
            num_samples=2,
            fixed_seed=jax.random.PRNGKey(1),
        ),
        seed=jax.random.PRNGKey(0),
        batch_size=0,
    )

    # env is static; each sampled hparam set is its own job (default: no batching).
    assert len(sweep) == 2  # 2 envs × 2 samples
    assert [job.static_args["env"] for job in sweep.jobs] == [
        "CartPole-v1",
        "Acrobot-v1",
    ]
    assert all(job.num_runs == 2 for job in sweep.jobs)
    assert all(
        set(job.dynamic_args) == {"learning_rate_start", "gamma"} for job in sweep.jobs
    )

    # fixed_seed → both envs share the same (lr, gamma) draws.
    cartpole = [job for job in sweep.jobs if job.static_args["env"] == "CartPole-v1"]
    acrobot = [job for job in sweep.jobs if job.static_args["env"] == "Acrobot-v1"]
    assert [job.dynamic_args["learning_rate_start"] for job in cartpole] == [
        job.dynamic_args["learning_rate_start"] for job in acrobot
    ]
    assert [job.dynamic_args["gamma"] for job in cartpole] == [
        job.dynamic_args["gamma"] for job in acrobot
    ]

    results = [result for job in sweep.jobs for result in sweep.fn(job)]
    assert len(results) == 4
    assert all(r.num_runs == 2 for r in results)
    assert all(jnp.isfinite(r.result) for r in results)
    assert {r.arguments["env"] for r in results} == {"CartPole-v1", "Acrobot-v1"}
    assert all(1e-4 <= r.arguments["learning_rate_start"] <= 1e-2 for r in results)
    assert all(r.arguments["gamma"] in (0.98, 0.99) for r in results)


def test_one_at_a_time_search_configs():
    search = OneAtATimeSearch(
        {
            "gamma": [0.99, 0.95, 0.999],
            "lr": [0.1, 0.01],
        }
    )
    configs = search.configs(jax.random.PRNGKey(0))
    assert configs == [
        {"gamma": 0.99, "lr": 0.1},
        {"gamma": 0.95, "lr": 0.1},
        {"gamma": 0.999, "lr": 0.1},
        {"gamma": 0.99, "lr": 0.01},
    ]


def test_one_at_a_time_search_fewer_configs_than_grid():
    params = {"a": [1, 2, 3], "b": ["x", "y"]}
    oat = OneAtATimeSearch(params).configs(jax.random.PRNGKey(0))
    grid = GridSearch(params).configs(jax.random.PRNGKey(0))
    assert len(oat) == 1 + 2 + 1
    assert len(grid) == 3 * 2
    assert oat[0] in grid
    assert all(c in grid for c in oat)


def test_one_at_a_time_search_sweep():
    def fn(gamma, lr):
        return gamma * lr

    sweep = OneAtATimeSearch({"gamma": [0.99, 0.95], "lr": [0.1, 0.01]}).sweep(fn)

    assert len(sweep) == 3
    results = [r for run in sweep for r in run()]
    assert [r.arguments for r in results] == [
        {"gamma": 0.99, "lr": 0.1},
        {"gamma": 0.95, "lr": 0.1},
        {"gamma": 0.99, "lr": 0.01},
    ]
    assert [r.result for r in results] == pytest.approx([0.099, 0.095, 0.0099])


def test_one_at_a_time_search_single_valued_params():
    search = OneAtATimeSearch({"a": [1], "b": [2]})
    assert search.configs(jax.random.PRNGKey(0)) == [{"a": 1, "b": 2}]


def test_one_at_a_time_vmaps_dynamic_params():
    def fn(lr, gamma):
        return lr * gamma

    sweep = OneAtATimeSearch({"lr": [0.1, 0.2, 0.3], "gamma": [0.9, 0.99]}).sweep(
        fn, batch_size=0
    )

    # Both params are vmappable, so all four configurations share one job.
    assert len(sweep) == 1
    assert sweep.jobs[0].num_runs == 4

    results = sweep[0]()
    assert all(
        r.result == pytest.approx(r.arguments["lr"] * r.arguments["gamma"])
        for r in results
    )


def test_grid_then_one_at_a_time_chains():
    def fn(env_name, gamma, lr):
        return lr

    sweep = Sweep(
        fn,
        GridSearch({"env_name": ["a", "b"]}),
        OneAtATimeSearch({"gamma": [0.99, 0.95], "lr": [0.1, 0.01]}),
        batch_size=0,
    )

    # The same 3 one-at-a-time configurations are repeated within each env.
    results = [r for run in sweep for r in run()]
    assert len(results) == 6
    per_env = {
        env: sorted(
            (r.arguments["gamma"], r.arguments["lr"])
            for r in results
            if r.arguments["env_name"] == env
        )
        for env in ("a", "b")
    }
    assert per_env["a"] == per_env["b"]
    assert per_env["a"] == [(0.95, 0.1), (0.99, 0.01), (0.99, 0.1)]


def test_cost_estimate_unwraps_dynamic_args(capsys):
    def fn(seed, lr):
        return lr * jax.random.normal(jax.random.PRNGKey(seed))

    _sweep = GridSearch({"seed": [0, 1], "lr": [0.1, 0.2]}).sweep(
        fn, batch_size=2, print_cost_estimate=True
    )

    out = capsys.readouterr().out
    assert "Could not estimate" not in out
