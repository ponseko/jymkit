from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import Field, dataclass, fields, replace
from functools import partial
from typing import Any, ClassVar, Protocol, overload

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from ._probe import log_cost_estimate, split_static_dynamic_params

logger = logging.getLogger(__name__)


class ParameterSpaceSearch(Protocol):
    @property
    def params(self) -> dict[str, list | tuple] | dict[str, list]: ...

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    def configs(self, seed: PRNGKeyArray) -> list[dict[str, Any]]: ...


def _jsonify(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _jsonify(value.tolist())
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass
class SweepResult:
    """Simple container for the results of a sweep job.
    Most prominently, this contains the `result` and the `arguments` that produced it.

    **Attributes**:
        `start_time`: Unix time at which the job containing this run started.
        `end_time`: Unix time at which the job containing this run finished.
        `arguments`: The parameter values that produced this result.
        `result`: Whatever the swept function returned, for this configuration.
        `num_runs`: How many configurations shared the job, i.e. the `vmap` batch
            size. The timestamps cover all of them, not this run alone.
    """

    start_time: float
    end_time: float
    arguments: dict[str, Any]
    result: Any
    num_runs: int

    @property
    def duration(self) -> float:
        """Wall-clock seconds spent on the job containing this run.
        Note that some jobs may have multiple runs vmapped over (`num_runs` > 1).
        In that case, the duration is the time spent on the complete vmapped batch of job runs."""
        return self.end_time - self.start_time

    def to_json(self, **kwargs: Any) -> str:
        """Serialize to JSON, stringifying anything not natively serializable."""
        return json.dumps(
            {field.name: _jsonify(getattr(self, field.name)) for field in fields(self)},
            **kwargs,
        )

    def to_json_no_result(self, **kwargs: Any) -> str:
        """Serialize to JSON, stringifying anything not natively serializable, excluding the result."""
        return json.dumps(
            {
                field.name: _jsonify(getattr(self, field.name))
                for field in fields(self)
                if field.name != "result"
            },
            **kwargs,
        )


@dataclass
class SweepJob:
    """One dispatchable unit of work: a single call to the swept function.

    **Attributes**:
        `static_args`: Kwargs held fixed for the whole job, one value each.
        `dynamic_args`: Kwargs varied *within* the job, a batch of values each,
            all of the same length. These are what `vmap` maps over.
    """

    static_args: dict[str, Any]
    dynamic_args: dict[str, list]

    @property
    def num_runs(self) -> int:
        """How many configurations this job evaluates."""
        if not self.dynamic_args:
            return 1
        return len(next(iter(self.dynamic_args.values())))

    @property
    def arguments(self) -> dict[str, Any]:
        """Combined static and dynamic arguments of this job."""
        return {**self.static_args, **self.dynamic_args}


@dataclass(frozen=True)
class Sweep:
    """Build a parameter sweep over function `fn` from one or more stages of parameter
    space search, like `GridSearch`, `OneAtATimeSearch`, `RandomSearch`, or `SobolSearch`.

    For a single stage, `RandomSearch(...).sweep(fn, ...)` is equivalent.

    **Arguments**:
        `fn`: The function to sweep. Must accept all swept params as keywords.
        `*stages`: `GridSearch` / `OneAtATimeSearch` / `RandomSearch` / `SobolSearch` stages to nest.
        `seed`: Randomness for configs from sampling stages that omit `fixed_seed`.
            Required unless every sampling stage sets `fixed_seed` (or there are none).
        `batch_size`: How many configurations to `vmap` in one job. Defaults to
            `None` (no batching: one configuration per job). Pass `0` for
            unlimited batching of configs that share the same static args, or a
            positive int to cap the vmap size. `fn` will be traced to determine which
            parameters are dynamic and can be `vmap`ed together.
        `print_cost_estimate`: After creation, print the cost estimate of the first job to provide a
        rough estimate of the memory requirements of a job in the sweep. This triggers a compilation, so
        adds some overhead to the creation of the sweep.

    **Attributes**:
        `fn`: Runs one `SweepJob`, returning a `SweepResult` per configuration.
        `jobs`: All jobs, to run in a loop or to dispatch by index (e.g. over a slurm job array).

    **Example**:
    ```python
    sweep = Sweep(
        train,
        GridSearch({"env_name": ["CartPole-v1", "Acrobot-v1"]}),
        RandomSearch(
            {"learning_rate": (1e-4, 1e-2, "log")},
            num_samples=64,
            fixed_seed=jax.random.PRNGKey(1),  # same lrs in both envs
        ),
        seed=jax.random.PRNGKey(0),
        batch_size=16,
    )
    len(sweep)  # 2 envs * (64 learning rates / 16 per vmap)
    for run in sweep:
        for r in run():
            print(r.arguments, r.result)
    # Or dispatch one job: for r in sweep[0](): ...
    ```

    Or in a slurm job array:
    ```bash
    #SBATCH --array=0-5
    python create_jobs_and_run_id.py --batch_idx $SLURM_ARRAY_TASK_ID --seed 0
    ```
    """

    fn: Callable[[SweepJob], list[SweepResult]]
    jobs: list[SweepJob]

    def __len__(self):
        return len(self.jobs)

    def __iter__(self):
        return (partial(self.fn, job) for job in self.jobs)

    @overload
    def __getitem__(self, index: int) -> partial[list[SweepResult]]: ...
    @overload
    def __getitem__(self, index: slice) -> Sweep: ...

    def __getitem__(self, index: int | slice) -> partial[list[SweepResult]] | Sweep:
        if isinstance(index, slice):
            # Custom __init__ builds from stages; bypass it when slicing jobs.
            sliced = object.__new__(Sweep)
            object.__setattr__(sliced, "fn", self.fn)
            object.__setattr__(sliced, "jobs", self.jobs[index])
            return sliced
        return partial(self.fn, self.jobs[index])

    def __init__(
        self,
        fn: Callable,
        *stages: ParameterSpaceSearch,
        seed: PRNGKeyArray | None = None,
        batch_size: int | None = None,
        print_cost_estimate: bool = False,
    ):
        if not stages:
            raise ValueError("At least one sweep stage is required")
        if batch_size is not None and batch_size < 0:
            raise ValueError(
                f"batch_size must be None, 0, or positive, got {batch_size}"
            )
        max_batch = (
            1 if batch_size is None else (None if batch_size == 0 else batch_size)
        )

        needs_parent_seed = any(
            hasattr(stage, "fixed_seed") and stage.fixed_seed is None  # type: ignore
            for stage in stages
        )
        if seed is None:
            if needs_parent_seed:
                raise ValueError(
                    "seed is required when a Random/Sobol Search stage omits fixed_seed"
                )
            seed = jax.random.PRNGKey(0)  # unused when every stage pins fixed_seed

        all_params: dict[str, list | tuple] = {}
        for stage in stages:
            for name, values in stage.params.items():
                if name in all_params:
                    raise ValueError(f"Param {name!r} is swept by more than one stage")
                all_params[name] = values

        if batch_size is None:
            static_params = all_params
            dynamic_params = {}
        else:
            static_params, dynamic_params = split_static_dynamic_params(fn, all_params)

        # Order each stage so its static params vary slowest, keeping configurations
        # that can share a job (equal static args) adjacent in the expansion below.
        stages = tuple(
            replace(
                stage,
                params={
                    **{n: v for n, v in stage.params.items() if n in static_params},
                    **{n: v for n, v in stage.params.items() if n in dynamic_params},
                },
            )
            for stage in stages
        )

        # Make a list of all configuration combinations where we have sampled from random stages.
        configs: list[dict[str, Any]] = [{}]
        for stage in stages:
            seed, stage_seed = jax.random.split(seed)  # type: ignore
            expanded: list[dict[str, Any]] = []
            for i, config in enumerate(configs):
                if getattr(stage, "fixed_seed", None) is not None:
                    config_seed: PRNGKeyArray = stage.fixed_seed  # type: ignore
                else:  # Fresh draws per outer configuration when the stage does not pin a key.
                    config_seed: PRNGKeyArray = jax.random.fold_in(stage_seed, i)
                expanded.extend(
                    {**config, **inner} for inner in stage.configs(config_seed)
                )
            configs = expanded

        # Static args are put adjecent, so we vmap dynamic args where possible (if static args are the same).
        jobs: list[SweepJob] = []
        for config in configs:
            static_args = {name: config[name] for name in static_params}
            # fmt: off
            if not (
                dynamic_params # if there are dynamic params
                and jobs # if there are jobs
                and jobs[-1].static_args == static_args # if the last job has the same static args
                and (max_batch is None or jobs[-1].num_runs < max_batch)
            ):
                # if not any of the above, we create a new job
                jobs.append(SweepJob(static_args, {name: [] for name in dynamic_params}))
            for name in dynamic_params:
                jobs[-1].dynamic_args[name].append(config[name]) # add the dynamic arg to the last job

        # fmt: on

        def mapped_fn(job: SweepJob) -> list[SweepResult]:
            if not job.dynamic_args:
                start_time = time.time()
                result = jax.block_until_ready(fn(**job.static_args))
                return [
                    SweepResult(
                        start_time=start_time,
                        end_time=time.time(),
                        arguments=dict(job.static_args),
                        result=result,
                        num_runs=1,
                    )
                ]

            names = list(job.dynamic_args)
            batch = tuple(jnp.asarray(job.dynamic_args[name]) for name in names)
            batched_fn = jax.vmap(
                lambda *values: fn(**job.static_args, **dict(zip(names, values)))
            )

            start_time = time.time()
            results = jax.block_until_ready(batched_fn(*batch))
            end_time = time.time()

            return [
                SweepResult(
                    start_time=start_time,
                    end_time=end_time,
                    arguments={
                        **job.static_args,
                        **{name: job.dynamic_args[name][i] for name in names},
                    },
                    result=jax.tree.map(lambda leaf, _i=i: leaf[_i], results),
                    num_runs=job.num_runs,
                )
                for i in range(job.num_runs)
            ]

        largest_batch = max(job.num_runs for job in jobs)
        requested = "unlimited" if max_batch is None else max_batch
        logger.info(
            f"Created {len(jobs)} sweep jobs for {len(configs)} configurations "
            f"(per job: {list(static_params)} fixed, {list(dynamic_params)} vmapped). "
            f"Largest job batches {largest_batch} configuration(s), of {requested} allowed."
        )
        if batch_size is not None and dynamic_params and largest_batch == 1:
            logger.warning(
                f"Batching was requested (batch_size={batch_size}) but no configurations "
                "could be batched, so nothing is vmapped. A job batches configurations "
                f"that agree on the fixed parameters {list(static_params)} and that are "
                "adjacent, and stages are nested in the order given — so a stage varying "
                "only vmappable parameters has to come last to end up adjacent. Try "
                f"moving the stage(s) sweeping {list(dynamic_params)} to the end."
            )

        if print_cost_estimate:
            first_job = jobs[0]
            un_vmapped_first_job = SweepJob(
                first_job.static_args,
                {
                    name: [first_job.dynamic_args[name][0]]
                    for name in first_job.dynamic_args
                },
            )
            print(
                "Estimating the cost of the first job without any arguments mapped over.",
                " Note that this is only a proxy. Jobs in this sweep with different input arguments may have different costs,"
                " especially when sweeping over different environments, num_envs etc.",
                " Compiling...",
            )
            print(log_cost_estimate(fn, **un_vmapped_first_job.arguments))
            print("Done.")
            print(
                "Proxied a single configuration. In this sweep, the largest job runs",
                f"{max(jobs, key=lambda j: j.num_runs).num_runs} configuration(s) in parallel (vmap batch size).",
            )

        object.__setattr__(self, "fn", mapped_fn)
        object.__setattr__(self, "jobs", jobs)
