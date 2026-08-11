from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

import jaxnasium as jym

from ._sweep import _jsonify

if TYPE_CHECKING:
    from jaxnasium.algorithms import RLAlgorithm


def _get_algorithm(algorithm: RLAlgorithm | str, hyperparameters: dict) -> RLAlgorithm:
    if isinstance(algorithm, str):
        import jaxnasium.algorithms as jxalgs

        algorithm = algorithm.upper()
        try:
            cls = getattr(jxalgs, algorithm)
        except AttributeError as e:
            available = [
                name
                for name, obj in vars(jxalgs).items()
                if isinstance(obj, type) and issubclass(obj, jxalgs.RLAlgorithm)
            ]
            raise ValueError(
                f"Unknown algorithm {algorithm!r}. Available: {available}"
            ) from e
        return cls(**hyperparameters)

    return replace(algorithm, **hyperparameters) if hyperparameters else algorithm


def _get_env(env: jym.Environment | str) -> tuple[jym.Environment, str]:
    if isinstance(env, str):
        return jym.make(env), env
    return env, type(env).__name__


def _algorithm_params(algorithm: RLAlgorithm) -> dict:
    return {f.name: getattr(algorithm, f.name) for f in fields(algorithm)}


def _as_key(seed: int | PRNGKeyArray) -> PRNGKeyArray:
    """A scalar integer (also as a tracer, when swept) becomes a fresh key; an
    existing key array is passed through."""
    if isinstance(seed, jax.Array) and (
        jnp.issubdtype(seed.dtype, jax.dtypes.prng_key) or seed.ndim > 0
    ):
        return seed
    return jax.random.PRNGKey(seed)


@dataclass(frozen=True)
class AlgorithmEvaluationConfig:
    algorithm: RLAlgorithm
    env: jym.Environment
    env_name: str
    seed: int | PRNGKeyArray
    hyperparameters: dict[str, Any]


@dataclass(frozen=True)
class AlgorithmEvaluation:
    """Train and evaluate a `jaxnasium` algorithm on a given environment for a single seed.
    A possible trail function for a `Sweep`. Typically a `Sweep` should sweep this
    function at least over `seed`.


    Calling this traces cleanly, so it never writes anything itself. Use `context` from a
    `postprocess_fn` to recover the metadata a `vmap` cannot return — most usefully the
    algorithm's *full* field values, including the defaults that were never swept.

    **Arguments**:
        `env`: Environment to train on, or a name to pass to `jym.make`.
        `algorithm`: `RLAlgorithm` instance or name (e.g. `"PPO"`).
        `seed`: Seed for training and evaluation.
        `num_evaluations`: Episodes to evaluate the trained agent over.


    **Example**:
    ```python
    evaluation = AlgorithmEvaluation(env="CartPole-v1")  # or leave empty and sweep
    sweep = Sweep(
        evaluation,
        GridSearch({"algorithm": ["PPO", "DQN"]}),
        RandomSearch({"learning_rate": (1e-4, 1e-2, "log")}, num_samples=64),
        GridSearch({"seed": [0, 1, 2, 3, 4]}),
        seed=jax.random.PRNGKey(0),
        batch_size=8,
    )
    ```
    """

    env: jym.Environment | str = "CartPole-v1"
    algorithm: RLAlgorithm | str = "PPO"
    seed: int | PRNGKeyArray = 0
    num_evaluations: int = 50
    return_train_metrics: bool = False

    def _create_config(self, kwargs: dict[str, Any]) -> AlgorithmEvaluationConfig:
        """Create a config from the given kwargs, filling in defaults from this instance."""
        hyperparameters = dict(kwargs)
        seed = hyperparameters.pop("seed", self.seed)
        env, env_name = _get_env(hyperparameters.pop("env", self.env))
        algorithm = _get_algorithm(
            hyperparameters.pop("algorithm", self.algorithm), hyperparameters
        )
        return AlgorithmEvaluationConfig(
            algorithm, env, env_name, seed, hyperparameters
        )

    def __call__(self, **kwargs: Any) -> Any:
        """Train on one seed and return the evaluation episode returns.

        Returns a dict {"evaluation": evaluation_returns, "train_metrics": train_metrics} if
        `return_train_metrics` is True, otherwise just the evaluation returns.
        """
        config = self._create_config(kwargs)
        train_key, eval_key = jax.random.split(_as_key(config.seed))

        agent, train_metrics = config.algorithm.train(train_key, config.env)
        evaluation = agent.evaluate(
            eval_key, config.env, num_eval_episodes=self.num_evaluations
        )

        if not self.return_train_metrics:
            return evaluation
        return {"evaluation": evaluation, "train_metrics": train_metrics}

    def static_params(self, params: dict[str, Any]) -> set[str]:
        """Static parameters on the env/algorithm won't cannot be vmapped, so marking them here
        stops attempting to trace them.
        """
        static = {"env", "algorithm"}
        for algorithm in params.get("algorithm", [self.algorithm]):
            resolved = _get_algorithm(algorithm, {})
            static |= {
                f.name for f in fields(resolved) if f.metadata.get("static", False)
            }
        return static

    def context(self, **kwargs: Any) -> dict[str, Any]:
        """
        JSON-ready metadata for a configuration, to pair with its result.
        Used to retrieve the full algorithm and env parameters that were used for this run.
        """
        config = self._create_config(kwargs)
        return {
            "algorithm": type(config.algorithm).__name__,
            "env": config.env_name,
            "seed": _jsonify(config.seed),
            "num_evaluations": self.num_evaluations,
            "input_parameters": _jsonify(config.hyperparameters),
            "algorithm_parameters": _jsonify(_algorithm_params(config.algorithm)),
        }

    def __repr__(self) -> str:
        algorithm = (
            self.algorithm
            if isinstance(self.algorithm, str)
            else type(self.algorithm).__name__
        )
        return f"AlgorithmEvaluation({algorithm} on {_get_env(self.env)[1]})"
