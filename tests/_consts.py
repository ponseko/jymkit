from typing import Type

from jaxnasium.algorithms import DQN, PPO, PQN, SAC  # noqa: F401
from jaxnasium.algorithms.architectures import MLP
from jaxnasium.wrappers import Wrapper

DISCRETE_ALGS = [PPO, PQN, DQN, SAC]
CONTINUOUS_ALGS = [PPO, SAC]

# Optional wrappers applied before environment use.
ENV_WRAPPERS: dict[str, list[Type[Wrapper]]] = {}

AGENT_MIN_CONFIG = {
    "num_envs": 1,
    "total_timesteps": 512,
    "log_function": None,
    "normalize_observations": False,
    "normalize_rewards": False,
    "actor_kwargs": {"body": MLP.with_params(hidden_sizes=(8,))},
    "critic_kwargs": {"body": MLP.with_params(hidden_sizes=(8,))},
}


# Skipped due some bugs in the environments
SKIP_ENVS: dict[str, str] = {
    "SimpleBandit-bsuite": "bug on reset: https://github.com/RobertTLange/gymnax/issues/110",
    "_SUITE_:jaxmarl": "JaxMarl technically works on old versions of jax/flax, but is left out of tests until jaxmarl 2.0 is released https://github.com/FLAIROx/JaxMARL/pull/186",
}

# Skipped for some limatation in default configuration of algorithms
SKIP_AGENT_ENVS: dict[str, str] = {
    "BinPack-v2": "Action masking not supported due to actions being conditionally dependent",
    "Tetris-v0": "Action masking not supported due to actions being conditionally dependent",
    "Minesweeper-v0": "Action masking not supported due to actions being conditionally dependent",
    "Sudoku-v0": "Action masking not supported due to actions being conditionally dependent",
    "Sudoku-very-easy-v0": "Action masking not supported due to actions being conditionally dependent",
    "FlatPack-v0": "Action masking not supported due to actions being conditionally dependent",
    "RubiksCube-v0": "Heterogeous MultiDiscrete action space",
    "RubiksCube-partly-scrambled-v0": "Heterogeous MultiDiscrete action space",
}
