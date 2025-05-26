import difflib
import importlib
from typing import Optional

import jymkit.envs
from jymkit._environment import Environment

from ._wrappers import Wrapper
from ._wrappers_ext import BraxWrapper, GymnaxWrapper, JumanjiWrapper

JYMKIT_ENVS = [
    "CartPole-v1",
    "Acrobot-v1",
    "Pendulum-v1",
    "MountainCar-v0",
    "ContinuousMountainCar-v0",
]

# External environments, requires the respective packages to be installed
GYMNAX_ENVS = [
    "gymnax:CartPole-v1", "gymnax:Acrobot-v1", "gymnax:Pendulum-v1", "gymnax:MountainCar-v0", "gymnax:ContinuousMountainCar-v0",
    "Asterix-MinAtar", "Breakout-MinAtar", "Freeway-MinAtar",
    "SpaceInvaders-MinAtar", "DeepSea-bsuite", "Catch-bsuite", "MemoryChain-bsuite",
    "UmbrellaChain-bsuite", "DiscountingChain-bsuite", "MNISTBandit-bsuite", "SimpleBandit-bsuite",
    "FourRooms-misc", "MetaMaze-misc", "PointRobot-misc", "BernoulliBandit-misc",
    "GaussianBandit-misc", "Reacher-misc", "Swimmer-misc", "Pong-misc",
]  # fmt: skip

JUMANJI_ENVS = [
    "Game2048-v1", "GraphColoring-v0", "Minesweeper-v0", "RubiksCube-v0",
    "RubiksCube-partly-scrambled-v0", "SlidingTilePuzzle-v0", "Sudoku-v0", "Sudoku-very-easy-v0",
    "BinPack-v1", "FlatPack-v0", "JobShop-v0", "Knapsack-v1",
    "Tetris-v0", "Cleaner-v0", "Connector-v2", "CVRP-v1",
    "MultiCVRP-v0", "Maze-v0", "RobotWarehouse-v0", "Snake-v1",
    "TSP-v1", "MMST-v0", "PacMan-v1", "Sokoban-v0",
    "LevelBasedForaging-v0", "SearchAndRescue-v0",
]  # fmt: skip

BRAX_ENVS = [
    "ant", "halfcheetah", "hopper", "humanoid",
    "humanoidstandup", "inverted_pendulum", "inverted_double_pendulum", "pusher", 
    "reacher", "walker2d",
]  # fmt: skip

ALL_ENVS = JYMKIT_ENVS + GYMNAX_ENVS + JUMANJI_ENVS + BRAX_ENVS  # fmt: skip


def make(
    env_name: str,
    wrapper: Optional[Wrapper] = None,
    external_package: Optional[str] = None,
    **env_kwargs,
) -> Environment:
    if env_name is None:
        raise ValueError("Environment name cannot be None.")
    if external_package is not None:
        # try to import package_name
        try:
            ext_module = importlib.import_module(external_package)
        except ImportError:
            raise ImportError(f"{external_package} is not found. Is it installed?")
        try:
            env = getattr(ext_module, env_name)(**env_kwargs)
        except AttributeError:
            raise AttributeError(
                f"Environment {env_name} is not found in {external_package}."
            )

    elif env_name in JYMKIT_ENVS:
        if env_name == "CartPole-v1":
            env = jymkit.envs.CartPole(**env_kwargs)
        elif env_name == "Acrobot-v1":
            env = jymkit.envs.Acrobot(**env_kwargs)
        elif env_name == "Pendulum-v1":
            env = jymkit.envs.Pendulum(**env_kwargs)
        elif env_name == "MountainCar-v0":
            env = jymkit.envs.MountainCar(**env_kwargs)
        elif env_name == "ContinuousMountainCar-v0":
            env = jymkit.envs.ContinuousMountainCar(**env_kwargs)

    elif env_name in GYMNAX_ENVS:
        try:
            import gymnax
        except ImportError:
            raise ImportError(
                "Using an environment from Gymnax, but Gymnax is not installed."
                "Please install it with `pip install gymnax`."
            )
        print(f"Using an environment from Gymnax via gymnax.make({env_name}).")
        env_name = env_name.strip("gymnax:")
        env, _ = gymnax.make(env_name, **env_kwargs)
        if wrapper is None:
            print(
                "Wrapping Gymnax environment with GymnaxWrapper\n",
                " Disable this behavior by passing wrapper=False",
            )
            env = GymnaxWrapper(env)
    elif env_name in JUMANJI_ENVS:
        try:
            import jumanji
        except ImportError:
            raise ImportError(
                "Using an environment from Jumanji, but Jumanji is not installed."
                "Please install it with `pip install jumanji`."
            )
        print(f"Using an environment from Jumanji via jumanji.make({env_name}).")
        env = jumanji.make(env_name, **env_kwargs)  # type: ignore
        if wrapper is None:
            print(
                "Wrapping Jumanji environment with JumanjiWrapper\n",
                " Disable this behavior by passing wrapper=False",
            )
            env = JumanjiWrapper(env)

    elif env_name in BRAX_ENVS:
        try:
            import brax.envs
        except ImportError:
            raise ImportError(
                "Using an environment from Brax, but Brax is not installed."
                "Please install it with `pip install brax`."
            )
        print(
            f"Using an environment from Brax via brax.envs.get_environment({env_name})."
        )
        env = brax.envs.get_environment(env_name, **env_kwargs)
        if wrapper is None:
            print(
                "Wrapping Brax environment with BraxWrapper\n",
                " Disable this behavior by passing wrapper=False",
            )
            env = BraxWrapper(env)
    else:
        matches = difflib.get_close_matches(env_name, ALL_ENVS, n=1, cutoff=0.6)

        def format_env_group(envs, group_name):
            if not envs:
                return ""
            envs_per_line = 4
            max_length = max(len(env) for env in envs)
            formatted_envs = "\n".join(
                [
                    " | ".join(
                        env.ljust(max_length) for env in envs[i : i + envs_per_line]
                    )
                    for i in range(0, len(envs), envs_per_line)
                ]
            )
            return f"{group_name}:\n{formatted_envs}\n"

        suggestion = (
            f" Did you mean {matches[0]}?"
            if matches
            else " \nAvailable environments are:\n\n"
            + format_env_group(JYMKIT_ENVS, "JymKit Envs")
            + "\n"
            + format_env_group(GYMNAX_ENVS, "Gymnax Envs")
            + "\n"
            + format_env_group(JUMANJI_ENVS, "Jumanji Envs")
            + "\n"
            + format_env_group(BRAX_ENVS, "Brax Envs")
        )
        raise ValueError(f"Environment {env_name} not found.{suggestion}")

    if wrapper is not None:
        if isinstance(wrapper, Wrapper):
            env = wrapper(env)  # type: ignore
        else:
            raise ValueError("Wrapper must be an instance of Wrapper class.")
    return env  # type: ignore
