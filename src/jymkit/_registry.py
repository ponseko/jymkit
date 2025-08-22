import difflib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Type

from ._environment import Environment
from ._wrappers import LogWrapper
from ._wrappers_ext import BraxWrapper, GymnaxWrapper, JumanjiWrapper, Wrapper

logger = logging.getLogger(__name__)


def _wrap_env(env: Environment | Any, wrapper: Type[Wrapper]) -> Environment:
    """Simply wraps an environment and outputs what happened to a logger"""
    logger.info(f"Wrapping environment with {wrapper.__name__}")
    return wrapper(env)


@dataclass
class Registry:
    _environments: Dict[str, Type[Environment]] = field(default_factory=dict)
    _aliases: Dict[str, str] = field(default_factory=dict)

    def register(self, id: str, **kwargs):
        """Register an environment with the registry.

        **Arguments**:
            `id`: The environment ID (e.g., "CartPole-v1")
            `entry_point`: Optional entry point string for lazy loading
            `**kwargs`: currently unused
        """

        def decorator(env_class: Type[Environment]) -> Type[Environment]:
            self._environments[id] = env_class
            return env_class

        return decorator

    def register_alias(self, alias: str, target: str):
        """Register an alias for an environment.

        **Arguments**:
            `alias`: The alias to register
            `target`: The target environment ID
        """
        self._aliases[alias] = target

    def make(
        self,
        id: str,
        wrappers: List[Type[Wrapper] | Literal["external_lib_wrapper"]] = [
            "external_lib_wrapper",
            LogWrapper,
        ],
        **env_kwargs,
    ) -> Environment:
        """Create an environment instance.

        **Arguments**:
            `id`: The environment ID
            `wrappers`: List of wrappers to apply to the environment.
                - `external_lib_wrapper` (string): Wrapper for external libraries (e.g. Gymnax, Jumanji, Brax).
                    only used if a environment is loaded from a supported external library.
                - `LogWrapper` (class): Wrapper for logging the actions
            `**env_kwargs`: Environment constructor arguments
        """
        # Handle aliases
        env = None
        if id in self._aliases:
            id = self._aliases[id]

        # Try direct registration first
        if id in self._environments:
            env_class = self._environments[id]
            env = env_class(**env_kwargs)

        # Try external package detection
        if ":" in id:
            package, env_name = id.split(":", 1)
            wrap = "external_lib_wrapper" in wrappers
            env = self._make_external(package, env_name, wrap=wrap, **env_kwargs)

        if env is not None:
            for wrapper in wrappers:
                if wrapper == "external_lib_wrapper":
                    continue  # applied in _make_external (if applicable)
                env = _wrap_env(env, wrapper)
            return env

        matches = difflib.get_close_matches(id, self.get_env_list(), n=1, cutoff=0.6)
        if matches:
            raise ValueError(
                f"Environment {id} not found in registry. Did you mean {matches[0]}?"
            )
        else:
            raise ValueError(f"Environment {id} not found in registry")

    def _make_external(
        self, package: str, env_name: str, *, wrap: bool, **env_kwargs
    ) -> Environment:
        """Create an external environment with appropriate wrapper.

        **Arguments**:
            `package`: The package to use (e.g., "gymnax", "jumanji", "brax", ...)
            `env_name`: The environment name
            `**env_kwargs`: Environment constructor arguments
        """
        try:
            if package == "gymnax":
                import gymnax

                env, _ = gymnax.make(env_name, **env_kwargs)
                if wrap:
                    return _wrap_env(env, GymnaxWrapper)
                return env  # type: ignore
            elif package == "jumanji":
                import jumanji

                env = jumanji.make(env_name, **env_kwargs)  # type: ignore
                if wrap:
                    return _wrap_env(env, JumanjiWrapper)
                return env  # type: ignore
            elif package == "brax":
                import brax.envs

                env = brax.envs.get_environment(env_name, **env_kwargs)
                if wrap:
                    return _wrap_env(env, BraxWrapper)
                return env  # type: ignore
            else:
                raise ValueError(f"Unsupported/unknown external package: {package}")
        except ImportError as e:
            raise ImportError(
                f"Package {package} not installed. Please install manually via pip: {e}"
            )

    def get_env_class(self, id: str) -> Type[Environment]:
        """Get the environment class for an environment ID.

        **Arguments**:
            `id`: The environment ID
        """
        if id in self._aliases:
            id = self._aliases[id]

        if id in self._environments:
            return self._environments[id]

        if ":" in id:
            raise ValueError("Cannot get environment class for external environments")

        raise ValueError(f"Environment {id} not found in registry")

    def get_env_list(self) -> List[str]:
        """List all environments in the registry as a flat list."""
        return list(self._environments.keys()) + list(self._aliases.keys())

    def show_envs(self) -> None:
        """Pretty prints the available environments in the registry."""
        print("Available environments in JymKit:")
        print("=" * 50)

        def format_block(envs, title, icon):
            if not envs:
                return ""
            envs_per_line = 3
            # Get max length across ALL environments for consistent column width
            all_envs = list(self._environments.keys()) + [
                alias for alias in self._aliases.keys()
            ]
            max_length = max(len(env) for env in all_envs) if all_envs else 0
            formatted = "\n".join(
                " ".join(
                    f"• {env}".ljust(max_length + 2)
                    for env in envs[i : i + envs_per_line]
                )
                for i in range(0, len(envs), envs_per_line)
            )
            return f"\n{icon} {title}:\n{formatted}"

        # Native environments
        if self._environments:
            native_envs = sorted(self._environments.keys())
            print(format_block(native_envs, "Native environments", "🔧"))

        # Group external environments by package
        external_packages = {}
        for alias, target in self._aliases.items():
            if ":" in target:
                package = target.split(":")[0]
                if package not in external_packages:
                    external_packages[package] = []
                external_packages[package].append(alias)

        # Print external environments grouped by package
        for package in sorted(external_packages.keys()):
            envs = sorted(external_packages[package])
            print(format_block(envs, f"{package.title()} environments", "📦"))

        print(f"\nTotal: {len(self._environments) + len(self._aliases)} environments")
        print(
            "Note that external libraries📦 are not bundled as dependencies and need to be installed manually (e.g. via pip)."
        )
        print("=" * 50)


registry = Registry()
make = registry.make


# # External environments, requires the respective packages to be installed
# GYMNAX_ENVS = [
#     "gymnax:CartPole-v1", "gymnax:Acrobot-v1", "gymnax:Pendulum-v1", "gymnax:MountainCar-v0", "gymnax:ContinuousMountainCar-v0",
#     "Asterix-MinAtar", "Breakout-MinAtar", "Freeway-MinAtar",
#     "SpaceInvaders-MinAtar", "DeepSea-bsuite", "Catch-bsuite", "MemoryChain-bsuite",
#     "UmbrellaChain-bsuite", "DiscountingChain-bsuite", "MNISTBandit-bsuite", "SimpleBandit-bsuite",
#     "FourRooms-misc", "MetaMaze-misc", "PointRobot-misc", "BernoulliBandit-misc",
#     "GaussianBandit-misc", "Reacher-misc", "Swimmer-misc", "Pong-misc",
# ]  # fmt: skip

# JUMANJI_ENVS = [
#     "Game2048-v1", "GraphColoring-v0", "Minesweeper-v0", "RubiksCube-v0",
#     "RubiksCube-partly-scrambled-v0", "SlidingTilePuzzle-v0", "Sudoku-v0", "Sudoku-very-easy-v0",
#     "BinPack-v1", "FlatPack-v0", "JobShop-v0", "Knapsack-v1",
#     "Tetris-v0", "Cleaner-v0", "Connector-v2", "CVRP-v1",
#     "MultiCVRP-v0", "Maze-v0", "RobotWarehouse-v0", "Snake-v1",
#     "TSP-v1", "MMST-v0", "PacMan-v1", "Sokoban-v0",
#     "LevelBasedForaging-v0", "SearchAndRescue-v0",
# ]  # fmt: skip

# BRAX_ENVS = [
#     "ant", "halfcheetah", "hopper", "humanoid",
#     "humanoidstandup", "inverted_pendulum", "inverted_double_pendulum", "pusher",
#     "reacher", "walker2d",
# ]  # fmt: skip


# Gymnax envs
# Classic control accessible only with "gymnax:" prefix as they are included in JymKit
registry.register_alias("gymnax:CartPole-v1", "gymnax:CartPole-v1")
registry.register_alias("gymnax:Acrobot-v1", "gymnax:Acrobot-v1")
registry.register_alias("gymnax:Pendulum-v1", "gymnax:Pendulum-v1")
registry.register_alias("gymnax:MountainCar-v0", "gymnax:MountainCar-v0")
registry.register_alias(
    "gymnax:ContinuousMountainCar-v0", "gymnax:ContinuousMountainCar-v0"
)
registry.register_alias("Asterix-MinAtar", "gymnax:Asterix-MinAtar")
registry.register_alias("Breakout-MinAtar", "gymnax:Breakout-MinAtar")
registry.register_alias("Freeway-MinAtar", "gymnax:Freeway-MinAtar")
registry.register_alias("SpaceInvaders-MinAtar", "gymnax:SpaceInvaders-MinAtar")
registry.register_alias("DeepSea-bsuite", "gymnax:DeepSea-bsuite")
registry.register_alias("Catch-bsuite", "gymnax:Catch-bsuite")
registry.register_alias("MemoryChain-bsuite", "gymnax:MemoryChain-bsuite")
registry.register_alias("UmbrellaChain-bsuite", "gymnax:UmbrellaChain-bsuite")
registry.register_alias("DiscountingChain-bsuite", "gymnax:DiscountingChain-bsuite")
registry.register_alias("MNISTBandit-bsuite", "gymnax:MNISTBandit-bsuite")
registry.register_alias("SimpleBandit-bsuite", "gymnax:SimpleBandit-bsuite")
registry.register_alias("FourRooms-misc", "gymnax:FourRooms-misc")
registry.register_alias("MetaMaze-misc", "gymnax:MetaMaze-misc")
registry.register_alias("PointRobot-misc", "gymnax:PointRobot-misc")
registry.register_alias("BernoulliBandit-misc", "gymnax:BernoulliBandit-misc")
registry.register_alias("GaussianBandit-misc", "gymnax:GaussianBandit-misc")
registry.register_alias("Reacher-misc", "gymnax:Reacher-misc")
registry.register_alias("Swimmer-misc", "gymnax:Swimmer-misc")
registry.register_alias("Pong-misc", "gymnax:Pong-misc")

# Jumanji envs
registry.register_alias("Game2048-v1", "jumanji:Game2048-v1")
registry.register_alias("GraphColoring-v0", "jumanji:GraphColoring-v0")
registry.register_alias("Minesweeper-v0", "jumanji:Minesweeper-v0")
registry.register_alias("RubiksCube-v0", "jumanji:RubiksCube-v0")
registry.register_alias(
    "RubiksCube-partly-scrambled-v0", "jumanji:RubiksCube-partly-scrambled-v0"
)
registry.register_alias("SlidingTilePuzzle-v0", "jumanji:SlidingTilePuzzle-v0")
registry.register_alias("Sudoku-v0", "jumanji:Sudoku-v0")
registry.register_alias("Sudoku-very-easy-v0", "jumanji:Sudoku-very-easy-v0")
registry.register_alias("BinPack-v1", "jumanji:BinPack-v1")
registry.register_alias("FlatPack-v0", "jumanji:FlatPack-v0")
registry.register_alias("JobShop-v0", "jumanji:JobShop-v0")
registry.register_alias("Knapsack-v1", "jumanji:Knapsack-v1")
registry.register_alias("Tetris-v0", "jumanji:Tetris-v0")
registry.register_alias("Cleaner-v0", "jumanji:Cleaner-v0")
registry.register_alias("Connector-v2", "jumanji:Connector-v2")
registry.register_alias("CVRP-v1", "jumanji:CVRP-v1")
registry.register_alias("MultiCVRP-v0", "jumanji:MultiCVRP-v0")
registry.register_alias("Maze-v0", "jumanji:Maze-v0")
registry.register_alias("RobotWarehouse-v0", "jumanji:RobotWarehouse-v0")
registry.register_alias("Snake-v1", "jumanji:Snake-v1")
registry.register_alias("TSP-v1", "jumanji:TSP-v1")
registry.register_alias("MMST-v0", "jumanji:MMST-v0")
registry.register_alias("PacMan-v1", "jumanji:PacMan-v1")
registry.register_alias("Sokoban-v0", "jumanji:Sokoban-v0")
registry.register_alias("LevelBasedForaging-v0", "jumanji:LevelBasedForaging-v0")
registry.register_alias("SearchAndRescue-v0", "jumanji:SearchAndRescue-v0")

# Brax envs
registry.register_alias("ant", "brax:ant")
registry.register_alias("halfcheetah", "brax:halfcheetah")
registry.register_alias("hopper", "brax:hopper")
registry.register_alias("humanoid", "brax:humanoid")
registry.register_alias("humanoidstandup", "brax:humanoidstandup")
registry.register_alias("inverted_pendulum", "brax:inverted_pendulum")
registry.register_alias("inverted_double_pendulum", "brax:inverted_double_pendulum")
registry.register_alias("pusher", "brax:pusher")
registry.register_alias("reacher", "brax:reacher")
registry.register_alias("walker2d", "brax:walker2d")
