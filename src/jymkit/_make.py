import difflib
import importlib
from typing import Optional

import jymkit.envs
from jymkit._environment import Environment

from ._wrappers import GymnaxWrapper, Wrapper

JYMKIT_ENVS = [
    "CartPole",
    "Acrobot",
]

# Gymnax environments, requires gymnax package to be installed
# Included for convenience, but not all environments are compatible with the latest version of Jax
GYMNAX_ENVS = [
    "Pendulum-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "Asterix-MinAtar",  # Gymnax no longer compatible with newer versions of Jax for this env
    "Breakout-MinAtar",
    "Freeway-MinAtar",  # Gymnax no longer compatible with newer versions of Jax for this env
    "SpaceInvaders-MinAtar",
    "DeepSea-bsuite",
    # Untested envs:
    "Catch-bsuite",
    "MemoryChain-bsuite",
    "UmbrellaChain-bsuite",
    "DiscountingChain-bsuite",
    "MNISTBandit-bsuite",
    "SimpleBandit-bsuite",
    "FourRooms-misc",
    "MetaMaze-misc",
    "PointRobot-misc",
    "BernoulliBandit-misc",
    "GaussianBandit-misc",
    "Reacher-misc",
    "Swimmer-misc",
    "Pong-misc",
]

ALL_ENVS = JYMKIT_ENVS + GYMNAX_ENVS


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
        if env_name == "CartPole":
            env = jymkit.envs.CartPole(**env_kwargs)
        elif env_name == "Acrobot":
            env = jymkit.envs.Acrobot(**env_kwargs)

    elif env_name in GYMNAX_ENVS:
        try:
            import gymnax
        except ImportError:
            raise ImportError(
                "Using an environment from Gymnax, but Gymnax is not installed."
                "Please install it with `pip install gymnax`."
            )
        print(f"Using an environment from Gymnax via gymnax.make({env_name}).")
        env, _ = gymnax.make(env_name, **env_kwargs)
        if wrapper is None:
            print(
                "Wrapping Gymnax environment with GymnaxWrapper\n",
                " Disable this behavior by passing wrapper=False",
            )
            env = GymnaxWrapper(env)
            print("some minor change")
    else:
        matches = difflib.get_close_matches(env_name, ALL_ENVS, n=1, cutoff=0.6)
        envs_per_line = 3
        max_length = max(len(env) for env in ALL_ENVS)  # Longest env name
        suggestion = (
            f" Did you mean {matches[0]}?"
            if matches
            else " \nAvailable environments are:\n"
            + "\n".join(
                [
                    " | ".join(
                        env.ljust(max_length) for env in ALL_ENVS[i : i + envs_per_line]
                    )
                    for i in range(0, len(ALL_ENVS), envs_per_line)
                ]
            )
        )
        raise ValueError(f"Environment {env_name} not found.{suggestion}")

    if wrapper is not None:
        if isinstance(wrapper, Wrapper):
            env = wrapper(env)  # type: ignore
        else:
            raise ValueError("Wrapper must be an instance of Wrapper class.")
    return env  # type: ignore
