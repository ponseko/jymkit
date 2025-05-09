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

GYMNAX_ENVS = [
    "Pendulum-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    # "Asterix-MinAtar",
    "Breakout-MinAtar",
    # "Freeway-MinAtar",
    "SpaceInvaders-MinAtar",
    "DeepSea-bsuite",
]

ALL_ENVS = JYMKIT_ENVS + GYMNAX_ENVS


def make(
    env_name: str,
    wrapper: Optional[Wrapper] = None,
    external_package: Optional[str] = None,
    **env_kwargs,
) -> Environment:
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
                "Gymnax is not installed. Please install it with `pip install gymnax`."
            )
        env, _ = gymnax.make(env_name, **env_kwargs)
        if wrapper is None:
            print(
                "Wrapping Gymnax environment with GymnaxWrapper\n",
                " Disable this behavior by passing wrapper=False",
            )
            env = GymnaxWrapper(env)
    else:
        matches = difflib.get_close_matches(env_name, ALL_ENVS, n=1, cutoff=0.6)
        suggestion = (
            f" Did you mean {matches[0]}?"
            if matches
            else " Available environments are:\n" + "\n".join(ALL_ENVS)
        )
        raise ValueError(f"Environment {env_name} not found.{suggestion}")

    if wrapper is not None:
        if isinstance(wrapper, Wrapper):
            env = wrapper(env)  # type: ignore
        else:
            raise ValueError("Wrapper must be an instance of Wrapper class.")
    return env  # type: ignore
