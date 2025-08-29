import copy
import inspect
import logging
import types
import warnings
from abc import abstractmethod
from dataclasses import replace
from typing import Any, Callable, List, Literal, Optional

import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from jymkit import Environment, FlattenActionSpaceWrapper, VecEnvWrapper, is_wrapped
from jymkit.algorithms.utils import transform_multi_agent

logger = logging.getLogger(__name__)

DEFAULT_PER_AGENT_FUNCTIONS = [
    "get_action",
    "get_value",
    "_update_agent_state",
    "_make_agent_state",
    "_postprocess_rollout",
]


class RLAlgorithm(eqx.Module):
    state: eqx.AbstractVar[PyTree[eqx.Module]]

    multi_agent: bool = eqx.field(static=True, default=False)
    auto_upgrade_multi_agent: bool = eqx.field(static=True, default=True)
    actor_kwargs: dict[str, Any] = eqx.field(static=True, default_factory=dict)
    critic_kwargs: dict[str, Any] = eqx.field(static=True, default_factory=dict)
    log_function: Optional[Callable | Literal["simple", "tqdm"]] = eqx.field(
        static=True, default="simple"
    )
    log_interval: int | float = eqx.field(static=True, default=0.05)

    @property
    def is_initialized(self) -> bool:
        return self.state is not None

    def save_state(self, file_path: str):
        with open(file_path, "wb") as f:
            eqx.tree_serialise_leaves(f, self.state)

    def load_state(self, file_path: str) -> "RLAlgorithm":
        with open(file_path, "rb") as f:
            state = eqx.tree_deserialise_leaves(f, self.state)
        agent = replace(self, state=state)
        return agent

    @abstractmethod
    def train(self, key: PRNGKeyArray, env: Environment) -> "RLAlgorithm":
        pass

    @abstractmethod
    def evaluate(
        self, key: PRNGKeyArray, env: Environment, num_eval_episodes: int = 10
    ) -> Float[Array, " num_eval_episodes"]:
        pass

    def __make_multi_agent__(
        self, *, upgrade_func_names: List[str] = DEFAULT_PER_AGENT_FUNCTIONS
    ):
        cls = self.__class__
        new_attrs: dict[str, object] = {}

        for name in upgrade_func_names:
            try:
                attr_obj = inspect.getattr_static(cls, name)
            except AttributeError:
                if (
                    name in DEFAULT_PER_AGENT_FUNCTIONS
                    and upgrade_func_names == DEFAULT_PER_AGENT_FUNCTIONS
                ):  # If algorithm is just using defaults, then we skip missing methods
                    # If upgrade_func_names is set, then we expect methods to be present.
                    continue
                raise AttributeError(f"Method {name!r} not found in {cls.__name__}. ")

            if isinstance(attr_obj, staticmethod):
                orig_fn: Callable = attr_obj.__func__
                new_attrs[name] = staticmethod(transform_multi_agent(orig_fn))

            elif callable(attr_obj) or callable(
                attr_obj.method
            ):  # instance or class method
                orig_fn: Callable = (
                    attr_obj if callable(attr_obj) else attr_obj.method
                )  # .method compatibility with older equinox versions
                new_attrs[name] = transform_multi_agent(orig_fn)

            else:
                raise TypeError(f"Attribute {name!r} is not a (static/class)method")

        NewCls = types.new_class(
            f"{cls.__name__}__MultiAgent", (cls,), {}, lambda ns: ns.update(new_attrs)
        )

        new_instance = copy.copy(self)  # keeps parameters unchanged
        new_instance = replace(new_instance, multi_agent=True)
        object.__setattr__(new_instance, "__class__", NewCls)  # safe: NewCls ⊂ cls
        return new_instance

    def __check_env__(
        self,
        env: Environment,
        vectorized: bool = False,
        flatten_action_space: bool = False,
    ) -> Environment:
        """
        Some validation checks on the current environment and its compatibility with the current
        algorithm setup.
        Additionally wraps the environment in a `VecEnvWrapper` if it is not already wrapped
        and `vectorized` is True.
        """
        if is_wrapped(env, "JumanjiWrapper"):
            logger.warning(
                "Some Jumanji environments rely on specific action masking logic "
                "that may not be compatible with this algorithm. "
                "If this is the case, training will crash during compilation."
            )
        if is_wrapped(env, "JaxMARLWrapper"):
            if getattr(env, "name", None) == "coin_game":
                logger.warning(
                    "Coin game is currently not supported due to an inconsistent API"
                )
        if is_wrapped(env, "NormalizeVecObsWrapper") and getattr(
            self, "normalize_obs", False
        ):
            warnings.warn(
                "Using both environment-side normalization (NormalizeVecObsWrapper) and algorithm-side normalization."
                "This likely leads to incorrect results. We recommend only using algorithm-side normalization, "
                "as it allows for easier checkpointing and resuming training."
            )
        if is_wrapped(env, "NormalizeVecRewardWrapper") and getattr(
            self, "normalize_reward", False
        ):
            warnings.warn(
                "Using both environment-side normalization (NormalizeVecRewardWrapper) and algorithm-side normalization."
                "This likely leads to incorrect results. We recommend only using algorithm-side normalization, "
                "as it allows for easier checkpointing and resuming training."
            )
        if vectorized and not is_wrapped(env, VecEnvWrapper):
            logger.info("Wrapping environment in VecEnvWrapper")
            env = VecEnvWrapper(env)

        if flatten_action_space and not is_wrapped(env, FlattenActionSpaceWrapper):
            logger.info("Wrapping environment in FlattenActionSpaceWrapper")
            env = FlattenActionSpaceWrapper(env)

        return env
