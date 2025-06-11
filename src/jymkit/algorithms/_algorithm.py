import logging
from abc import abstractmethod
from dataclasses import replace

import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from jymkit import Environment, JumanjiWrapper, VecEnvWrapper, is_wrapped

logger = logging.getLogger(__name__)


class RLAlgorithm(eqx.Module):
    state: eqx.AbstractVar[PyTree[eqx.Module]]

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

    def __check_env__(self, env: Environment, vectorized: bool = False):
        if is_wrapped(env, JumanjiWrapper):
            logger.warning(
                "Some Jumanji environments rely on specific action masking logic "
                "that may not be compatible with this algorithm. "
                "If this is the case, training will crash during compilation."
            )
        if vectorized and not is_wrapped(env, VecEnvWrapper):
            logger.info("Wrapping environment in VecEnvWrapper")
            env = VecEnvWrapper(env)

        return env
