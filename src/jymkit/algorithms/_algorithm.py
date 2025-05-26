import logging
from abc import abstractmethod
from dataclasses import replace

import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from jymkit import Environment, JumanjiWrapper, is_wrapped

logger = logging.getLogger(__name__)


class RLAlgorithm(eqx.Module):
    state: eqx.AbstractVar[PyTree[eqx.Module]]

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

    def __check_env__(self, env: Environment):
        if is_wrapped(env, JumanjiWrapper):
            logger.warning(
                "Some Jumanji environments rely on specific action masking logic "
                "that may not be compatible with this algorithm. "
                "If this is the case, training will crash during compilation."
            )
