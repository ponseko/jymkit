from abc import abstractmethod
from dataclasses import replace

import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from jymkit import Environment


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
