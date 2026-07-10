import logging
import warnings
from abc import abstractmethod
from dataclasses import dataclass, replace
from typing import Callable, Literal, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

import jaxnasium as jym
from jaxnasium import Environment, Space, VecEnvWrapper, is_wrapped, remove_wrapper

logger = logging.getLogger(__name__)


class RLAlgorithm(eqx.Module):
    """Base class for reinforcement learning algorithms in JAXnasium.

    This abstract base class provides a common interface for implementing RL algorithms
    in JAX. It supports both single-agent and multi-agent scenarios, with automatic
    multi-agent transformation capabilities.

    The philosophy of Jaxnasium algorithms is to maintain close to single-file implementations. All training
    logic is therefor implemented in the algorithms themselves, and are designed for single-agent settings.
    This base class provides only a default evaluation loop, environment compatibility checks, and
    automatic multi-agent transformation capabilities.

    Key Features:
    - Multi-agent support: Automatic transformation of single-agent algorithms to multi-agent
    - Environment compatibility: Built-in checks and warnings for environment compatibility
    - Evaluation: Standardized evaluation interface for comparing algorithm performance

    Attributes:
        agent_state: Abstract variable representing the algorithm's internal state (PyTree of modules)
        multi_agent: Whether this algorithm instance operates in multi-agent mode
        auto_upgrade_multi_agent: Whether to automatically upgrade single-agent methods to multi-agent
        log_function: Logging function to use ("simple", "tqdm", or custom callable)
        log_interval: Interval for logging (as fraction of total steps or absolute number)

    """

    agent: eqx.AbstractVar[PyTree[eqx.Module]]
    "Trainable state of the algorithm, usually containing the networks, optimizer state and optional normalization running statistics."

    multi_agent: bool = eqx.field(static=True, default=False)
    auto_upgrade_multi_agent: bool = eqx.field(static=True, default=True)
    log_function: Optional[Callable | Literal["simple", "tqdm"]] = eqx.field(
        static=True, default="simple"
    )
    log_interval: int | float = eqx.field(static=True, default=0.05)

    @property
    def is_initialized(self) -> bool:
        return self.agent is not None

    def save_state(self, file_path: str):
        with open(file_path, "wb") as f:
            eqx.tree_serialise_leaves(f, self.agent)

    def load_state(self, file_path: str) -> "RLAlgorithm":
        with open(file_path, "rb") as f:
            agent = eqx.tree_deserialise_leaves(f, self.agent)
        algorithm = replace(self, agent=agent)
        return algorithm

    def get_action(
        self,
        key: PRNGKeyArray,
        observation: PyTree,
        deterministic: bool = False,
        **kwargs,
    ):
        return self.agent.get_action(key, observation, deterministic, **kwargs)

    @abstractmethod
    def train(self, key: PRNGKeyArray, env: Environment) -> "RLAlgorithm":
        pass

    def evaluate(
        self, key: PRNGKeyArray, env: Environment, num_eval_episodes: int = 10
    ) -> Float[Array, " num_eval_episodes"]:
        assert self.is_initialized, (
            "Agent state is not initialized. Create one via e.g. train() or init_state()."
        )
        if is_wrapped(env, VecEnvWrapper):
            # Cannot vectorize because terminations may occur at different times
            # use jax.vmap(agent.evaluate) if you can ensure episodes are of equal length
            env = remove_wrapper(env, VecEnvWrapper)

        def eval_episode(key, _) -> Tuple[PRNGKeyArray, PyTree[float]]:
            def step_env(carry):
                episode_reward, rng, obs, env_state, done = carry
                rng, action_key, step_key = jax.random.split(rng, 3)

                action = self.get_action(action_key, obs, deterministic=True)
                (obs, reward, terminated, truncated, info), env_state = env.step(
                    step_key, env_state, action
                )
                done = jax.tree.map(jnp.logical_or, terminated, truncated)
                done = jnp.all(jnp.array(jax.tree.leaves(done)))
                episode_reward += jym.tree.mean(reward)
                return (episode_reward, rng, obs, env_state, done)

            key, reset_key = jax.random.split(key)
            obs, env_state = env.reset(reset_key)
            done = False
            episode_reward = 0.0

            episode_reward, key, obs, env_state, done = jax.lax.while_loop(
                lambda carry: jnp.logical_not(carry[-1]),
                step_env,
                (episode_reward, key, obs, env_state, done),
            )

            return key, episode_reward

        _, episode_rewards = jax.lax.scan(
            eval_episode, key, jnp.arange(num_eval_episodes)
        )

        return episode_rewards

    def __check_env__(self, env: Environment, vectorized: bool = False) -> Environment:
        """
        Some validation checks on the current environment and its compatibility with the current
        algorithm setup.
        Additionally wraps the environment in a `VecEnvWrapper` if it is not already wrapped
        and `vectorized` is True.
        """
        if env.multi_agent:

            def first_level_structure(tree):
                return jax.tree.structure(tree, is_leaf=lambda x: x is not tree)

            first_level_action_space = first_level_structure(env.action_space)
            first_level_observation_space = first_level_structure(env.observation_space)
            assert first_level_action_space == first_level_observation_space, (
                "Action space and observation space must have the same first-level structure for multi-agent environments."
            )
            dummy_action = env.sample_action(jax.random.PRNGKey(0))
            obs, state = env.reset(jax.random.PRNGKey(0))
            timestep, _ = env.step(jax.random.PRNGKey(0), state, dummy_action)
            first_level_obs = first_level_structure(timestep.observation)
            first_level_action = first_level_structure(dummy_action)
            first_level_reward = first_level_structure(timestep.reward)
            assert first_level_obs == first_level_reward == first_level_action, (
                "Observation, reward, terminated, and truncated must have the same first-level structure for multi-agent environments."
            )
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

        return env


class HackuinoxModule(type(eqx.Module)):
    # Temporary name.
    # Here, we override the regular __call__ of equinox modules to allow __new__ methods of
    # modules to return something other than an instance of the module class (e.g. a MultiAgentWrapper of multiple instances).
    # Normally, Equinox will bookkeep what classes should be created and are being created (presumably to keep track of what should be frozen).

    def __call__(cls, *args, **kwargs):
        # Subclasses with a __new_wrapped__ classmethod can use it to intercept the regular __call__
        # and return something other than an instance of the class (e.g. a MultiAgentWrapper of multiple instances).
        # The __new_wrapped__ method should make sure infinite recursion is avoided
        dispatch = getattr(cls, "__new_wrapped__", None)
        if dispatch is not None:
            dispatched_result = dispatch(*args, **kwargs)
            if dispatched_result:
                return dispatched_result
        return super().__call__(*args, **kwargs)


class RLAgent(eqx.Module, metaclass=HackuinoxModule):
    @abstractmethod
    def __init__(self, key: PRNGKeyArray, env: Environment, trainer: RLAlgorithm):
        pass

    def replace(self, **updates):
        keys, values = zip(*updates.items())
        return eqx.tree_at(lambda c: [c.__dict__[key] for key in keys], self, values)

    @classmethod
    def __new_wrapped__(cls, key: PRNGKeyArray, env: Environment, trainer: RLAlgorithm):
        @dataclass
        class SingleAgentEnvView:
            env: Environment
            action_space: Space
            observation_space: Space

            def __getattr__(self, name):
                return getattr(self.env, name)

        auto_upgrade_multi_agent = getattr(trainer, "auto_upgrade_multi_agent", False)
        if getattr(env, "multi_agent", False) and auto_upgrade_multi_agent:
            from ._core._multi_agent import (
                MultiAgentWrapper,
                map_multi_agent,
                to_per_agent,
            )

            # `map_multi_agent` infers the agent structure from the first non-key argument
            # As such, we create a per-agent environment (with each environment having the obs/action space of a single agent)
            agent_structure = jax.tree.structure(env.observation_space)
            obs_spaces = eqx.tree_flatten_one_level(env.observation_space)[0]
            action_spaces = eqx.tree_flatten_one_level(env.action_space)[0]
            envs = [
                SingleAgentEnvView(env, a, o) for a, o in zip(action_spaces, obs_spaces)
            ]
            envs = jax.tree.unflatten(agent_structure, envs)

            # Also create a per-agent trainer that may have per agent hyperparemeters
            trainer_args = {}
            for k, value in trainer.__dict__.items():
                if k == "auto_upgrade_multi_agent":
                    continue  # prevent infinite recursion
                trainer_args[k] = to_per_agent(value, agent_structure)
            trainers = [
                type(trainer)(
                    auto_upgrade_multi_agent=False,  # prevent infinite recursion
                    **{
                        key: eqx.tree_flatten_one_level(trainer_args[key])[0][i]
                        for key in trainer_args
                    },
                )
                for i in range(agent_structure.num_leaves)
            ]
            trainers = jax.tree.unflatten(agent_structure, trainers)

            return MultiAgentWrapper(
                map_multi_agent(lambda k, e, t: cls(k, e, t), key, envs, trainers)
            )
        return None  # continue with regular __call__
