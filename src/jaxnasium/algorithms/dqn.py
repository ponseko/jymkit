import logging
from dataclasses import replace
from functools import partial
from typing import Any, Self

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import PRNGKeyArray, PyTree

import jaxnasium as jym
from jaxnasium import Environment
from jaxnasium._environment import ORIGINAL_OBSERVATION_KEY
from jaxnasium.algorithms import RLAgent, RLAlgorithm
from jaxnasium.algorithms.core import (
    Normalizer,
    Schedule,
    Transition,
    TransitionBuffer,
    scan_callback,
)

from .agent_networks import QValueNetwork

logger = logging.getLogger(__name__)


class DQNAgent(RLAgent):
    critic: QValueNetwork
    critic_target: QValueNetwork
    optimizer_state: optax.OptState
    normalizer: Normalizer

    def __init__(self, key, env: Environment, trainer: "DQN"):
        self.critic = QValueNetwork(
            env.observation_space,
            env.action_space,
            key=key,
            **trainer.critic_kwargs,
        )
        self.critic_target = jax.tree.map(lambda x: x, self.critic)

        self.optimizer_state = trainer.optimizer.init(
            eqx.filter(self.critic, eqx.is_inexact_array)
        )

        self.normalizer = Normalizer(
            obs_space=env.observation_space,
            normalize_obs=trainer.normalize_observations,
            normalize_rew=trainer.normalize_rewards,
            gamma=trainer.gamma,
            rew_shape=(trainer.rollout_length, trainer.num_envs),
        )

    def get_action(
        self,
        key: PRNGKeyArray,
        observation: PyTree,
        deterministic: bool = False,
        epsilon: float = 0.0,
    ):
        if deterministic:
            assert epsilon == 0.0, "Non-zero epsilon for deterministic action"
        observation = self.normalizer.normalize_obs(observation)
        q_values = self.critic(observation)
        action_dist = distrax.Joint(  # support pytrees of output distributions
            jax.tree.map(lambda x: distrax.EpsilonGreedy(x, epsilon=epsilon), q_values)
        )
        return action_dist.sample(seed=key)

    def get_value(self, observation: PyTree):
        observation = self.normalizer.normalize_obs(observation)
        return self.critic(observation)

    def normalize_observation(self, observations: PyTree):
        return self.normalizer.normalize_obs(observations)

    def normalize_reward(self, rewards: PyTree):
        return self.normalizer.normalize_reward(rewards)

    def update_normalizer(self, batch: Transition):
        updated_normalizer = self.normalizer.update(batch)
        return self.replace(normalizer=updated_normalizer)

    def update_params(self, batch: Transition, trainer: "DQN"):
        @eqx.filter_grad
        def __dqn_loss(params: QValueNetwork, train_batch: Transition):
            q_out_1 = jax.vmap(params)(train_batch.observation)
            q_taken = jym.tree.gather_actions(q_out_1, train_batch.action)
            q_taken = jym.tree.batch_sum(q_taken)
            q_loss = optax.huber_loss(q_taken, target)
            return jym.tree.mean(q_loss)

        # Compute target
        q_target_output = jax.vmap(self.critic_target)(batch.next_observation)
        q_target_output = jym.tree.batch_sum(
            jax.tree.map(lambda q: jnp.max(q, axis=-1), q_target_output)
        )
        target = batch.reward + ~batch.terminated * trainer.gamma * q_target_output

        grads = __dqn_loss(self.critic, batch)
        updates, optimizer_state = trainer.optimizer.update(grads, self.optimizer_state)
        new_critic = eqx.apply_updates(self.critic, updates)

        # update target policy
        new_critic_target = jax.tree.map(
            lambda x, y: (1 - trainer.tau) * x + trainer.tau * y,
            self.critic_target,
            new_critic,
        )

        return self.replace(
            critic=new_critic,
            critic_target=new_critic_target,
            optimizer_state=optimizer_state,
        )


class DQN(RLAlgorithm):
    """Deep Q-Network (DQN) algorithm implementation.

    This implementation uses target networks with soft updates (polyak averaging),
    a replay buffer and epsilon-greedy exploration with optional annealing.
    """

    agent: DQNAgent = eqx.field(default=None)
    "State of the DQN agent, containing the networks, optimizer state and optional normalization running statistics."

    learning_rate_start: float = 2.5e-3
    learning_rate_end: float | None = eqx.field(static=True, default=None)
    epsilon_start: float = 0.1
    epsilon_end: float | None = eqx.field(static=True, default=None)
    gamma: float = 0.99
    max_grad_norm: float = 1.0
    update_every: int = eqx.field(static=True, default=int(2e2))
    replay_buffer_size: int = int(1e4)
    batch_size: int = 64
    tau: float = 0.05

    total_timesteps: int = eqx.field(static=True, default=int(1e6))
    num_envs: int = eqx.field(static=True, default=8)

    normalize_observations: bool = eqx.field(static=True, default=True)
    normalize_rewards: bool = eqx.field(static=True, default=True)

    critic_kwargs: dict[str, Any] = eqx.field(static=True, default_factory=dict)

    @property
    def learning_rate_schedule(self) -> Schedule:
        return Schedule(
            self.learning_rate_start, self.learning_rate_end, self.num_training_updates
        )

    @property
    def epsilon_schedule(self) -> Schedule:
        return Schedule(self.epsilon_start, self.epsilon_end, self.num_training_updates)

    @property
    def optimizer(self):
        return optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adabelief(learning_rate=self.learning_rate_schedule),
        )

    @property
    def num_iterations(self):
        return int(self.total_timesteps // self.update_every)

    @property
    def rollout_length(self):
        return int(self.update_every // self.num_envs)

    @property
    def num_training_updates(self):
        return self.num_iterations  # * num_epochs

    @eqx.filter_jit
    def init_agent(self, key: PRNGKeyArray, env: Environment) -> Self:
        return replace(self, agent=DQNAgent(key=key, env=env, trainer=self))

    @eqx.filter_jit
    def train(self, key: PRNGKeyArray, env: Environment, **hyperparams) -> Self:
        env = self.__check_env__(env, vectorized=True)
        self = replace(self, **hyperparams)

        if not self.is_initialized:
            self = self.init_agent(key, env)

        obsv, env_state = env.reset(jax.random.split(key, self.num_envs))

        # Set up the buffer
        _, dummy_trajectory = self._collect_rollout(
            (env_state, obsv, key), env, length=self.batch_size // self.num_envs
        )
        buffer = TransitionBuffer(
            max_size=self.replay_buffer_size,
            sample_batch_size=self.batch_size,
            data_sample=dummy_trajectory,
        )
        buffer = buffer.insert(dummy_trajectory)  # Add minimum data to the buffer

        train_iteration_fn = partial(self.train_iteration, env=env)
        train_iteration_fn = scan_callback(
            func=train_iteration_fn,
            callback_fn=self.log_function,
            callback_interval=self.log_interval,
            n=self.num_iterations,
        )

        runner_state = (self, buffer, env_state, obsv, key)
        runner_state, metrics = jax.lax.scan(
            train_iteration_fn, runner_state, jnp.arange(self.num_iterations)
        )
        updated_self = runner_state[0]
        return updated_self

    @staticmethod
    def train_iteration(runner_state, train_iter, *, env: Environment):
        """
        Performs a single training iteration (A single `Collect data + Update` run).

        Typically, the method is wrapped in a partial `train_fn = partial(train_iteration, env=env)`,
        and scanned over until the total number of timesteps is reached.
        """
        # Do rollout of single trajactory
        self: DQN = runner_state[0]
        buffer: TransitionBuffer = runner_state[1]
        rollout_state = runner_state[2:]
        (env_state, last_obs, rng), trajectory_batch = self._collect_rollout(
            rollout_state, env
        )
        metric = trajectory_batch.info or {}

        # Update normalizer with new data from the trajectory
        agent: DQNAgent = self.agent.update_normalizer(trajectory_batch)

        # Add new data to buffer & Sample update batch from the buffer
        buffer = buffer.insert(trajectory_batch)
        train_batch = buffer.sample(rng)

        train_batch = train_batch.normalize(agent.normalizer)

        # Update
        updated_agent = agent.update_params(train_batch, self)
        self = replace(self, agent=updated_agent)

        runner_state = (self, buffer, env_state, last_obs, rng)
        return runner_state, metric

    def _collect_rollout(self, rollout_state, env: Environment, length=None):
        def env_step(rollout_state, _):
            env_state, last_obs, rng = rollout_state
            rng, sample_key, step_key = jax.random.split(rng, 3)

            # select an action
            sample_key = jax.random.split(sample_key, self.num_envs)
            update_count = jym.tree.get_first(self.agent, "count")
            get_action = partial(
                self.get_action, epsilon=self.epsilon_schedule(update_count)
            )
            action = jax.vmap(get_action, in_axes=(0, 0))(sample_key, last_obs)

            # take a step in the environment
            step_key = jax.random.split(step_key, self.num_envs)
            (obsv, reward, terminated, truncated, info), env_state = env.step(
                step_key, env_state, action
            )

            # Build a single transition. jax.lax.scan builds a batch of transitions.
            transition = Transition(
                observation=last_obs,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
                next_observation=info[ORIGINAL_OBSERVATION_KEY],
            )

            rollout_state = (env_state, obsv, rng)
            return rollout_state, transition

        if length is None:
            length = self.rollout_length

        # Do rollout
        rollout_state, trajectory_batch = jax.lax.scan(
            env_step, rollout_state, None, length
        )

        return rollout_state, trajectory_batch
