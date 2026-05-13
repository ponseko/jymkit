import logging
from dataclasses import replace
from functools import partial
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, PRNGKeyArray, PyTree

import jaxnasium as jym
from jaxnasium import Environment
from jaxnasium._environment import ORIGINAL_OBSERVATION_KEY
from jaxnasium.algorithms import RLAgent, RLAlgorithm
from jaxnasium.algorithms.utils import (
    # MultiAgentWrapper,
    Normalizer,
    Transition,
    scan_callback,
)

from .networks import ActorNetwork, ValueNetwork

logger = logging.getLogger(__name__)


class PPOAgent(RLAgent):
    actor: ActorNetwork
    critic: ValueNetwork
    optimizer_state: optax.OptState
    normalizer: Normalizer

    def __init__(self, key, env: Environment, trainer: "PPO"):
        actor_key, critic_key = jax.random.split(key)
        actor_kwargs = trainer.actor_kwargs
        critic_kwargs = trainer.critic_kwargs
        normalize_observations = trainer.normalize_observations
        normalize_rewards = trainer.normalize_rewards
        gamma = trainer.gamma
        num_steps = trainer.num_steps
        num_envs = trainer.num_envs
        optimizer = trainer.optimizer
        actor = ActorNetwork(
            key=actor_key,
            obs_space=env.observation_space,
            output_space=env.action_space,
            **actor_kwargs,
        )
        critic = ValueNetwork(
            key=critic_key,
            obs_space=env.observation_space,
            **critic_kwargs,
        )
        optimizer_state = optimizer.init(
            eqx.filter((actor, critic), eqx.is_inexact_array)
        )

        dummy_obs = jax.tree.map(
            lambda space: space.sample(jax.random.PRNGKey(0)), env.observation_space
        )
        normalization_state = Normalizer(
            dummy_obs,
            normalize_obs=normalize_observations,
            normalize_rew=normalize_rewards,
            gamma=gamma,
            rew_shape=(num_steps, num_envs),
        )

        self.actor = actor
        self.critic = critic
        self.optimizer_state = optimizer_state
        self.normalizer = normalization_state

    def get_action(
        self,
        key: PRNGKeyArray,
        observation: PyTree,
        deterministic: bool = False,
        get_log_prob: bool = False,
    ) -> Array | Tuple[Array, Array]:
        observation = self.normalizer.normalize_obs(observation)
        action_dist = self.actor(observation)
        if deterministic:
            assert not get_log_prob, "Cannot get log prob in deterministic mode"
            return action_dist.mode()
        if get_log_prob:
            return action_dist.sample_and_log_prob(seed=key)  # type: ignore
        return action_dist.sample(seed=key)

    def get_value(self, observation: PyTree):
        observation = self.normalizer.normalize_obs(observation)
        return self.critic(observation)

    def update_normalizer(self, trajectory_batch: Transition):
        updated_normalizer = self.normalizer.update(trajectory_batch)
        return self.replace(normalizer=updated_normalizer)

    def normalize_observation(self, observations: PyTree):
        return self.normalizer.normalize_obs(observations)

    def normalize_reward(self, rewards: PyTree):
        return self.normalizer.normalize_reward(rewards)

    def replace(self, **updates):
        keys, values = zip(*updates.items())
        return eqx.tree_at(lambda c: [c.__dict__[key] for key in keys], self, values)


class PPO(RLAlgorithm):
    """Proximal Policy Optimization (PPO) algorithm implementation."""

    agent: PPOAgent = eqx.field(default=None)

    learning_rate_start: float = 2.5e-3
    learning_rate_end: float | None = eqx.field(static=True, default=0.0)
    ent_coef_start: float = 0.01
    ent_coef_end: float | None = eqx.field(static=True, default=None)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    clip_coef: float = 0.2
    clip_coef_vf: float = 10.0
    vf_coef: float = 0.25

    total_timesteps: int = eqx.field(static=True, default=int(1e6))
    num_envs: int = eqx.field(static=True, default=6)
    num_steps: int = eqx.field(static=True, default=128)  # steps per environment
    num_minibatches: int = eqx.field(static=True, default=4)  # Number of mini-batches
    num_epochs: int = eqx.field(static=True, default=4)  # K epochs

    normalize_observations: bool = eqx.field(static=True, default=False)
    normalize_rewards: bool = eqx.field(static=True, default=True)

    @property
    def learning_rate(self) -> optax.Schedule:
        if self.learning_rate_end is not None:
            return optax.linear_schedule(
                init_value=self.learning_rate_start,
                end_value=self.learning_rate_end,
                transition_steps=self.num_training_updates,
            )
        return optax.constant_schedule(self.learning_rate_start)

    @property
    def ent_coef(self) -> optax.Schedule:
        if self.ent_coef_end is not None:
            return optax.linear_schedule(
                init_value=self.ent_coef_start,
                end_value=self.ent_coef_end,
                transition_steps=self.num_training_updates,
            )
        return optax.constant_schedule(self.ent_coef_start)

    @property
    def optimizer(self):
        return optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adabelief(learning_rate=self.learning_rate),
        )

    @property
    def minibatch_size(self):
        return self.num_envs * self.num_steps // self.num_minibatches

    @property
    def num_iterations(self):
        return self.total_timesteps // self.num_steps // self.num_envs

    @property
    def batch_size(self):
        return self.minibatch_size * self.num_minibatches

    @property
    def num_training_updates(self):
        return self.num_iterations * self.num_epochs

    def init_state(self, key: PRNGKeyArray, env: Environment) -> "PPO":
        return replace(self, agent=PPOAgent(key=key, env=env, trainer=self))

    def train(self, key: PRNGKeyArray, env: Environment, **hyperparams) -> "PPO":
        @scan_callback(
            callback_fn=self.log_function,
            callback_interval=self.log_interval,
            n=self.num_iterations,
        )
        def train_iteration(runner_state, _):
            """
            Performs a single training iteration (A single `Collect data + Update` run).
            This is repeated until the total number of timesteps is reached.
            """

            # Do rollout of single trajactory
            self: PPO = runner_state[0]
            rollout_state = runner_state[1:]
            (env_state, last_obs, rng), trajectory_batch = self._collect_rollout(
                rollout_state, env
            )
            metric = trajectory_batch.info or {}

            agent = self.agent.update_normalizer(trajectory_batch)

            # Post-process the trajectory batch (GAE, returns, normalization)
            trajectory_batch = self._postprocess_rollout(trajectory_batch)

            # Update agent
            updated_agent = self._update_agent_state(rng, agent, trajectory_batch)
            self = replace(self, agent=updated_agent)

            runner_state = (self, env_state, last_obs, rng)
            return runner_state, metric

        env = self.__check_env__(env, vectorized=True)
        self = replace(self, **hyperparams)

        if not self.is_initialized:
            self = self.init_state(key, env)

        obsv, env_state = env.reset(jax.random.split(key, self.num_envs))
        runner_state = (self, env_state, obsv, key)
        runner_state, metrics = jax.lax.scan(
            train_iteration, runner_state, jnp.arange(self.num_iterations)
        )
        updated_self = runner_state[0]
        return updated_self

    def _collect_rollout(self, rollout_state, env: Environment):
        def env_step(rollout_state, _):
            env_state, last_obs, rng = rollout_state
            rng, sample_key, step_key = jax.random.split(rng, 3)

            # select an action
            sample_key = jax.random.split(sample_key, self.num_envs)
            get_action_and_log_prob = partial(self.agent.get_action, get_log_prob=True)

            action, log_prob = jax.vmap(get_action_and_log_prob)(sample_key, last_obs)

            # take a step in the environment
            step_key = jax.random.split(step_key, self.num_envs)
            (obsv, reward, terminated, truncated, info), env_state = env.step(
                step_key, env_state, action
            )
            value = jax.vmap(self.agent.get_value)(last_obs)
            next_value = jax.vmap(self.agent.get_value)(info[ORIGINAL_OBSERVATION_KEY])

            # Build a single transition. Jax.lax.scan will build the batch
            # returning num_steps transitions.
            transition = Transition(
                observation=last_obs,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                log_prob=log_prob,
                info=info,
                value=value,
                next_value=next_value,
            )

            rollout_state = (env_state, obsv, rng)
            return rollout_state, transition

        # Do rollout
        rollout_state, trajectory_batch = jax.lax.scan(
            env_step, rollout_state, None, self.num_steps
        )

        return rollout_state, trajectory_batch

    def _postprocess_rollout(self, trajectory_batch: Transition) -> Transition:
        """
        1) Computes GAE and Returns and adds them to the trajectory batch.
        2) Returns updated normalization based on the new trajectory batch.
        """

        def compute_gae_scan(gae, batch: Transition):
            """
            Computes the Generalized Advantage Estimation (GAE) for the given batch of transitions.
            """

            assert batch.value is not None
            assert batch.next_value is not None

            done = batch.terminated
            if done.ndim < batch.reward.ndim:
                # correct for multi-agent envs that do not return done flags per agent
                done = jnp.expand_dims(done, axis=-1)

            delta = (
                batch.reward + self.gamma * batch.next_value * (1 - done) - batch.value
            )
            gae = delta + self.gamma * self.gae_lambda * (1 - done) * gae
            return gae, (gae, gae + batch.value)

        trajectory_batch = replace(
            trajectory_batch,
            reward=self.agent.normalize_reward(trajectory_batch.reward),
        )
        assert trajectory_batch.value is not None
        _, (advantages, returns) = trajectory_batch.scan(
            compute_gae_scan, jnp.zeros(self.num_envs), reverse=True, unroll=16
        )

        trajectory_batch = replace(
            trajectory_batch,
            advantage=advantages,
            return_=returns,
        )

        return trajectory_batch

    def _update_agent_state(
        self, key, current_agent: PPOAgent, train_data: Transition
    ) -> PPOAgent:
        @eqx.filter_grad
        def __ppo_los_fn(
            params: Tuple[ActorNetwork, ValueNetwork],
            train_batch: Transition,
        ):
            assert train_batch.advantage is not None
            assert train_batch.return_ is not None
            assert train_batch.log_prob is not None
            assert train_batch.value is not None

            actor, critic = params
            action_dist = jax.vmap(actor)(train_batch.observation)
            log_prob = action_dist.log_prob(train_batch.action)
            entropy = action_dist.entropy()
            value = jax.vmap(critic)(train_batch.observation)
            init_log_prob = train_batch.log_prob

            log_prob = jym.tree.batch_sum(log_prob)
            init_log_prob = jym.tree.batch_sum(init_log_prob)
            entropy = jym.tree.batch_sum(entropy)

            ratio = jnp.exp(log_prob - init_log_prob)
            _advantages = (train_batch.advantage - train_batch.advantage.mean()) / (
                train_batch.advantage.std() + 1e-8
            )
            actor_loss1 = _advantages * ratio

            actor_loss2 = (
                jnp.clip(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                * _advantages
            )
            actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()

            # critic loss
            value_pred_clipped = train_batch.value + (
                jnp.clip(
                    value - train_batch.value,
                    -self.clip_coef_vf,
                    self.clip_coef_vf,
                )
            )
            value_losses = jnp.square(value - train_batch.return_)
            value_losses_clipped = jnp.square(value_pred_clipped - train_batch.return_)
            value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()

            update_count = jym.tree.get_first(current_agent.optimizer_state, "count")
            ent_coef = self.ent_coef(update_count)

            # Total loss
            total_loss = (
                actor_loss + self.vf_coef * value_loss - ent_coef * entropy.mean()
            )
            return total_loss  # , (actor_loss, value_loss, entropy)

        def scan_minibatch_update(current_agent, minibatch):
            actor, critic = current_agent.actor, current_agent.critic
            grads = __ppo_los_fn((actor, critic), minibatch)
            updates, optimizer_state = self.optimizer.update(
                grads, current_agent.optimizer_state
            )
            new_actor, new_critic = eqx.apply_updates((actor, critic), updates)
            updated_agent = current_agent.replace(
                actor=new_actor, critic=new_critic, optimizer_state=optimizer_state
            )
            return updated_agent, None

        train_data = jax.tree.map(
            lambda x: x.reshape((self.batch_size,) + x.shape[2:]),
            train_data,
        )
        train_data = replace(
            train_data,
            observation=current_agent.normalize_observation(train_data.observation),
        )
        train_data = train_data.make_minibatches(
            key, self.num_minibatches, self.num_epochs
        )
        updated_agent, _ = train_data.scan(
            scan_minibatch_update, current_agent, unroll=16
        )
        return updated_agent
