import logging
from dataclasses import replace
from typing import Any

import distrax
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
    Normalizer,
    Schedule,
    Transition,
    TransitionBuffer,
    scan_callback,
)

from .networks import ActorNetwork, QValueNetwork

logger = logging.getLogger(__name__)


@eqx.filter_vmap(in_axes=(eqx.if_array(0), None, None))
def ensambled_vmap(model, *x):
    return jax.vmap(model)(*x)


class Alpha(eqx.Module):
    ent_coef: jnp.ndarray

    def __init__(self, ent_coef_init=jnp.log(0.2)):
        self.ent_coef = jnp.array(ent_coef_init)

    def __call__(self) -> jnp.ndarray:
        return jnp.exp(self.ent_coef)


class SACAgent(RLAgent):
    actor: ActorNetwork
    critics: QValueNetwork
    critics_target: QValueNetwork
    alpha: Alpha
    optimizer_state: dict[str, optax.OptState]
    normalizer: Normalizer

    def __init__(self, key, env: Environment, trainer: "SAC"):
        actor_key, critics_key = jax.random.split(key, 2)
        self.actor = ActorNetwork(
            key=actor_key,
            obs_space=env.observation_space,
            output_space=env.action_space,
            **trainer.actor_kwargs,
        )
        ensamble_critics_keys = jax.random.split(critics_key, 2)  # 2 critics
        self.critics = jax.vmap(
            lambda key: QValueNetwork(
                key=key,
                obs_space=env.observation_space,
                output_space=env.action_space,
                **trainer.critic_kwargs,
            )
        )(ensamble_critics_keys)

        self.critics_target = jax.tree.map(lambda x: x, self.critics)
        self.alpha = Alpha(jnp.log(trainer.init_alpha))

        self.optimizer_state = jym.tree.map_one_level(
            lambda opt, params: opt.init(eqx.filter(params, eqx.is_inexact_array)),
            trainer.optimizer,
            {
                "actor": self.actor,
                "critics": self.critics,
                "alpha": self.alpha,
            },
        )

        self.normalizer = Normalizer(
            obs_space=env.observation_space,
            normalize_obs=trainer.normalize_observations,
            normalize_rew=trainer.normalize_rewards,
            gamma=trainer.gamma,
            rew_shape=(trainer.num_steps, trainer.num_envs),
        )

    def get_action(
        self, key: PRNGKeyArray, observation, deterministic: bool = False
    ) -> Array:
        observation = self.normalizer.normalize_obs(observation)
        action_dist = self.actor(observation)
        if deterministic:
            return action_dist.mode()
        return action_dist.sample(seed=key)

    def update_normalizer(self, batch: Transition):
        updated_normalizer = self.normalizer.update(batch)
        return self.replace(normalizer=updated_normalizer)

    def normalize_observation(self, observations: PyTree):
        return self.normalizer.normalize_obs(observations)

    def normalize_reward(self, rewards: PyTree):
        return self.normalizer.normalize_reward(rewards)

    def _compute_soft_target(self, action_dist, q, action_log_prob):
        if isinstance(action_dist, distrax.Categorical):
            action_log_prob = jnp.log(action_dist.probs + 1e-8)
        min_q = q.min(axis=0)
        target = min_q - self.alpha() * action_log_prob
        if isinstance(action_dist, distrax.Categorical):
            weighted_target = (action_dist.probs * target).sum(axis=-1)
            return weighted_target
        return target

    def update_actor_params(self, key, batch: Transition, trainer: "SAC"):
        @eqx.filter_grad
        def __sac_actor_loss(params, train_batch: Transition):
            action_dist = jax.vmap(params)(train_batch.observation)
            action, log_prob = action_dist.sample_and_log_prob(seed=key)
            q = ensambled_vmap(self.critics, train_batch.observation, action)
            target = jym.tree.map_distribution(
                self._compute_soft_target, action_dist, q, log_prob
            )
            target = jym.tree.batch_sum(target)
            return -jym.tree.mean(target)

        actor_grads = __sac_actor_loss(self.actor, batch)

        updates, optimizer_state = trainer.optimizer["actor"].update(
            actor_grads, self.optimizer_state["actor"]
        )
        new_actor = eqx.apply_updates(self.actor, updates)
        optimizer_state = {**self.optimizer_state, "actor": optimizer_state}
        return self.replace(actor=new_actor, optimizer_state=optimizer_state)

    def update_critics_params(self, key, batch: Transition, trainer: "SAC"):
        @eqx.filter_grad
        def __sac_qnet_loss(params, train_batch: Transition):
            q_out = jax.vmap(params)(train_batch.observation, train_batch.action)
            q_taken = jym.tree.gather_actions(q_out, train_batch.action)
            q_taken = jym.tree.batch_sum(q_taken)
            q_loss = optax.losses.huber_loss(q_taken, q_target)
            return jym.tree.mean(q_loss)

        action_dist = jax.vmap(self.actor)(batch.next_observation)
        action, log_prob = action_dist.sample_and_log_prob(seed=key)
        q = ensambled_vmap(self.critics_target, batch.next_observation, action)
        target = jym.tree.map_distribution(
            self._compute_soft_target, action_dist, q, log_prob
        )
        target = jym.tree.batch_sum(target)
        q_target = batch.reward + (1.0 - batch.terminated) * trainer.gamma * target
        grads = jax.vmap(__sac_qnet_loss, in_axes=(0, None))(self.critics, batch)
        updates, optimizer_state = trainer.optimizer["critics"].update(
            grads, self.optimizer_state["critics"]
        )
        new_critics = eqx.apply_updates(self.critics, updates)

        new_critics_target = jax.tree.map(
            lambda x, y: trainer.tau * x + (1 - trainer.tau) * y,
            self.critics_target,
            new_critics,
        )
        optimizer_state = {**self.optimizer_state, "critics": optimizer_state}
        return self.replace(
            critics=new_critics,
            critics_target=new_critics_target,
            optimizer_state=optimizer_state,
        )

    def update_alpha_params(self, key, batch: Transition, trainer: "SAC"):
        @eqx.filter_grad
        def __sac_alpha_loss(params: Alpha, train_batch: Transition):
            def _compute_alpha_signal(action_dist):
                target_entropy = trainer.target_entropy
                if isinstance(action_dist, distrax.Categorical):
                    action_probs = action_dist.probs
                    log_probs = jnp.log(action_probs + 1e-8)
                    if target_entropy is None:
                        action_dim = jnp.prod(jnp.array(log_probs.shape[1:]))
                        target_entropy = (
                            target_entropy_scale * 0.5 * jnp.log(action_dim)
                        )
                    return (action_probs * (log_probs + target_entropy)).sum(axis=-1)
                else:  # Continuous action space
                    _, log_probs = action_dist.sample_and_log_prob(seed=key)
                    if target_entropy is None:
                        action_dim = jnp.prod(jnp.array(train_batch.action.shape[1:]))
                        target_entropy = target_entropy_scale * -action_dim
                    return log_probs + target_entropy

            signals = jym.tree.map_distribution(_compute_alpha_signal, action_dist)
            signals = jym.tree.batch_sum(signals)
            return -jnp.mean(params() * signals)

        count = jym.tree.get_first(self.optimizer_state["alpha"], "count")
        target_entropy_scale = trainer.target_entropy_scale_schedule(count)
        action_dist = jax.vmap(self.actor)(batch.observation)
        alpha_grads = __sac_alpha_loss(self.alpha, batch)

        updates, optimizer_state = trainer.optimizer["alpha"].update(
            alpha_grads, self.optimizer_state["alpha"]
        )
        new_alpha = eqx.apply_updates(self.alpha, updates)
        optimizer_state = {**self.optimizer_state, "alpha": optimizer_state}
        return self.replace(alpha=new_alpha, optimizer_state=optimizer_state)


class SAC(RLAlgorithm):
    """Soft Actor-Critic (SAC) algorithm implementation.

    This implementation uses soft target updates, a replay buffer, and a target entropy scale with optional annealing.
    """

    agent: SACAgent = eqx.field(default=None)

    learning_rate_actor_start: float = 3e-3
    learning_rate_actor_end: float | None = eqx.field(static=True, default=None)
    learning_rate_critics_start: float = 3e-4
    learning_rate_critics_end: float | None = eqx.field(static=True, default=None)
    learning_rate_alpha_start: float = 3e-3
    learning_rate_alpha_end: float | None = eqx.field(static=True, default=None)

    @property
    def optimizer(self):
        def _create_optimizer(lr_schedule: Schedule):
            return optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.adabelief(learning_rate=lr_schedule),
            )

        actor_schedule = Schedule(
            self.learning_rate_actor_start,
            self.learning_rate_actor_end,
            self.num_training_updates_actor,
        )
        critics_schedule = Schedule(
            self.learning_rate_critics_start,
            self.learning_rate_critics_end,
            self.num_training_updates_critics,
        )
        alpha_schedule = Schedule(
            self.learning_rate_alpha_start,
            self.learning_rate_alpha_end,
            self.num_training_updates_actor,
        )
        return {
            "actor": _create_optimizer(actor_schedule),
            "critics": _create_optimizer(critics_schedule),
            "alpha": _create_optimizer(alpha_schedule),
        }

    gamma: float = 0.99
    max_grad_norm: float = 0.5
    num_steps: int = eqx.field(static=True, default=64)
    replay_buffer_size: int = 500_000
    batch_size: int = 512
    init_alpha: float = 0.2
    learn_alpha: bool = eqx.field(static=True, default=True)
    target_entropy: float | None = eqx.field(static=True, default=None)
    target_entropy_scale_start: float = 1.0
    target_entropy_scale_end: float | None = eqx.field(static=True, default=None)
    tau: float = 0.95

    actor_num_epochs: int = eqx.field(static=True, default=1)
    actor_num_minibatches: int = eqx.field(static=True, default=1)
    critics_num_epochs: int = eqx.field(static=True, default=8)
    critics_num_minibatches: int = eqx.field(static=True, default=1)
    alpha_num_epochs: int = eqx.field(static=True, default=1)
    alpha_num_minibatches: int = eqx.field(static=True, default=1)
    total_timesteps: int = eqx.field(static=True, default=int(1e6))
    num_envs: int = eqx.field(static=True, default=8)

    normalize_observations: bool = eqx.field(static=True, default=False)
    normalize_rewards: bool = eqx.field(static=True, default=False)
    actor_kwargs: dict[str, Any] = eqx.field(
        static=True, default_factory=lambda: {"continuous_output_dist": "tanhNormal"}
    )

    @property
    def target_entropy_scale_schedule(self):
        return Schedule(
            self.target_entropy_scale_start,
            self.target_entropy_scale_end,
            self.num_training_updates_alpha,
        )

    @property
    def num_iterations(self):
        return int(self.total_timesteps // self.num_steps // self.num_envs)

    @property
    def update_every(self):
        return int(self.num_steps * self.num_envs)

    @property
    def num_training_updates_actor(self):
        return self.num_iterations * self.actor_num_epochs * self.actor_num_minibatches

    @property
    def num_training_updates_critics(self):
        return (
            self.num_iterations * self.critics_num_epochs * self.critics_num_minibatches
        )

    @property
    def num_training_updates_alpha(self):
        return self.num_iterations * self.alpha_num_epochs * self.alpha_num_minibatches

    def init_agent(self, key: PRNGKeyArray, env: Environment) -> "SAC":
        return replace(self, agent=SACAgent(key=key, env=env, trainer=self))

    def train(self, key: PRNGKeyArray, env: Environment, **hyperparams) -> "SAC":
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
            self: SAC = runner_state[0]
            buffer: TransitionBuffer = runner_state[1]
            rollout_state = runner_state[2:]
            (env_state, last_obs, rng), trajectory_batch = self._collect_rollout(
                rollout_state, env
            )

            # Update normalizer
            agent = self.agent.update_normalizer(trajectory_batch)

            # Add new data to buffer & Sample update batch from the buffer
            buffer = buffer.insert(trajectory_batch)
            train_batch = buffer.sample(rng)

            # Update
            updated_agent = self._update_agent_state(rng, agent, train_batch)

            metric = trajectory_batch.info or {}
            self = replace(self, agent=updated_agent)
            runner_state = (self, buffer, env_state, last_obs, rng)
            return runner_state, metric

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

        runner_state = (self, buffer, env_state, obsv, key)
        runner_state, metrics = jax.lax.scan(
            train_iteration, runner_state, jnp.arange(self.num_iterations)
        )
        updated_self = runner_state[0]
        return updated_self

    def _collect_rollout(self, rollout_state, env: Environment, length=None):
        def env_step(rollout_state, _):
            env_state, last_obs, rng = rollout_state
            rng, sample_key, step_key = jax.random.split(rng, 3)

            # select an action
            sample_key = jax.random.split(sample_key, self.num_envs)
            action = jax.vmap(self.get_action, in_axes=(0, 0))(sample_key, last_obs)

            # take a step in the environment
            step_key = jax.random.split(step_key, self.num_envs)
            (obsv, reward, terminated, truncated, info), env_state = env.step(
                step_key, env_state, action
            )

            # Build a single transition. Jax.lax.scan will build the batch
            # returning num_steps transitions.
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
            length = self.num_steps

        # Do rollout
        rollout_state, trajectory_batch = jax.lax.scan(
            env_step, rollout_state, None, length
        )

        return rollout_state, trajectory_batch

    def _update_agent_state(
        self, key: PRNGKeyArray, current_state: SACAgent, train_batch: Transition
    ) -> SACAgent:
        # Normalize all used inputs, if normalization is disabled, these are no-ops
        normalizer = current_state.normalizer
        train_batch = replace(
            train_batch,
            observation=normalizer.normalize_obs(train_batch.observation),
            next_observation=normalizer.normalize_obs(train_batch.next_observation),
            reward=normalizer.normalize_reward(train_batch.reward),
        )

        def scan_critics_epoch_update(current_agent: SACAgent, key):
            minibatches = train_batch.make_minibatches(
                key, self.critics_num_minibatches
            )

            def do_update(key_and_current_agent, minibatch):
                key, current_agent = key_and_current_agent
                key, next_key = jax.random.split(key, 2)
                updated_agent = current_agent.update_critics_params(
                    key, minibatch, self
                )
                return (next_key, updated_agent), None

            (_, updated_agent), _ = jax.lax.scan(
                do_update, (key, current_agent), minibatches
            )
            return updated_agent, None

        update_keys = jax.random.split(key, self.critics_num_epochs)
        updated_agent, _ = jax.lax.scan(
            scan_critics_epoch_update, current_state, update_keys
        )

        def scan_actor_epoch_update(current_agent: SACAgent, key):
            minibatches = train_batch.make_minibatches(key, self.actor_num_minibatches)

            def do_update(key_and_current_agent, minibatch):
                key, current_agent = key_and_current_agent
                key, next_key = jax.random.split(key, 2)
                updated_agent = current_agent.update_actor_params(key, minibatch, self)
                return (next_key, updated_agent), None

            (_, updated_agent), _ = jax.lax.scan(
                do_update, (key, current_agent), minibatches
            )
            return updated_agent, None

        update_keys = jax.random.split(key, self.actor_num_epochs)
        updated_agent, _ = jax.lax.scan(
            scan_actor_epoch_update, updated_agent, update_keys
        )

        def scan_alpha_epoch_update(current_agent: SACAgent, key):
            minibatches = train_batch.make_minibatches(key, self.alpha_num_minibatches)

            def do_update(key_and_current_agent, minibatch):
                key, current_agent = key_and_current_agent
                key, next_key = jax.random.split(key, 2)
                updated_agent = current_agent.update_alpha_params(key, minibatch, self)
                return (next_key, updated_agent), None

            (_, updated_agent), _ = jax.lax.scan(
                do_update, (key, current_agent), minibatches
            )
            return updated_agent, None

        if self.learn_alpha:
            update_keys = jax.random.split(key, self.alpha_num_epochs)
            updated_agent, _ = jax.lax.scan(
                scan_alpha_epoch_update, updated_agent, update_keys
            )

        return updated_agent
