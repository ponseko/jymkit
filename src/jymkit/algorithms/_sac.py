import logging
from dataclasses import replace
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

import jymkit as jym
from jymkit import Environment, VecEnvWrapper, is_wrapped, remove_wrapper
from jymkit._environment import ORIGINAL_OBSERVATION_KEY
from jymkit.algorithms import ActorNetwork, QValueNetwork, RLAlgorithm
from jymkit.algorithms.utils import (
    Normalizer,
    Transition,
    TransitionBuffer,
    scan_callback,
)

logger = logging.getLogger(__name__)


class Alpha(eqx.Module):
    ent_coef: jnp.ndarray

    def __init__(self, ent_coef_init=jnp.log(0.2)):
        self.ent_coef = jnp.array(ent_coef_init)

    def __call__(self) -> jnp.ndarray:
        return jnp.exp(self.ent_coef)


class SACState(eqx.Module):
    actor: ActorNetwork
    critic1: QValueNetwork
    critic2: QValueNetwork
    critic1_target: QValueNetwork
    critic2_target: QValueNetwork
    alpha: Alpha
    optimizer_state: optax.OptState
    normalizer: Normalizer


class SAC(RLAlgorithm):
    state: PyTree[SACState] = None
    optimizer: optax.GradientTransformation = eqx.field(static=True, default=None)

    learning_rate: float | optax.Schedule = eqx.field(static=True, default=3e-4)
    gamma: float = 0.99
    max_grad_norm: float = 0.5
    update_every: int = eqx.field(static=True, default=int(10 * 6))
    replay_buffer_size: int = 1000  # int(512 * 36)
    batch_size: int = 128  # int(512 * 36)
    init_alpha: float = 0.2  # is passed through an exponential function
    learn_alpha: bool = eqx.field(static=True, default=True)
    target_entropy_scale = 0.5
    tau: float = 0.95

    total_timesteps: int = eqx.field(static=True, default=int(1e6))
    num_envs: int = eqx.field(static=True, default=6)

    normalize_obs: bool = eqx.field(static=True, default=True)
    normalize_rew: bool = eqx.field(static=True, default=True)

    @property
    def num_iterations(self):
        return int(self.total_timesteps // self.update_every)

    @property
    def num_steps(self):  # rollout length
        return int(self.update_every // self.num_envs)

    @staticmethod
    def get_action(key: PRNGKeyArray, state: SACState, observation) -> Array:
        observation = state.normalizer.normalize_obs(observation)
        action_dist = state.actor(observation)
        return action_dist.sample(seed=key)

    def init(self, key: PRNGKeyArray, env: Environment) -> "SAC":
        if getattr(env, "_multi_agent", False) and self.auto_upgrade_multi_agent:
            self = self.__make_multi_agent__(
                upgrade_func_names=[
                    "get_action",
                    "_update_agent_state",
                    "_make_agent_state",
                    "_postprocess_rollout",
                ]
            )

        if self.optimizer is None:
            self = replace(
                self,
                optimizer=optax.chain(
                    optax.clip_by_global_norm(self.max_grad_norm),
                    optax.adabelief(learning_rate=self.learning_rate),
                ),
            )

        agent_states = self._make_agent_state(
            key=key,
            obs_space=env.observation_space,
            output_space=env.action_space,
            actor_features=self.policy_kwargs.get("actor_features", [64, 64]),
            critic_features=self.policy_kwargs.get("critic_features", [64, 64]),
        )

        return replace(self, state=agent_states)

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

            # Post-process the trajectory batch: normalization update (possibly per-agent)
            updated_state = self._postprocess_rollout(
                trajectory_batch.view_transposed, self.state
            )

            # Add new data to buffer & Sample update batch from the buffer
            buffer = buffer.insert(trajectory_batch)
            train_data = buffer.sample(rng)

            # Update
            updated_state = self._update_agent_state(
                rng,
                updated_state,  # <-- use updated_state w/ updated norm
                train_data.view_transposed,
            )

            metric = trajectory_batch.info
            self = replace(self, state=updated_state)
            runner_state = (self, buffer, env_state, last_obs, rng)
            return runner_state, metric

        env = self.__check_env__(env, vectorized=True)
        self = replace(self, **hyperparams)

        if not self.is_initialized:
            self = self.init(key, env)

        obsv, env_state = env.reset(jax.random.split(key, self.num_envs))

        # Set up the buffer
        _, dummy_trajectory = self._collect_rollout((env_state, obsv, key), env)
        buffer = TransitionBuffer(
            max_size=self.replay_buffer_size,
            sample_batch_size=self.batch_size,
            data_sample=dummy_trajectory,
        )
        buffer = buffer.insert(dummy_trajectory)  # Add initial data to the buffer

        runner_state = (self, buffer, env_state, obsv, key)
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
            action = jax.vmap(self.get_action, in_axes=(0, None, 0))(
                sample_key, self.state, last_obs
            )

            # take a step in the environment
            step_key = jax.random.split(step_key, self.num_envs)
            (obsv, reward, terminated, truncated, info), env_state = env.step(
                step_key, env_state, action
            )

            # TODO: variable gamma from env
            # gamma = self.gamma
            # if "discount" in info:
            #     gamma = info["discount"]

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

        # Do rollout
        rollout_state, trajectory_batch = jax.lax.scan(
            env_step, rollout_state, None, self.num_steps
        )

        return rollout_state, trajectory_batch

    def _postprocess_rollout(
        self, trajectory_batch: Transition, current_state: SACState
    ) -> SACState:
        """
        1) Returns updated normalization based on the new trajectory batch.
        """
        # Update normalization params
        updated_state = replace(
            current_state, normalizer=current_state.normalizer.update(trajectory_batch)
        )

        return updated_state

    def _update_agent_state(
        self, key: PRNGKeyArray, current_state: SACState, batch: Transition
    ) -> SACState:
        def _compute_soft_target(action_dist, q_1, q_2):
            def soft_target(action_probs, q_1, q_2):
                action_log_prob = jnp.log(action_probs + 1e-8)
                target = (
                    action_probs
                    * (jnp.minimum(q_1, q_2) - current_state.alpha() * action_log_prob)
                ).sum(axis=-1)
                return target

            return jax.tree.map(soft_target, action_dist.probs, q_1, q_2)

        @eqx.filter_grad
        def __sac_qnet_loss(params, batch: Transition):
            q_out = jax.vmap(params)(batch.observation, batch.action)

            def get_q_from_actions(q, actions):
                """Get the Q value from the actions that were taken."""
                return jnp.take_along_axis(q, actions[..., None], axis=-1).squeeze()

            q_out = jax.tree.map(get_q_from_actions, q_out, batch.action)
            q_loss = jax.tree.map(lambda q, t: jnp.mean((q - t) ** 2), q_out, q_target)
            return jnp.array(jax.tree.leaves(q_loss)).mean()

        @eqx.filter_grad
        def __sac_actor_loss(params, batch: Transition):
            action_dist = jax.vmap(params)(batch.observation)
            q_1 = jax.vmap(current_state.critic1)(batch.observation)
            q_2 = jax.vmap(current_state.critic2)(batch.observation)
            target = _compute_soft_target(action_dist, q_1, q_2)
            # might still be lingering dimensions: flatten everything
            target = jax.tree.map(lambda x: jnp.reshape(x, (-1,)), target)
            return -jnp.mean(jnp.concatenate(jax.tree.leaves(target)))

        @eqx.filter_grad
        def __sac_alpha_loss(params: Alpha):
            def alpha_loss_per_action_dist(action_probs):
                action_dim = jnp.prod(jnp.array(action_probs.shape[1:]))
                target_entropy = -(self.target_entropy_scale) * jnp.log(1 / action_dim)
                action_log_probs = jnp.log(action_probs + 1e-8)
                return -jnp.mean(params() * (action_log_probs + target_entropy))

            action_dist = jax.vmap(current_state.actor)(batch.observation)
            loss = jax.tree.map(alpha_loss_per_action_dist, action_dist.probs)
            loss = jnp.mean(jnp.array(jax.tree.leaves(loss)))
            return loss

        rng, target_key, actor_loss_key = jax.random.split(key, 3)

        action_dist = jax.vmap(current_state.actor)(batch.next_observation)
        q_1 = jax.vmap(current_state.critic1_target)(batch.next_observation)
        q_2 = jax.vmap(current_state.critic2_target)(batch.next_observation)
        target = _compute_soft_target(action_dist, q_1, q_2)
        reward = current_state.normalizer.normalize_reward(batch.reward)

        def broadcast_to_match(x, target_ndim):
            """Add dimensions to x to match target_ndim."""
            return jnp.reshape(x, x.shape + (1,) * (target_ndim - x.ndim))

        q_target = jax.tree.map(
            lambda targ: broadcast_to_match(reward, targ.ndim)
            + (1.0 - broadcast_to_match(batch.terminated, targ.ndim))
            * self.gamma
            * targ,
            target,
        )

        critic1_grads = __sac_qnet_loss(current_state.critic1, batch)
        critic2_grads = __sac_qnet_loss(current_state.critic2, batch)
        actor_grads = __sac_actor_loss(current_state.actor, batch)
        alpha_grads = __sac_alpha_loss(current_state.alpha)

        updates, optimizer_state = self.optimizer.update(
            (actor_grads, critic1_grads, critic2_grads, alpha_grads),
            current_state.optimizer_state,
        )
        new_actor, new_critic1, new_critic2, new_alpha = eqx.apply_updates(
            (
                current_state.actor,
                current_state.critic1,
                current_state.critic2,
                current_state.alpha,
            ),
            updates,
        )

        # Update target networks
        new_critic1_target, new_critic2_target = jax.tree.map(
            lambda x, y: self.tau * x + (1 - self.tau) * y,
            (current_state.critic1_target, current_state.critic2_target),
            (new_critic1, new_critic2),
        )

        updated_state = SACState(
            actor=new_actor,
            critic1=new_critic1,
            critic2=new_critic2,
            critic1_target=new_critic1_target,
            critic2_target=new_critic2_target,
            optimizer_state=optimizer_state,
            alpha=new_alpha if self.learn_alpha else current_state.alpha,
            normalizer=current_state.normalizer,
        )
        return updated_state

    def _make_agent_state(
        self,
        key: PRNGKeyArray,
        obs_space: jym.Space,
        output_space: jym.Space,
        actor_features: list,
        critic_features: list,
    ):
        actor_key, critic1_key, critic2_key = jax.random.split(key, 3)
        actor = ActorNetwork(
            key=actor_key,
            obs_space=obs_space,
            hidden_dims=actor_features,
            output_space=output_space,
        )
        critic1 = QValueNetwork(
            key=critic1_key,
            obs_space=obs_space,
            hidden_dims=critic_features,
            output_space=output_space,
        )
        critic2 = QValueNetwork(
            key=critic2_key,
            obs_space=obs_space,
            hidden_dims=critic_features,
            output_space=output_space,
        )
        critic1_target = jax.tree.map(lambda x: x, critic1)
        critic2_target = jax.tree.map(lambda x: x, critic2)
        alpha = Alpha(jnp.log(self.init_alpha))

        optimizer_state = self.optimizer.init(
            eqx.filter((actor, critic1, critic2, alpha), eqx.is_inexact_array)
        )

        dummy_obs = jax.tree.map(
            lambda space: space.sample(jax.random.PRNGKey(0)),
            obs_space,
        )
        normalization_state = Normalizer(
            dummy_obs,
            normalize_obs=self.normalize_obs,
            normalize_rew=self.normalize_rew,
            gamma=self.gamma,
            rew_shape=(self.num_steps, self.num_envs),
        )

        return SACState(
            actor=actor,
            critic1=critic1,
            critic2=critic2,
            critic1_target=critic1_target,
            critic2_target=critic2_target,
            alpha=alpha,
            optimizer_state=optimizer_state,
            normalizer=normalization_state,
        )

    def evaluate(
        self, key: PRNGKeyArray, env: Environment, num_eval_episodes: int = 10
    ) -> Float[Array, " num_eval_episodes"]:
        assert self.is_initialized, (
            "Agent state is not initialized. Create one via e.g. train() or init()."
        )
        if is_wrapped(env, VecEnvWrapper):
            # Cannot vectorize because terminations may occur at different times
            # use jax.vmap(agent.evaluate) if you can ensure episodes are of equal length
            env = remove_wrapper(env, VecEnvWrapper)

        def eval_episode(key, _) -> Tuple[PRNGKeyArray, PyTree[float]]:
            def step_env(carry):
                rng, obs, env_state, done, episode_reward = carry
                rng, action_key, step_key = jax.random.split(rng, 3)

                action = self.get_action(action_key, self.state, obs)
                (obs, reward, terminated, truncated, info), env_state = env.step(
                    step_key, env_state, action
                )
                done = jnp.logical_or(terminated, truncated)
                episode_reward += jnp.mean(jnp.array(jax.tree.leaves(reward)))
                return (rng, obs, env_state, done, episode_reward)

            key, reset_key = jax.random.split(key)
            obs, env_state = env.reset(reset_key)
            done = False
            episode_reward = 0.0

            key, obs, env_state, done, episode_reward = jax.lax.while_loop(
                lambda carry: jnp.logical_not(carry[3]),
                step_env,
                (key, obs, env_state, done, episode_reward),
            )

            return key, episode_reward

        _, episode_rewards = jax.lax.scan(
            eval_episode, key, jnp.arange(num_eval_episodes)
        )

        return episode_rewards
