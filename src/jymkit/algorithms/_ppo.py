import logging
from dataclasses import replace
from functools import partial
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

import jymkit as jym
from jymkit import Environment, VecEnvWrapper, is_wrapped, remove_wrapper
from jymkit._environment import ORIGINAL_OBSERVATION_KEY
from jymkit.algorithms import ActorNetwork, RLAlgorithm, ValueNetwork
from jymkit.algorithms.utils import Normalizer, Transition, scan_callback

logger = logging.getLogger(__name__)


class PPOState(eqx.Module):
    actor: ActorNetwork
    critic: ValueNetwork
    optimizer_state: optax.OptState
    normalizer: Normalizer


class PPO(RLAlgorithm):
    state: PPOState = eqx.field(default=None)
    optimizer: optax.GradientTransformation = eqx.field(static=True, default=None)

    learning_rate: float | optax.Schedule = eqx.field(static=True, default=2.5e-4)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    clip_coef: float = 0.2
    clip_coef_vf: float = 10.0  # Depends on the reward scaling !
    ent_coef: float | optax.Schedule = eqx.field(static=True, default=0.01)
    vf_coef: float = 0.25

    total_timesteps: int = eqx.field(static=True, default=int(1e6))
    num_envs: int = eqx.field(static=True, default=6)
    num_steps: int = eqx.field(static=True, default=128)  # steps per environment
    num_minibatches: int = eqx.field(static=True, default=4)  # Number of mini-batches
    num_epochs: int = eqx.field(static=True, default=4)  # K epochs

    normalize_obs: bool = eqx.field(static=True, default=False)
    normalize_rewards: bool = eqx.field(static=True, default=True)

    @property
    def minibatch_size(self):
        return self.num_envs * self.num_steps // self.num_minibatches

    @property
    def num_iterations(self):
        return self.total_timesteps // self.num_steps // self.num_envs

    @property
    def batch_size(self):
        return self.minibatch_size * self.num_minibatches

    @staticmethod
    def get_action(
        key: PRNGKeyArray, state: PPOState, observation, *, get_log_prob: bool = False
    ) -> Array | Tuple[Array, Array]:
        observation = state.normalizer.normalize_obs(observation)
        action_dist = state.actor(observation)
        if get_log_prob:
            return action_dist.sample_and_log_prob(seed=key)  # type: ignore
        return action_dist.sample(seed=key)

    @staticmethod
    def get_value(state: PPOState, observation):
        observation = state.normalizer.normalize_obs(observation)
        return jax.vmap(state.critic)(observation)

    def init(self, key: PRNGKeyArray, env: Environment) -> "PPO":
        if getattr(env, "_multi_agent", False) and self.auto_upgrade_multi_agent:
            self = self.__make_multi_agent__(
                upgrade_func_names=[
                    "get_action",
                    "get_value",
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

            # Post-process the trajectory batch (GAE, returns, normalization)
            trajectory_batch, updated_state = self._postprocess_rollout(
                trajectory_batch.view_transposed, self.state
            )
            trajectory_batch = Transition.from_transposed(trajectory_batch)

            # Make train batch
            train_data = trajectory_batch.make_minibatches(
                rng, self.num_minibatches, self.num_epochs, n_batch_axis=2
            )

            # Update agent
            updated_state, _ = jax.lax.scan(
                lambda state, data: self._update_agent_state(state, data),
                updated_state,  # <-- Use updated state with updated normalizer
                train_data.view_transposed,
            )
            self = replace(self, state=updated_state)

            metric = trajectory_batch.info
            runner_state = (self, env_state, last_obs, rng)
            return runner_state, metric

        env = self.__check_env__(env, vectorized=True)
        self = replace(self, **hyperparams)

        if not self.is_initialized:
            self = self.init(key, env)

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
            get_action_and_log_prob = partial(self.get_action, get_log_prob=True)
            action, log_prob = jax.vmap(get_action_and_log_prob, in_axes=(0, None, 0))(
                sample_key, self.state, last_obs
            )

            # take a step in the environment
            step_key = jax.random.split(step_key, self.num_envs)
            (obsv, reward, terminated, truncated, info), env_state = env.step(
                step_key, env_state, action
            )

            value = self.get_value(self.state, last_obs)
            next_value = self.get_value(self.state, info[ORIGINAL_OBSERVATION_KEY])

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

    def _postprocess_rollout(
        self, trajectory_batch: Transition, current_state: PPOState
    ) -> Tuple[Transition, PPOState]:
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

            reward = current_state.normalizer.normalize_reward(batch.reward)
            done = batch.terminated
            if done.ndim < reward.ndim:
                # correct for multi-agent envs that do not return done flags per agent
                done = jnp.expand_dims(done, axis=-1)

            delta = reward + self.gamma * batch.next_value * (1 - done) - batch.value
            gae = delta + self.gamma * self.gae_lambda * (1 - done) * gae
            return gae, (gae, gae + batch.value)

        assert trajectory_batch.value is not None
        _, (advantages, returns) = jax.lax.scan(
            compute_gae_scan,
            optax.tree.zeros_like(trajectory_batch.value[-1]),
            trajectory_batch,
            reverse=True,
            unroll=16,
        )

        trajectory_batch = replace(
            trajectory_batch,
            advantage=advantages,
            return_=returns,
        )

        # Update normalization params
        updated_state = replace(
            current_state, normalizer=current_state.normalizer.update(trajectory_batch)
        )

        return trajectory_batch, updated_state

    def _update_agent_state(
        self, current_state: PPOState, minibatch: Transition
    ) -> Tuple[PPOState, None]:
        @eqx.filter_grad
        def __ppo_los_fn(
            params: Tuple[ActorNetwork, ValueNetwork],
            train_batch: Transition,
        ):
            assert train_batch.advantage is not None
            assert train_batch.return_ is not None
            assert train_batch.log_prob is not None

            def pytree_batch_sum(values):
                batch_wise_sums = jax.tree.map(
                    lambda x: jnp.sum(x, axis=tuple(range(1, x.ndim)))
                    if x.ndim > 1
                    else x,
                    values,
                )
                return jax.tree.reduce(lambda a, b: a + b, batch_wise_sums)

            actor, critic = params
            norm_obs = current_state.normalizer.normalize_obs(train_batch.observation)
            action_dist = jax.vmap(actor)(norm_obs)
            log_prob = action_dist.log_prob(train_batch.action)
            entropy = action_dist.entropy()
            value = jax.vmap(critic)(norm_obs)
            init_log_prob = train_batch.log_prob

            log_prob = pytree_batch_sum(log_prob)  # Assume independent actions
            init_log_prob = pytree_batch_sum(init_log_prob)
            entropy = pytree_batch_sum(entropy).mean()

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

            ent_coef = self.ent_coef
            if not isinstance(ent_coef, float):
                # ent_coef is a schedule # TODO
                ent_coef = ent_coef(  # pyright: ignore
                    current_state.optimizer_state[1][1].count  # type: ignore
                )

            # Total loss
            total_loss = actor_loss + self.vf_coef * value_loss - ent_coef * entropy
            return total_loss  # , (actor_loss, value_loss, entropy)

        actor, critic = current_state.actor, current_state.critic
        grads = __ppo_los_fn((actor, critic), minibatch)
        updates, optimizer_state = self.optimizer.update(
            grads, current_state.optimizer_state
        )
        new_actor, new_critic = eqx.apply_updates((actor, critic), updates)

        updated_state = PPOState(
            actor=new_actor,
            critic=new_critic,
            optimizer_state=optimizer_state,
            normalizer=current_state.normalizer,  # already updated
        )
        return updated_state, None

    def _make_agent_state(
        self,
        key: PRNGKeyArray,
        obs_space: jym.Space,
        output_space: jym.Space,
        actor_features: list,
        critic_features: list,
    ):
        actor_key, critic_key = jax.random.split(key)
        actor = ActorNetwork(
            key=actor_key,
            obs_space=obs_space,
            hidden_dims=actor_features,
            output_space=output_space,
        )
        critic = ValueNetwork(
            key=critic_key,
            obs_space=obs_space,
            hidden_dims=critic_features,
        )
        optimizer_state = self.optimizer.init(
            eqx.filter((actor, critic), eqx.is_inexact_array)
        )

        dummy_obs = jax.tree.map(
            lambda space: space.sample(jax.random.PRNGKey(0)),
            obs_space,
        )
        normalization_state = Normalizer(
            dummy_obs,
            normalize_obs=self.normalize_obs,
            normalize_rew=self.normalize_rewards,
            gamma=self.gamma,
            rew_shape=(self.num_steps, self.num_envs),
        )

        return PPOState(
            actor=actor,
            critic=critic,
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
