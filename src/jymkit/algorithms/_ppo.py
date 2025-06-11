import logging
from dataclasses import replace
from typing import Callable, Literal, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

import jymkit as jym
from jymkit import Environment, VecEnvWrapper, is_wrapped, remove_wrapper
from jymkit._environment import ORIGINAL_OBSERVATION_KEY
from jymkit.algorithms import ActorNetwork, CriticNetwork, RLAlgorithm
from jymkit.algorithms.utils import (
    Transition,
    scan_callback,
    split_key_over_agents,
    transform_multi_agent,
)

logger = logging.getLogger(__name__)


class PPOState(eqx.Module):
    actor: ActorNetwork
    critic: CriticNetwork
    optimizer_state: optax.OptState


class PPO(RLAlgorithm):
    state: PyTree[PPOState] = None
    optimizer: optax.GradientTransformation = eqx.field(static=True, default=None)
    multi_agent_env: bool = eqx.field(static=True, default=False)

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
    actor_features: list = eqx.field(static=True, default_factory=lambda: [64, 64])
    critic_features: list = eqx.field(static=True, default_factory=lambda: [64, 64])
    use_bronet: bool = eqx.field(static=True, default=False)

    log_function: Optional[Callable | Literal["simple", "tqdm"]] = eqx.field(
        static=True, default="simple"
    )
    log_interval: int | float = eqx.field(static=True, default=0.05)

    @property
    def minibatch_size(self):
        return self.num_envs * self.num_steps // self.num_minibatches

    @property
    def num_iterations(self):
        return self.total_timesteps // self.num_steps // self.num_envs

    @property
    def batch_size(self):
        return self.minibatch_size * self.num_minibatches

    def get_action(self, key: PRNGKeyArray, observation, get_log_prob=False):
        @transform_multi_agent(multi_agent=self.multi_agent_env)
        def _get_action(agent: PPOState, key: PRNGKeyArray, obs):
            action_dist = agent.actor(obs)
            if get_log_prob:
                return action_dist.sample_and_log_prob(seed=key)
            return action_dist.sample(seed=key)

        structure = jax.tree.structure(
            self.state, is_leaf=lambda x: isinstance(x, PPOState)
        )
        key = split_key_over_agents(key, structure)
        return _get_action(self.state, key, observation)

    def init(self, key: PRNGKeyArray, env: Environment, **hyperparams) -> "PPO":
        hyperparams["multi_agent_env"] = getattr(env, "_multi_agent", False)
        self = replace(self, **hyperparams)

        @transform_multi_agent(multi_agent=self.multi_agent_env)
        def _make_agent_state(
            key: PRNGKeyArray,
            obs_space: jym.Space,
            output_space: jym.Space,
            actor_features: list,
            critic_features: list,
            use_bronet: bool,
            optimizer: optax.GradientTransformation,
        ):
            actor_key, critic_key = jax.random.split(key)
            actor = ActorNetwork(
                key=actor_key,
                obs_space=obs_space,
                hidden_dims=actor_features,
                output_space=output_space,
                use_bronet=use_bronet,
            )
            critic = CriticNetwork(
                key=critic_key,
                obs_space=obs_space,
                hidden_dims=critic_features,
                use_bronet=use_bronet,
            )
            optimizer_state = optimizer.init(
                eqx.filter((actor, critic), eqx.is_inexact_array)
            )

            return PPOState(
                actor=actor,
                critic=critic,
                optimizer_state=optimizer_state,
            )

        # TODO: can define multiple optimizers by using map
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(
                learning_rate=self.learning_rate,
                eps=1e-5,
            ),
        )

        keys_per_agent = split_key_over_agents(key, env.agent_structure)
        agent_states = _make_agent_state(
            output_space=env.action_space,
            key=keys_per_agent,
            actor_features=self.actor_features,
            critic_features=self.critic_features,
            obs_space=env.observation_space,
            use_bronet=self.use_bronet,
            optimizer=optimizer,
        )

        return replace(self, state=agent_states, optimizer=optimizer)

    def train(self, key: PRNGKeyArray, env: Environment, **hyperparams) -> "PPO":
        # Functions prepended with `_` are called within the `train_iteration` scan loop.

        env = self.__check_env__(env, vectorized=True)
        hyperparams["multi_agent_env"] = getattr(env, "_multi_agent", False)
        self = replace(self, **hyperparams)

        if not self.is_initialized:
            self = self.init(key, env)

        def _collect_rollout(train_state: PPOState, rollout_state):
            def env_step(rollout_state, _):
                @transform_multi_agent(multi_agent=self.multi_agent_env)
                def get_value(agent: PPOState, observation):
                    return jax.vmap(agent.critic)(observation)

                @transform_multi_agent(multi_agent=self.multi_agent_env)
                def get_action_and_log_prob(agent: PPOState, key, observation):
                    action_dist = jax.vmap(agent.actor)(observation)
                    return action_dist.sample_and_log_prob(seed=key)

                env_state, last_obs, rng = rollout_state
                rng, sample_key, step_key = jax.random.split(rng, 3)

                # select an action
                sample_key = split_key_over_agents(sample_key, env.agent_structure)
                action, log_prob = get_action_and_log_prob(
                    train_state, sample_key, last_obs
                )

                # take a step in the environment
                step_key = jax.random.split(step_key, self.num_envs)
                (obsv, reward, terminated, truncated, info), env_state = env.step(
                    step_key, env_state, action
                )

                value = get_value(train_state, last_obs)
                try:  # Try to bootstrap correctly
                    next_value = get_value(train_state, info[ORIGINAL_OBSERVATION_KEY])
                except KeyError:
                    next_value = get_value(train_state, obsv)
                    done = jnp.logical_or(terminated, truncated)
                    next_value = jax.tree.map(
                        lambda nv, d: nv * (1 - d), next_value, done
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
                    log_prob=log_prob,
                    info=info,
                    value=value,
                    next_value=next_value,
                )

                rollout_state = (env_state, obsv, rng)
                return rollout_state, transition

            def compute_gae(gae, transition: Transition):
                assert transition.view_flat.value is not None
                assert transition.view_flat.next_value is not None

                value = transition.view_flat.value
                reward = transition.view_flat.reward
                next_value = transition.view_flat.next_value
                done = transition.view_flat.terminated

                if done.ndim < reward.ndim:
                    # correct for multi-agent envs that do not return done flags per agent
                    done = jnp.expand_dims(done, axis=-1)

                delta = reward + self.gamma * next_value * (1 - done) - value
                gae = delta + self.gamma * self.gae_lambda * (1 - done) * gae
                return gae, (gae, gae + value)

            # Do rollout
            rollout_state, trajectory_batch = jax.lax.scan(
                env_step, rollout_state, None, self.num_steps
            )

            # Calculate GAE & returns
            assert trajectory_batch.view_flat.value is not None
            _, (advantages, returns) = jax.lax.scan(
                compute_gae,
                jnp.zeros_like(trajectory_batch.view_flat.value[-1]),
                trajectory_batch,
                reverse=True,
                unroll=16,
            )

            # Return to multi-agent structure
            if self.multi_agent_env:
                advantages = jnp.moveaxis(advantages, -1, 0)
                returns = jnp.moveaxis(returns, -1, 0)
                advantages = jax.tree.unflatten(trajectory_batch.structure, advantages)
                returns = jax.tree.unflatten(trajectory_batch.structure, returns)

            trajectory_batch = replace(
                trajectory_batch,
                return_=returns,
                advantage_=advantages,
            )

            return rollout_state, trajectory_batch

        @transform_multi_agent(multi_agent=self.multi_agent_env)
        def _update_agent_state(
            current_state: PPOState, minibatch: Transition
        ) -> Tuple[PPOState, None]:
            @eqx.filter_grad
            def __ppo_los_fn(
                params: Tuple[ActorNetwork, CriticNetwork],
                train_batch: Transition,
            ):
                assert train_batch.advantage_ is not None
                assert train_batch.return_ is not None
                assert train_batch.log_prob is not None

                actor, critic = params
                action_dist = jax.vmap(actor)(train_batch.observation)
                log_prob = action_dist.log_prob(train_batch.action)
                entropy = action_dist.entropy().mean()
                value = jax.vmap(critic)(train_batch.observation)

                init_log_prob = train_batch.log_prob
                if log_prob.ndim == 2:  # MultiDiscrete Action Space
                    log_prob = jnp.sum(log_prob, axis=-1)
                    init_log_prob = jnp.sum(init_log_prob, axis=-1)

                # actor loss
                ratio = jnp.exp(log_prob - init_log_prob)
                _advantages = (
                    train_batch.advantage_ - train_batch.advantage_.mean()
                ) / (train_batch.advantage_.std() + 1e-8)
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
                value_losses_clipped = jnp.square(
                    value_pred_clipped - train_batch.return_
                )
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
            )
            return updated_state, None

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
            train_state = runner_state[0]
            rollout_state = runner_state[1:]
            (env_state, last_obs, rng), trajectory_batch = _collect_rollout(
                train_state, rollout_state
            )

            # Make train batch
            train_data = trajectory_batch.make_minibatches(
                rng, self.num_minibatches, self.num_epochs, n_batch_axis=2
            )

            # Update
            train_state, _ = jax.lax.scan(
                _update_agent_state, train_state, train_data.view_transposed
            )

            metric = trajectory_batch.info
            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        obsv, env_state = env.reset(jax.random.split(key, self.num_envs))
        runner_state = (self.state, env_state, obsv, key)
        runner_state, metrics = jax.lax.scan(
            train_iteration, runner_state, jnp.arange(self.num_iterations)
        )
        updated_state = runner_state[0]
        return replace(self, state=updated_state)

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

                action = self.get_action(action_key, obs)
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
