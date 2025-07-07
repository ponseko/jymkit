import logging
from dataclasses import replace
from typing import Tuple

import distrax
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
    Transition,
    TransitionBuffer,
    scan_callback,
    split_key_over_agents,
    transform_multi_agent,
)

logger = logging.getLogger(__name__)


class Alpha(eqx.Module):
    ent_coef: jnp.ndarray

    def __init__(self, ent_coef_init: float = 0.0):
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


class SAC(RLAlgorithm):
    state: PyTree[SACState] = None
    optimizer: optax.GradientTransformation = eqx.field(static=True, default=None)

    learning_rate: float | optax.Schedule = eqx.field(static=True, default=2.5e-4)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    update_every: int = 256
    replay_buffer_size: int = 10000
    batch_size: int = 256
    alpha: float = 0.0  # is passed through an exponential function
    target_entropy_scale = 1.0
    tau: float = 0.95

    total_timesteps: int = eqx.field(static=True, default=int(1e6))
    num_envs: int = eqx.field(static=True, default=6)

    @property
    def num_iterations(self):
        return self.total_timesteps // self.update_every

    @property
    def num_steps(self):  # rollout length
        return self.update_every // self.num_envs

    def get_action(self, key: PRNGKeyArray, observation):
        @transform_multi_agent(multi_agent=self.multi_agent)
        def _get_action(agent: SACState, key: PRNGKeyArray, obs):
            action_dist = agent.actor(obs)
            return action_dist.sample(seed=key)

        structure = jax.tree.structure(
            self.state, is_leaf=lambda x: isinstance(x, SACState)
        )
        key = split_key_over_agents(key, structure)
        return _get_action(self.state, key, observation)

    def init(self, key: PRNGKeyArray, env: Environment, **hyperparams) -> "SAC":
        hyperparams["multi_agent"] = getattr(env, "_multi_agent", False)
        self = replace(self, **hyperparams)

        @transform_multi_agent(multi_agent=self.multi_agent)
        def _make_agent_state(
            key: PRNGKeyArray,
            obs_space: jym.Space,
            output_space: jym.Space,
            actor_features: list,
            critic_features: list,
            optimizer: optax.GradientTransformation,
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
            alpha = Alpha(self.alpha)

            optimizer_state = optimizer.init(
                eqx.filter((actor, critic1, critic2, alpha), eqx.is_inexact_array)
            )

            return SACState(
                actor=actor,
                critic1=critic1,
                critic2=critic2,
                critic1_target=critic1_target,
                critic2_target=critic2_target,
                alpha=alpha,
                optimizer_state=optimizer_state,
            )

        # TODO: can define multiple optimizers by using map
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adabelief(learning_rate=self.learning_rate),
        )

        keys_per_agent = split_key_over_agents(key, env.agent_structure)
        agent_states = _make_agent_state(
            output_space=env.action_space,
            key=keys_per_agent,
            actor_features=self.policy_kwargs.get("actor_features", [64, 64]),
            critic_features=self.policy_kwargs.get("critic_features", [64, 64]),
            obs_space=env.observation_space,
            optimizer=optimizer,
        )

        return replace(self, state=agent_states, optimizer=optimizer)

    def train(self, key: PRNGKeyArray, env: Environment, **hyperparams) -> "SAC":
        # Functions prepended with `_` are called within the `train_iteration` scan loop.

        env = self.__check_env__(env, vectorized=True)
        hyperparams["multi_agent"] = getattr(env, "_multi_agent", False)
        self = replace(self, **hyperparams)

        if not self.is_initialized:
            self = self.init(key, env)

        def _collect_rollout(
            train_state: SACState, rollout_state, length: int = self.num_steps
        ):
            def env_step(rollout_state, _):
                @transform_multi_agent(multi_agent=self.multi_agent)
                def get_train_action(key: PRNGKeyArray, agent: SACState, observation):
                    action_dist = jax.vmap(agent.actor)(observation)
                    return action_dist.sample(seed=key)

                env_state, last_obs, rng = rollout_state
                rng, sample_key, step_key = jax.random.split(rng, 3)

                # select an action
                sample_key = split_key_over_agents(sample_key, env.agent_structure)
                action = get_train_action(sample_key, train_state, last_obs)

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
                    info=info,
                    next_observation=info[ORIGINAL_OBSERVATION_KEY],
                )

                rollout_state = (env_state, obsv, rng)
                return rollout_state, transition

            # Do rollout
            rollout_state, trajectory_batch = jax.lax.scan(
                env_step, rollout_state, None, length
            )

            return rollout_state, trajectory_batch

        @transform_multi_agent(multi_agent=self.multi_agent)
        def _update_agent_state(
            key: PRNGKeyArray, current_state: SACState, batch: Transition
        ) -> Tuple[SACState, None]:
            def compute_inner_target(
                key,
                actor: ActorNetwork,
                critics: Tuple[QValueNetwork, QValueNetwork],
                obs,
            ):
                """
                Returns the target value for the SAC actor along with the action log probs.
                """
                critic1, critic2 = critics
                action_dist: distrax.Distribution = jax.vmap(actor)(obs)
                if isinstance(action_dist, distrax.Categorical):
                    # Handle discrete action space
                    action_probs = action_dist.probs
                    action_log_prob = jnp.log(action_probs + 1e-8)

                    q_1 = jax.vmap(critic1)(obs)
                    q_2 = jax.vmap(critic2)(obs)

                    return (
                        action_probs
                        * (
                            jnp.minimum(q_1, q_2)
                            - current_state.alpha() * action_log_prob
                        )
                    ).sum(axis=-1), (action_probs * action_log_prob)

                else:  # Continuous action space
                    action, action_log_prob = action_dist.sample_and_log_prob(seed=key)
                    q_1 = jax.vmap(critic1)(obs, action)
                    q_2 = jax.vmap(critic2)(obs, action)
                    return (
                        jnp.minimum(q_1, q_2) - current_state.alpha() * action_log_prob
                    ), (action_log_prob)

            @eqx.filter_grad
            def __sac_qnet_loss(params, batch: Transition):
                q_out = jax.vmap(params)(batch.observation, batch.action)
                if not q_out.squeeze().shape == batch.action.squeeze().shape:
                    # Q-net outputs a value per possible action (Discrete action space)
                    # Hence we get the q-values for the actions taken in the batch
                    q_out = q_out[jnp.arange(q_out.shape[0]), batch.action]
                q_loss = jnp.mean((q_out - target) ** 2)
                return q_loss

            @eqx.filter_grad(has_aux=True)
            def __sac_actor_loss(params, batch: Transition):
                target, log_probs = compute_inner_target(
                    key,
                    params,
                    (current_state.critic1, current_state.critic2),
                    batch.observation,
                )
                loss = -jnp.mean(target)

                return loss, (log_probs)

            @eqx.filter_grad
            def __sac_alpha_loss(params):
                def get_action_dim(space):
                    # TODO
                    if hasattr(space, "n") or (
                        (space, "shape") and len(space.shape) == 0
                    ):
                        return 2
                    elif hasattr(space, "shape"):
                        return jnp.prod(jnp.array(space.shape))
                    elif hasattr(space, "nvec"):
                        return jnp.array(space.nvec).size
                    else:
                        raise ValueError("Unsupported action space type")

                num_actions = get_action_dim(env.action_space)
                target_entropy = -(num_actions)
                # target_entropy = -(self.target_entropy_scale) * jnp.log(1 / num_actions)
                return -jnp.mean(
                    jnp.log(params()) * (action_log_probs + target_entropy)
                )

            target, _ = compute_inner_target(
                key,
                current_state.actor,
                (current_state.critic1_target, current_state.critic2_target),
                batch.next_observation,
            )
            target = batch.reward + ~batch.terminated * self.gamma * target

            def _zero_grads(model):
                return jax.tree.map(lambda x: jnp.zeros_like(x), model)

            critic1_grads = __sac_qnet_loss(current_state.critic1, batch)
            critic2_grads = __sac_qnet_loss(current_state.critic2, batch)
            # Should technically do the below code to update the critics first and use
            # the new critic parameters to compute the actor loss.
            # updates, _ = self.optimizer.update(
            #     (
            #         _zero_grads(current_state.actor),
            #         critic1_grads,
            #         critic2_grads,
            #         _zero_grads(current_state.alpha),
            #     ),
            #     current_state.optimizer_state,
            # )
            # _, new_critic1, new_critic2, _ = eqx.apply_updates(
            #     (
            #         current_state.actor,
            #         current_state.critic1,
            #         current_state.critic2,
            #         current_state.alpha,
            #     ),
            #     updates,
            # )

            actor_grads, (action_log_probs) = __sac_actor_loss(
                current_state.actor, batch
            )
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
                alpha=new_alpha,
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
            buffer: TransitionBuffer = runner_state[1]
            rollout_state = runner_state[2:]
            (env_state, last_obs, rng), trajectory_batch = _collect_rollout(
                train_state, rollout_state
            )

            # Add new data to buffer & Sample update batch from the buffer
            buffer = buffer.insert(trajectory_batch)
            train_data = buffer.sample(rng)

            # Update
            train_state, _ = _update_agent_state(
                rng, train_state, train_data.view_transposed
            )

            metric = trajectory_batch.info
            runner_state = (train_state, buffer, env_state, last_obs, rng)
            return runner_state, metric

        obsv, env_state = env.reset(jax.random.split(key, self.num_envs))

        # Set up the buffer
        _, dummy_trajectory = _collect_rollout(
            self.state, (env_state, obsv, key), self.batch_size
        )
        buffer = TransitionBuffer(
            max_size=self.replay_buffer_size,
            sample_batch_size=self.batch_size,
            data_sample=dummy_trajectory,
        )
        buffer = buffer.insert(dummy_trajectory)  # Add minimum data to buffer

        runner_state = (self.state, buffer, env_state, obsv, key)
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
