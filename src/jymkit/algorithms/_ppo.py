import json
from dataclasses import fields, replace
from functools import partial
from typing import Callable, Literal, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

import jymkit as jym
from jymkit import Environment, VecEnvWrapper, is_wrapped, remove_wrapper
from jymkit._environment import ORIGINAL_OBSERVATION_KEY
from jymkit.algorithms.utils import (
    ActorNetwork,
    CriticNetwork,
    Transition,
    map_each_agent,
    scan_callback,
    split_key_over_agents,
)


class AgentState(eqx.Module):
    actor: ActorNetwork
    critic: CriticNetwork
    optimizer_state: optax.OptState


class PPO(eqx.Module):
    state: PyTree[AgentState] = None
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
    update_epochs: int = eqx.field(static=True, default=4)  # K epochs

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

    @property
    def is_initialized(self):
        return self.state is not None

    def init(self, key: PRNGKeyArray, env: Environment, **hyperparams) -> "PPO":
        observation_space = env.observation_space
        action_space = env.action_space
        hyperparams["multi_agent_env"] = env.multi_agent
        self = replace(self, **hyperparams)

        @map_each_agent(
            shared_argnames=["actor_features", "critic_features", "optimizer"],
            identity=not self.multi_agent_env,
        )
        def create_agent_state(
            key: PRNGKeyArray,
            obs_space: jym.Space,
            output_space: int | jym.Space,
            actor_features: list,
            critic_features: list,
            optimizer: optax.GradientTransformation,
        ) -> AgentState:
            actor_key, critic_key = jax.random.split(key)

            actor = ActorNetwork(
                key=actor_key,
                obs_space=obs_space,
                hidden_dims=actor_features,
                output_space=output_space,
            )
            critic = CriticNetwork(
                key=critic_key, obs_space=obs_space, hidden_dims=critic_features
            )
            optimizer_state = optimizer.init(
                eqx.filter((actor, critic), eqx.is_inexact_array)
            )

            return AgentState(
                actor=actor,
                critic=critic,
                optimizer_state=optimizer_state,
            )

        env_agent_structure = env.agent_structure
        keys_per_agent = split_key_over_agents(key, env_agent_structure)

        # TODO: can define multiple optimizers by using map
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(
                learning_rate=self.learning_rate,
                eps=1e-5,
            ),
        )

        agent_states = create_agent_state(
            output_space=action_space,
            key=keys_per_agent,
            actor_features=[64, 64],
            critic_features=[64, 64],
            obs_space=observation_space,
            optimizer=optimizer,
        )

        agent = replace(
            self,
            state=agent_states,
            optimizer=optimizer,
        )
        return agent

    def save(self, file_path: str):
        with open(file_path, "wb") as f:
            non_hyperparams = ["state", "optimizer"]
            if not isinstance(self.learning_rate, float):
                non_hyperparams.append("learning_rate")  # TODO: save something
            if not isinstance(self.ent_coef, float):
                non_hyperparams.append("ent_coef")
            hyperparams = {}
            for field in fields(self):
                if field.name in non_hyperparams:
                    continue
                try:
                    hyperparams[field.name] = getattr(self, field.name).item()
                except AttributeError:
                    hyperparams[field.name] = getattr(self, field.name)
            hyperparam_str = json.dumps(hyperparams)
            f.write((hyperparam_str + "\n").encode())
            eqx.tree_serialise_leaves(f, self.state)

    @classmethod
    def load(cls, file_path: str, env: Environment) -> "PPO":
        agent = cls()
        with open(file_path, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            agent = agent.init(jax.random.PRNGKey(0), env, **hyperparams)
            state = eqx.tree_deserialise_leaves(f, agent.state)
        agent = replace(agent, state=state)
        return agent

    def get_value(self, observation: PyTree[Array]):
        value = map_each_agent(
            lambda a, o: a.critic(o),
            identity=not self.multi_agent_env,
        )(a=self.state, o=observation)
        return value

    def get_action(self, key, observation, get_log_prob=False):
        if self.multi_agent_env:
            _, structure = eqx.tree_flatten_one_level(observation)
            key = split_key_over_agents(key, structure)
        action_dist = map_each_agent(
            lambda a, o: a.actor(o), identity=not self.multi_agent_env
        )(a=self.state, o=observation)

        action = map_each_agent(
            lambda dist, seed: dist.sample(seed=seed),
            identity=not self.multi_agent_env,
            # shared_argnames=["seed"],  # NOTE: not sharing the seed is causing nans
        )(dist=action_dist, seed=key)

        if not get_log_prob:
            return action
        log_prob = map_each_agent(
            lambda dist, act: dist.log_prob(act), identity=not self.multi_agent_env
        )(
            dist=action_dist,
            act=action,
        )

        return action, log_prob

    def evaluate(
        self, key: PRNGKeyArray, env: Environment, num_eval_episodes: int = 100
    ) -> Float[Array, "..."]:
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

    def _collect_rollout(self, rollout_state: tuple, env: Environment):
        def env_step(rollout_state, _):
            env_state, last_obs, rng = rollout_state
            rng, sample_key, step_key = jax.random.split(rng, 3)

            # select an action
            sample_key = jax.random.split(sample_key, self.num_envs)
            get_action_and_log_prob = partial(self.get_action, get_log_prob=True)
            action, log_prob = jax.vmap(get_action_and_log_prob)(sample_key, last_obs)

            # take a step in the environment
            rng, key = jax.random.split(rng)
            step_key = jax.random.split(key, self.num_envs)
            (obsv, reward, terminated, truncated, info), env_state = env.step(
                step_key, env_state, action
            )

            value = jax.vmap(self.get_value)(last_obs)
            next_value = jax.vmap(self.get_value)(info[ORIGINAL_OBSERVATION_KEY])

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

    def train(self, key: PRNGKeyArray, env: Environment) -> "PPO":
        @scan_callback(
            callback_fn=self.log_function,
            callback_interval=self.log_interval,
            n=self.num_iterations,
        )
        def train_iteration(runner_state, _):
            def update_epoch(trajectory_batch: Transition, key: PRNGKeyArray) -> PPO:
                """Do one epoch of update"""

                @map_each_agent(identity=not self.multi_agent_env)
                def __update_state_over_minibatch(
                    current_state: AgentState, minibatch: Transition
                ):
                    @eqx.filter_grad
                    def __ppo_los_fn(
                        params: Tuple[ActorNetwork, CriticNetwork],
                        train_batch: Transition,
                    ):
                        assert train_batch.advantage_ is not None
                        assert train_batch.return_ is not None

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
                        value_loss = jnp.maximum(
                            value_losses, value_losses_clipped
                        ).mean()

                        ent_coef = self.ent_coef
                        if not isinstance(ent_coef, float):
                            # ent_coef is a schedule # TODO
                            ent_coef = ent_coef(  # pyright: ignore
                                current_state.optimizer_state[1][1].count  # type: ignore
                            )

                        # Total loss
                        total_loss = (
                            actor_loss + self.vf_coef * value_loss - ent_coef * entropy
                        )
                        return total_loss  # , (actor_loss, value_loss, entropy)

                    actor, critic = current_state.actor, current_state.critic
                    grads = __ppo_los_fn((actor, critic), minibatch)
                    updates, optimizer_state = self.optimizer.update(
                        grads, current_state.optimizer_state
                    )
                    new_actor, new_critic = eqx.apply_updates((actor, critic), updates)
                    updated_state = AgentState(
                        actor=new_actor,
                        critic=new_critic,
                        optimizer_state=optimizer_state,
                    )

                    return updated_state, None

                batch_idx = jax.random.permutation(key, self.batch_size)

                # reshape (flatten over num_steps and num_envs)
                batch = jax.tree.map(
                    lambda x: x.reshape((self.batch_size,) + x.shape[2:]),
                    trajectory_batch,
                )
                # take from the batch in a new order (the order of the randomized batch_idx)
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, batch_idx, axis=0), batch
                )
                # split in minibatches
                minibatches = jax.tree.map(
                    lambda x: x.reshape((self.num_minibatches, -1) + x.shape[1:]),
                    shuffled_batch,
                )
                # Update (each) agent
                updated_state, _ = jax.lax.scan(
                    __update_state_over_minibatch,
                    self.state,
                    minibatches.view_transposed,
                )
                return replace(self, state=updated_state)

            self: PPO = runner_state[0]
            # Do rollout of single trajactory
            rollout_state = runner_state[1:]
            (env_state, last_obs, rng), trajectory_batch = self._collect_rollout(
                rollout_state, env
            )

            epoch_keys = jax.random.split(rng, self.update_epochs)
            for i in range(self.update_epochs):
                self = update_epoch(trajectory_batch, epoch_keys[i])

            metric = trajectory_batch.info

            runner_state = (self, env_state, last_obs, rng)
            return runner_state, metric

        if not is_wrapped(env, VecEnvWrapper):
            print("Wrapping environment in VecEnvWrapper")
            env = VecEnvWrapper(env=env)

        if not self.is_initialized:
            self = self.init(key, env)

        obsv, env_state = env.reset(jax.random.split(key, self.num_envs))
        runner_state = (self, env_state, obsv, key)
        runner_state, metrics = jax.lax.scan(
            train_iteration, runner_state, jnp.arange(self.num_iterations)
        )
        updated_self = runner_state[0]
        return updated_self
