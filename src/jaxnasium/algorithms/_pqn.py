import logging
from dataclasses import replace
from functools import partial

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
from jaxnasium.algorithms.utils import (
    DistraxContainer,
    Normalizer,
    Schedule,
    Transition,
    scan_callback,
)

from .networks import QValueNetwork

logger = logging.getLogger(__name__)


class PQNAgent(RLAgent):
    critic: QValueNetwork
    optimizer_state: optax.OptState
    normalizer: Normalizer

    def __init__(self, key, env: Environment, trainer: "PQN"):
        self.critic = QValueNetwork(
            key=key,
            obs_space=env.observation_space,
            output_space=env.action_space,
            **trainer.critic_kwargs,
        )

        self.optimizer_state = trainer.optimizer.init(
            eqx.filter(self.critic, eqx.is_inexact_array)
        )

        self.normalizer = Normalizer(
            obs_space=env.observation_space,
            normalize_obs=trainer.normalize_observations,
            normalize_rew=trainer.normalize_rewards,
            gamma=trainer.gamma,
            rew_shape=(trainer.num_steps, trainer.num_envs),
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
        action_dist = DistraxContainer(
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

    def update_params(self, batch: Transition, trainer: "PQN"):
        @eqx.filter_grad
        def __dqn_loss(params: QValueNetwork, train_batch: Transition):
            q_out_1 = jax.vmap(params)(train_batch.observation)
            q_taken = jym.tree.gather_actions(q_out_1, train_batch.action)
            q_taken = jym.tree.batch_sum(q_taken)
            q_loss = optax.huber_loss(q_taken, train_batch.return_)
            return jym.tree.mean(q_loss)

        grads = __dqn_loss(self.critic, batch)
        updates, optimizer_state = trainer.optimizer.update(grads, self.optimizer_state)
        new_critic = eqx.apply_updates(self.critic, updates)

        return self.replace(critic=new_critic, optimizer_state=optimizer_state)


class PQN(RLAlgorithm):
    """Parallel Q-Network (PQN) algorithm implementation."""

    agent: PQNAgent = eqx.field(default=None)

    learning_rate_start: float = 2.5e-4
    learning_rate_end: float | None = eqx.field(static=True, default=0.0)
    epsilon_start: float = 0.1
    epsilon_end: float | None = eqx.field(static=True, default=None)
    gamma: float = 0.99
    max_grad_norm: float = 10.0
    q_lambda: float = 0.65

    total_timesteps: int = eqx.field(static=True, default=int(1e6))
    num_envs: int = eqx.field(static=True, default=12)
    num_steps: int = eqx.field(static=True, default=128)  # steps per environment
    num_minibatches: int = eqx.field(static=True, default=4)  # Number of mini-batches
    num_epochs: int = eqx.field(static=True, default=4)  # K epochs

    normalize_observations: bool = eqx.field(static=True, default=True)
    normalize_rewards: bool = eqx.field(static=True, default=False)

    @property
    def learning_rate_schedule(self):
        return Schedule(
            start=self.learning_rate_start,
            end=self.learning_rate_end,
            transition_steps=self.num_training_updates,
        )

    @property
    def epsilon_schedule(self):
        return Schedule(
            start=self.epsilon_start,
            end=self.epsilon_end,
            transition_steps=self.num_training_updates,
        )

    @property
    def optimizer(self):
        return optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adabelief(learning_rate=self.learning_rate_schedule),
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
        return self.num_iterations * self.num_epochs * self.num_minibatches

    def init_agent(self, key: PRNGKeyArray, env: Environment) -> "PQN":
        return replace(self, agent=PQNAgent(key=key, env=env, trainer=self))

    def train(self, key: PRNGKeyArray, env: Environment, **hyperparams) -> "PQN":
        @scan_callback(
            callback_fn=self.log_function,
            callback_interval=self.log_interval,
            n=self.num_iterations,
        )
        def train_iteration(runner_state, train_iter):
            """
            Performs a single training iteration (A single `Collect data + Update` run).
            This is repeated until the total number of timesteps is reached.
            """

            # Do rollout of single trajactory
            self: PQN = runner_state[0]
            rollout_state = runner_state[1:]
            (env_state, last_obs, rng), trajectory_batch = self._collect_rollout(
                rollout_state, env
            )
            metric = trajectory_batch.info or {}

            # Update normalizer with new data from the trajectory
            agent: PQNAgent = self.agent.update_normalizer(trajectory_batch)

            trajectory_batch = trajectory_batch.normalize(agent.normalizer)

            # Calculate Qlambda returns, add to trajectory batch
            _, returns = (
                trajectory_batch.scan(  # We can use a normal scan, but this custom scan automatically handles multi-agent scenarios
                    lambda re, transition: self._compute_q_lambda_scan(re, transition),
                    jnp.zeros(self.num_envs),
                    reverse=True,
                    unroll=16,
                )
            )
            trajectory_batch = replace(trajectory_batch, return_=returns)

            # Update agent
            updated_agent = self._update_agent_state(rng, agent, trajectory_batch)
            self = replace(self, agent=updated_agent)

            runner_state = (self, env_state, last_obs, rng)
            return runner_state, metric

        env = self.__check_env__(env, vectorized=True)
        self = replace(self, **hyperparams)

        if not self.is_initialized:
            self = self.init_agent(key, env)

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
            update_count = jym.tree.get_first(self.agent, "count")
            current_epsilon = self.epsilon_schedule(update_count)
            get_action = partial(self.get_action, epsilon=current_epsilon)
            action = jax.vmap(get_action)(sample_key, last_obs)

            # take a step in the environment
            step_key = jax.random.split(step_key, self.num_envs)
            (obsv, reward, terminated, truncated, info), env_state = env.step(
                step_key, env_state, action
            )

            next_q_value = jax.vmap(self.agent.get_value)(
                info[ORIGINAL_OBSERVATION_KEY]
            )

            # Build a single transition. jax.lax.scan builds a batch of transitions.
            transition = Transition(
                observation=last_obs,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                # next_observation=info[ORIGINAL_OBSERVATION_KEY],
                info=info,
                next_value=next_q_value,
            )

            rollout_state = (env_state, obsv, rng)
            return rollout_state, transition

        # Do rollout
        rollout_state, trajectory_batch = jax.lax.scan(
            env_step, rollout_state, None, self.num_steps
        )

        return rollout_state, trajectory_batch

    def _compute_q_lambda_scan(self, next_return, transition: Transition):
        next_q_values = jax.tree.map(
            lambda q: jnp.max(q, axis=-1), transition.next_value
        )
        next_q_values = jym.tree.batch_sum(next_q_values)

        done = transition.terminated
        return_this_step = transition.reward + (1 - done) * self.gamma * (
            self.q_lambda * next_return + (1 - self.q_lambda) * next_q_values
        )
        return return_this_step, return_this_step

    def _update_agent_state(
        self, key, current_agent: PQNAgent, trajectory_batch: Transition
    ) -> PQNAgent:
        def scan_epoch_update(current_agent: PQNAgent, key):
            minibatches = trajectory_batch.make_minibatches(
                key, self.num_minibatches, n_batch_axis=2
            )

            def do_update(current_state, minibatch):
                return current_state.update_params(minibatch, self), None

            updated_agent, _ = jax.lax.scan(do_update, current_agent, minibatches)
            return updated_agent, None

        update_keys = jax.random.split(key, self.num_epochs)
        updated_agent, _ = jax.lax.scan(
            scan_epoch_update, current_agent, update_keys, unroll=4
        )
        return updated_agent
