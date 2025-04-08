import inspect
import time
from dataclasses import replace
from functools import partial
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, PRNGKeyArray

import jymkit as jym
from jymkit._environment import ORIGINAL_OBSERVATION_KEY

from .networks import ActorNetwork, CriticNetwork


# Define a simple tuple to hold the state of the environment.
# This is the format we will use to store transitions in our buffer.
class Transition(NamedTuple):
    observation: Array
    action: Array
    reward: Array
    done: Array
    value: Array
    next_value: Array
    log_prob: Array
    info: Array
    return_: Array = None
    advantage_: Array = None


class AgentState(NamedTuple):
    actor: ActorNetwork
    critic: CriticNetwork
    optimizer_state: optax.OptState


class PPOAgent(eqx.Module):
    state: AgentState = None
    optimizer: optax.GradientTransformation = eqx.field(static=True, default=None)
    multi_agent_env: bool = eqx.field(static=True, default=False)

    learning_rate: float | optax.Schedule = eqx.field(static=True, default=2.5e-4)
    gamma: float = eqx.field(static=True, default=0.99)
    gae_lambda: float = eqx.field(static=True, default=0.95)
    max_grad_norm: float = eqx.field(static=True, default=0.5)
    clip_coef: float = eqx.field(static=True, default=0.2)
    clip_coef_vf: float = eqx.field(
        static=True, default=10.0
    )  # Depends on the reward scaling !
    ent_coef: float = eqx.field(static=True, default=0.01)
    vf_coef: float = eqx.field(static=True, default=0.25)

    total_timesteps: int = eqx.field(static=True, default=1e6)
    num_envs: int = eqx.field(static=True, default=6)
    num_steps: int = eqx.field(static=True, default=128)  # steps per environment
    num_minibatches: int = eqx.field(static=True, default=4)  # Number of mini-batches
    update_epochs: int = eqx.field(
        static=True, default=4
    )  # K epochs to update the policy

    debug: bool = eqx.field(static=True, default=True)

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

    def init(self, key: PRNGKeyArray, env: jym.Environment) -> "PPOAgent":
        observation_space = env.observation_space
        action_space = env.action_space
        self = replace(self, multi_agent_env=env.multi_agent)

        env_agent_structure = jax.tree.structure(observation_space)
        keys_per_agent = jax.tree.unflatten(
            env_agent_structure,
            list(jax.random.split(key, len(jax.tree.leaves(observation_space)))),
        )

        # TODO: can define multiple optimizers by using map
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(
                learning_rate=self.learning_rate,
                eps=1e-5,
            ),
        )

        agent_states = self.do_for_each_agent(
            self.create_agent_state,
            output_space=action_space,
            key=keys_per_agent,
            actor_features=[64, 64],
            critic_features=[64, 64],
            obs_space=observation_space,
            optimizer=optimizer,
            shared_argnames=["actor_features", "critic_features", "optimizer"],
        )

        agent = replace(
            self,
            state=agent_states,
            optimizer=optimizer,
        )
        return agent

    def create_agent_state(
        self,
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
            key=critic_key,
            obs_space=obs_space,
            hidden_dims=critic_features,
            output_space=1,
        )
        optimizer_state = optimizer.init({"actor": actor, "critic": critic})
        return AgentState(
            actor=actor,
            critic=critic,
            optimizer_state=optimizer_state,
        )

    def forward_critic(self, observation):
        value = self.do_for_each_agent(
            lambda a, o: jax.vmap(a.critic)(o),
            a=self.state,
            o=observation,
        )
        return value

    def forward_actor(self, observation, key, training=False):
        # TODO
        if not training:
            action_dist = self.do_for_each_agent(
                lambda a, o: a.actor(o),
                a=self.state,
                o=observation,
            )
        else:
            action_dist = self.do_for_each_agent(
                lambda a, o: jax.vmap(a.actor)(o),
                a=self.state,
                o=observation,
            )
        action = self.do_for_each_agent(
            lambda a, key: a.sample(seed=key),
            a=action_dist,
            key=key,
            shared_argnames=["key"],  # TODO: tmp
        )
        if not training:
            return action
        log_prob = self.do_for_each_agent(
            lambda a, act: a.log_prob(act),
            a=action_dist,
            act=action,
        )
        return action, log_prob

    def evaluate(self, key: PRNGKeyArray, env: jym.Environment):
        def step_env(carry):
            rng, obs, env_state, done, episode_reward = carry
            rng, action_key, step_key = jax.random.split(rng, 3)

            action = self.forward_actor(obs, action_key)

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

        return episode_reward

    def _collect_rollout(self, rollout_state: tuple, env: jym.Environment):
        def env_step(rollout_state, _):
            env_state, last_obs, rng = rollout_state
            rng, sample_key, step_key = jax.random.split(rng, 3)

            # select an action
            action, log_prob = self.forward_actor(last_obs, sample_key, training=True)
            value = self.forward_critic(last_obs)

            # take a step in the environment
            rng, key = jax.random.split(rng)
            step_key = jax.random.split(key, self.num_envs)
            (obsv, reward, terminated, truncated, info), env_state = jax.vmap(
                env.step, in_axes=(0, 0, 0)
            )(step_key, env_state, action)

            # get value of next observation before autoreset
            next_value = self.forward_critic(info[ORIGINAL_OBSERVATION_KEY])

            # gamma = self.gamma
            # if "discount" in info:
            #     gamma = info["discount"]

            # Build a single transition. Jax.lax.scan will build the batch
            # returning num_steps transitions.
            transition = Transition(
                observation=last_obs,
                action=action,
                reward=reward,
                done=terminated,
                value=value,
                next_value=next_value,
                log_prob=log_prob,
                info=info,
            )

            rollout_state = (env_state, obsv, rng)
            return rollout_state, transition

        def compute_gae(gae, transition):
            value = jnp.stack(jax.tree.leaves(transition.value), axis=-1).squeeze()
            reward = jnp.stack(jax.tree.leaves(transition.reward), axis=-1).squeeze()
            done = jnp.stack(jax.tree.leaves(transition.done), axis=-1).squeeze()
            next_value = jnp.stack(
                jax.tree.leaves(transition.next_value), axis=-1
            ).squeeze()

            if len(done.shape) == 1 and len(reward.shape) == 2:
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
            jnp.zeros_like(trajectory_batch.value[-1]),
            trajectory_batch,
            reverse=True,
            unroll=16,
        )

        trajectory_batch = trajectory_batch._replace(
            return_=returns,
            advantage_=advantages,
        )

        return rollout_state, trajectory_batch

    def train(self, key: PRNGKeyArray, env: jym.Environment) -> "PPOAgent":
        def train_iteration(runner_state, _):
            def update_epoch(
                trajectory_batch: Transition, key: PRNGKeyArray
            ) -> PPOAgent:
                """Do one epoch of update"""

                @eqx.filter_grad
                def __ppo_los_fn(
                    params,
                    observations,
                    actions,
                    log_probs,
                    values,
                    advantages,
                    returns,
                ):
                    action_dist = jax.vmap(params["actor"])(observations)
                    log_prob = action_dist.log_prob(actions)
                    entropy = action_dist.entropy().mean()
                    value = jax.vmap(params["critic"])(observations)

                    if len(log_prob.shape) == 2:  # MultiDiscrete Action Space
                        log_prob = jnp.sum(log_prob, axis=-1)
                        log_probs = jnp.sum(log_probs, axis=-1)

                    # actor loss
                    ratio = jnp.exp(log_prob - log_probs)
                    _advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )
                    actor_loss1 = _advantages * ratio

                    actor_loss2 = (
                        jnp.clip(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                        * _advantages
                    )
                    actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()

                    value_pred_clipped = values + (
                        jnp.clip(
                            value - values,
                            -self.clip_coef_vf,
                            self.clip_coef_vf,
                        )
                    )
                    value_losses = jnp.square(value - returns)
                    value_losses_clipped = jnp.square(value_pred_clipped - returns)
                    value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()

                    # Total loss
                    total_loss = (
                        actor_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                    )
                    return total_loss  # , (actor_loss, value_loss, entropy)

                def __update_over_minibatch(train_state: AgentState, minibatch):
                    observations, actions, log_probs, values, advantages, returns = (
                        minibatch.observation,
                        minibatch.action,
                        minibatch.log_prob,
                        minibatch.value,
                        minibatch.advantage_,
                        minibatch.return_,
                    )
                    train_state = train_state.state

                    params = jax.tree.map(
                        lambda x: {"actor": x.actor, "critic": x.critic},
                        train_state,
                        is_leaf=lambda x: isinstance(x, AgentState),
                    )
                    if self.multi_agent_env:
                        structure = jax.tree.structure(
                            params, is_leaf=lambda x: x is not params
                        )
                        # in case of multiagent, advantages and returns are (mb_size, num_agents)
                        # and we need to transpose them to (num_agents, mb_size)
                        # in single agent case, they are (mb_size,), in which case transpose is a no-op
                        advantages = list(jnp.transpose(advantages))
                        returns = list(jnp.transpose(returns))
                        advantages = jax.tree.unflatten(structure, advantages)
                        returns = jax.tree.unflatten(structure, returns)

                    grads = self.do_for_each_agent(
                        lambda params, obs, act, logp, val, adv, ret: __ppo_los_fn(
                            params, obs, act, logp, val, adv, ret
                        ),
                        params=params,
                        obs=observations,
                        act=actions,
                        logp=log_probs,
                        val=values,
                        adv=advantages,
                        ret=returns,
                    )
                    opt_states = jax.tree.map(
                        lambda x: x.optimizer_state,
                        train_state,
                        is_leaf=lambda x: isinstance(x, AgentState),
                    )

                    updates_and_optstate = self.do_for_each_agent(
                        lambda u, s: self.optimizer.update(u, s),
                        u=grads,
                        s=opt_states,
                    )
                    updates = self.do_for_each_agent(
                        lambda x: x[0],
                        x=updates_and_optstate,
                    )
                    optimizer_state = self.do_for_each_agent(
                        lambda x: x[1],
                        x=updates_and_optstate,
                    )

                    new_networks = self.do_for_each_agent(
                        lambda params, updates: optax.apply_updates(params, updates),
                        params=params,
                        updates=updates,
                    )

                    train_state = self.do_for_each_agent(
                        lambda networks, opt_state: AgentState(
                            actor=networks["actor"],
                            critic=networks["critic"],
                            optimizer_state=opt_state,
                        ),
                        networks=new_networks,
                        opt_state=optimizer_state,
                    )
                    return replace(self, state=train_state), None

                batch_idx = jax.random.permutation(key, self.batch_size)
                batch = trajectory_batch

                # reshape (flatten over first dimension)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((self.batch_size,) + x.shape[2:]), batch
                )
                # take from the batch in a new order (the order of the randomized batch_idx)
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, batch_idx, axis=0), batch
                )
                # split in minibatches
                minibatches = jax.tree_util.tree_map(
                    lambda x: x.reshape((self.num_minibatches, -1) + x.shape[1:]),
                    shuffled_batch,
                )
                # update over minibatches
                updated_self, _ = jax.lax.scan(
                    __update_over_minibatch, self, minibatches
                )
                return updated_self

            self: PPOAgent = runner_state[0]
            # Do rollout of single trajactory
            rollout_state = runner_state[1:]
            (env_state, last_obs, rng), trajectory_batch = self._collect_rollout(
                rollout_state, env
            )

            epoch_keys = jax.random.split(rng, self.update_epochs)
            for i in range(self.update_epochs):
                self = update_epoch(trajectory_batch, epoch_keys[i])

            metric = trajectory_batch.info
            rng, eval_key = jax.random.split(rng)
            eval_rewards = self.evaluate(eval_key, env)
            metric["eval_rewards"] = eval_rewards

            if self.debug:

                def callback(info):
                    print(
                        f"timestep={(info['timestep'][-1][0] * self.num_envs)}, eval rewards={info['eval_rewards']}"
                    )

                jax.debug.callback(callback, metric)

            runner_state = (self, env_state, last_obs, rng)
            return runner_state, metric

        if not self.is_initialized:
            self = self.init(key, env)

        env = jym.LogWrapper(env)

        def train_fn():
            # We wrap this logic so we can compile ahead of time
            obsv, env_state = jax.vmap(env.reset)(jax.random.split(key, self.num_envs))
            runner_state = (self, env_state, obsv, key)
            runner_state, metrics = jax.lax.scan(
                train_iteration, runner_state, None, self.num_iterations
            )
            return runner_state[0]

        s_time = time.time()
        print("Starting JAX compilation...")
        train_fn = jax.jit(train_fn).lower().compile()
        print(
            f"JAX compilation finished in {(time.time() - s_time):.2f} seconds, starting training..."
        )
        updated_self = train_fn()
        return updated_self

    def do_for_each_agent(
        self,
        func,
        shared_argnames: list[str] = [],
        **func_args,
    ):
        if not self.multi_agent_env:
            # if not multiagent, just call the function
            return func(**func_args)

        def map_one_level(f, tree, *rest):
            seen_pytree = False

            def is_leaf(node):
                nonlocal seen_pytree
                if node is tree:
                    if seen_pytree:
                        """
                        From eqx.tree_flatten_one_level:
                        https://github.com/patrick-kidger/equinox/tree/main/equinox
                        """
                        try:
                            type_string = type(tree).__name__
                        except AttributeError:
                            type_string = "<unknown>"
                        raise ValueError(
                            f"PyTree node of type `{type_string}` is immediately "
                            "self-referential; that is to say it appears within its own PyTree "
                            "structure as an immediate subnode. (For example "
                            "`x = []; x.append(x)`.) This is not allowed."
                        )
                    else:
                        seen_pytree = True
                    return False
                else:
                    return True

            return jax.tree.map(f, tree, *rest, is_leaf=is_leaf)

        per_agent_args = {}
        shared_args = {}
        for k, v in func_args.items():
            if k in shared_argnames:
                shared_args[k] = v
            else:
                per_agent_args[k] = v

        func = partial(func, **shared_args)

        all_func_args = inspect.signature(func).parameters.keys()
        # remove the shared args from the all_func_args
        func_args_order = [arg for arg in all_func_args if arg in per_agent_args.keys()]

        # now per_agent_args needs to be reordered to match the func_args_order
        per_agent_args = {arg: per_agent_args[arg] for arg in func_args_order}

        res = map_one_level(lambda *args: func(*args), *per_agent_args.values())
        return res
