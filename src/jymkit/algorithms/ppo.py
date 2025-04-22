import inspect
import time
from dataclasses import replace
from functools import partial
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, PRNGKeyArray, PyTree, PyTreeDef

import jymkit as jym
from jymkit._environment import ORIGINAL_OBSERVATION_KEY

from .networks import ActorNetwork, CriticNetwork


# Define a simple tuple to hold the state of the environment.
# This is the format we will use to store transitions in our buffer.
class Transition(eqx.Module):
    observation: Array
    action: Array
    reward: Array
    terminated: Array
    log_prob: Array
    info: Array
    value: Array
    next_value: Array
    return_: Array = None
    advantage_: Array = None

    @property
    def structure(self) -> PyTreeDef:
        """
        Returns the top-level structure of the transition objects (using reward as a reference).
        This is either PyTreeDef(*) for single agents
        or PyTreeDef((*, x num_agents)) for multi-agent environments.
        usefull for unflattening Transition.flat.properties back to the original structure.
        """
        return jax.tree.structure(self.reward)

    @property
    def view_flat(self) -> "Transition":
        """
        Returns a flattened version of the transition.
        Where possible, this is a jnp.stack of the leaves.
        Otherwise, it returns a list of leaves.
        """

        def return_as_stack_or_list(x):
            x = jax.tree.leaves(x)
            try:
                return jnp.stack(x, axis=-1).squeeze()
            except ValueError:
                return x

        return jax.tree.map(
            return_as_stack_or_list,
            self,
            is_leaf=lambda y: y is not self,
        )

    @property
    def view_transposed(self) -> PyTree["Transition"]:
        """
        The original transition is a Transition of PyTrees
            e.g. Transition(observation={a1: ..., a2: ...}, action={a1: ..., a2: ...}, ...)
        The transposed transition is a PyTree of Transitions
            e.g. {a1: Transition(observation=..., action=..., ...), a2: Transition(observation=..., action=..., ...), ...}
        This is useful for multi-agent environments where we want to have a single Transition object per agent.
        In single-agent environments, this will be the same as the original transition.
        """
        if self.structure.num_leaves == 1:  # single agent
            return self

        field_names = list(self.__dataclass_fields__.keys())

        fields = {}
        for f in field_names:
            attr = getattr(self, f)
            fields[f] = jax.tree.leaves(attr, is_leaf=lambda x: x is not attr)

        per_agent_transitions = []
        for i in range(len(fields[field_names[0]])):
            agent_transition = Transition(
                **{
                    field_name: fields[field_name][i]
                    for field_name in field_names
                    if field_name != "info"
                    and (fields[field_name] is not None)
                    # and field_name != "advantage_"
                    and field_name != "terminated"
                },
                terminated=fields["terminated"][0],
                info=fields["info"],
            )
            per_agent_transitions.append(agent_transition)

        return jax.tree.unflatten(self.structure, per_agent_transitions)


class AgentState(eqx.Module):
    class Networks(eqx.Module):
        actor: ActorNetwork
        critic: CriticNetwork

    networks: Networks
    optimizer_state: optax.OptState


class PPOAgent(eqx.Module):
    state: PyTree[AgentState] = None
    optimizer: optax.GradientTransformation = eqx.field(static=True, default=None)
    multi_agent_env: bool = eqx.field(static=True, default=False)

    @property
    def networks(self):
        return jax.tree.map(
            lambda x: x.networks,
            self.state,
            is_leaf=lambda x: isinstance(x, AgentState),
        )

    @property
    def optimizer_state(self):
        return jax.tree.map(
            lambda x: x.optimizer_state,
            self.state,
            is_leaf=lambda x: isinstance(x, AgentState),
        )

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
        networks = AgentState.Networks(
            actor=actor,
            critic=critic,
        )
        optimizer_state = optimizer.init(networks)
        return AgentState(
            networks=networks,
            optimizer_state=optimizer_state,
        )

    def forward_critic(self, observation):
        value = self.do_for_each_agent(
            lambda a, o: jax.vmap(a.networks.critic)(o),
            a=self.state,
            o=observation,
        )
        return value

    def forward_actor(self, observation, key, get_log_prob=True):
        action_dist = self.do_for_each_agent(
            lambda a, o: a.networks.actor(o),
            a=self.state,
            o=observation,
        )
        action = self.do_for_each_agent(
            lambda a, key: a.sample(seed=key),
            a=action_dist,
            key=key,
            shared_argnames=["key"],  # TODO: tmp
        )
        if not get_log_prob:
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

            action = self.forward_actor(obs, action_key, get_log_prob=False)

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
            sample_key = jax.random.split(sample_key, self.num_envs)
            action, log_prob = jax.vmap(self.forward_actor)(last_obs, sample_key)

            # take a step in the environment
            rng, key = jax.random.split(rng)
            step_key = jax.random.split(key, self.num_envs)
            (obsv, reward, terminated, truncated, info), env_state = jax.vmap(env.step)(
                step_key, env_state, action
            )

            value = self.forward_critic(last_obs)
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

    def train(self, key: PRNGKeyArray, env: jym.Environment) -> "PPOAgent":
        def train_iteration(runner_state, _):
            def update_epoch(
                trajectory_batch: Transition, key: PRNGKeyArray
            ) -> PPOAgent:
                """Do one epoch of update"""

                @eqx.filter_grad
                def __ppo_los_fn(
                    params: AgentState.Networks,
                    train_batch: Transition,
                    # observations,
                    # actions,
                    # log_probs,
                    # values,
                    # advantages,
                    # returns,
                ):
                    # breakpoint()
                    action_dist = jax.vmap(params.actor)(train_batch.observation)
                    log_prob = action_dist.log_prob(train_batch.action)
                    entropy = action_dist.entropy().mean()
                    value = jax.vmap(params.critic)(train_batch.observation)

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

                    # Total loss
                    total_loss = (
                        actor_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                    )
                    return total_loss  # , (actor_loss, value_loss, entropy)

                def __update_over_minibatch(
                    current_self: PPOAgent, minibatch: Transition
                ):
                    # breakpoint()

                    grads = self.do_for_each_agent(
                        lambda params, batch: __ppo_los_fn(params, batch),
                        params=current_self.networks,
                        batch=minibatch.view_transposed,
                    )

                    # observations, actions, log_probs, values, advantages, returns = (
                    #     minibatch.observation,
                    #     minibatch.action,
                    #     minibatch.log_prob,
                    #     minibatch.value,
                    #     minibatch.advantage_,
                    #     minibatch.return_,
                    # )

                    # grads = self.do_for_each_agent(
                    #     lambda params, obs, act, logp, val, adv, ret: __ppo_los_fn(
                    #         params, obs, act, logp, val, adv, ret
                    #     ),
                    #     params=current_self.networks,
                    #     obs=observations,
                    #     act=actions,
                    #     logp=log_probs,
                    #     val=values,
                    #     adv=advantages,
                    #     ret=returns,
                    # )

                    updates, optimizer_state = self.do_for_each_agent(
                        lambda u, s: self.optimizer.update(u, s),
                        u=grads,
                        s=current_self.optimizer_state,
                    )

                    new_networks = self.do_for_each_agent(
                        lambda params, updates: optax.apply_updates(params, updates),
                        params=current_self.networks,
                        updates=updates,
                    )

                    train_state = self.do_for_each_agent(
                        lambda networks, opt_state: AgentState(
                            networks=AgentState.Networks(
                                actor=networks.actor,
                                critic=networks.critic,
                            ),
                            optimizer_state=opt_state,
                        ),
                        networks=new_networks,
                        opt_state=optimizer_state,
                    )
                    return replace(self, state=train_state), None

                batch_idx = jax.random.permutation(key, self.batch_size)

                # reshape (flatten over first dimension)
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

            if self.debug:
                # eval_rewards = self.evaluate(eval_key, env)
                # metric["eval_rewards"] = eval_rewards

                def callback(info):
                    # Only print after training for 10%
                    if (
                        info["timestep"][-1][0]
                        < (self.total_timesteps // self.num_envs) * 0.1
                    ):
                        return
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                        info["timestep"][info["returned_episode"]] * self.num_envs
                    )
                    for t in range(len(timesteps)):
                        print(
                            f"global step={timesteps[t]}, episodic return={return_values[t]}"
                        )
                    # print(
                    #     f"timestep={(info['timestep'][-1][0] * self.num_envs)}, eval rewards={info['eval_rewards']}"
                    # )

                jax.debug.callback(callback, metric)

            runner_state = (self, env_state, last_obs, rng)
            return runner_state, _

        if not self.is_initialized:
            self = self.init(key, env)

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
        print(f"Compilation took {(time.time() - s_time):.2f} s, starting training...")
        s_time = time.time()
        updated_self = train_fn()
        print(f"Training finished in {(time.time() - s_time):.2f} seconds")
        return updated_self

    def do_for_each_agent(
        self,
        func: Callable,
        shared_argnames: list[str] = [],
        **func_args,
    ):
        if not self.multi_agent_env:
            return func(**func_args)

        # Separate shared and per-agent args
        shared_args = {k: v for k, v in func_args.items() if k in shared_argnames}
        per_agent_args = {
            k: v for k, v in func_args.items() if k not in shared_argnames
        }

        # Prepare a function that takes only per-agent args
        def agent_func(*args):
            per_agent_kwargs = dict(zip(per_agent_args.keys(), args))
            return func(**per_agent_kwargs, **shared_args)

        def map_one_level(f, tree, *rest):
            # NOTE: Immidiately self-referential trees may pose a problem.
            # see eqx.tree_flatten_one_level
            # Likely not a problem here.
            return jax.tree.map(f, tree, *rest, is_leaf=lambda x: x is not tree)

        # Use jax.tree_map to apply agent_func to the per-agent args
        res = map_one_level(agent_func, *per_agent_args.values())

        one_level_leaves, structure = eqx.tree_flatten_one_level(res)
        if isinstance(one_level_leaves[0], tuple):
            tupled = tuple([list(x) for x in zip(*one_level_leaves)])
            res = tuple(jax.tree.unflatten(structure, leaves) for leaves in tupled)
        return res

        if not self.multi_agent_env:
            # if not multiagent, just call the function
            return func(**func_args)

        def map_one_level(f, tree, *rest):
            """
            NOTE: Immidiately self-referential trees may pose a problem.
            see eqx.tree_flatten_one_level
            https://github.com/patrick-kidger/equinox/blob/4995b2bed015d6922ca46868cbaf59c767b44682/equinox/_tree.py#L352

            We don't expect this to be a problem here.
            """
            return jax.tree.map(f, tree, *rest, is_leaf=lambda x: x is not tree)

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
        one_level_leaves, structure = eqx.tree_flatten_one_level(res)
        if isinstance(one_level_leaves[0], tuple):
            tupled = tuple([list(x) for x in zip(*one_level_leaves)])
            res = tuple(jax.tree.unflatten(structure, leaves) for leaves in tupled)
        return res
