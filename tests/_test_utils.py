"""Shared helpers for lightweight per-registry environment smoke tests."""

from __future__ import annotations

import warnings
from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import pytest
from _consts import AGENT_MIN_CONFIG, SKIP_AGENT_ENVS, SKIP_ENVS

import jaxnasium as jym
from jaxnasium.algorithms import DQN, PPO, PQN, SAC, RLAlgorithm
from jaxnasium.algorithms.core import Transition


def get_skip_envs(env_reason_dict: dict[str, str]) -> dict[str, str]:
    """Returns the environments that are skipped due to bugs or limitations"""
    skip_envs = env_reason_dict.copy()
    for pattern, value in env_reason_dict.items():
        if "_SUITE_" in pattern:
            suite = pattern.split(":")[1]
            for env_id in registry_envs_for_package(suite):
                skip_envs[env_id] = value
    return skip_envs


def registry_envs_for_package(package: str) -> list[str]:
    """Return sorted registry aliases that resolve to an external package."""
    if package == "jaxnasium":
        return sorted(jym.registry._environments.keys())
    prefix = f"{package}:"
    return sorted(
        alias
        for alias, target in jym.registry._aliases.items()
        if target.startswith(prefix)
    )


def _assert_equal_pytrees(reference, candidate, name: str = " ") -> None:
    assert jax.tree.structure(reference) == jax.tree.structure(candidate), (
        f"Tree structure mismatch for {name}: {jax.tree.structure(reference)} != {jax.tree.structure(candidate)}"
    )
    for ref_leaf, cand_leaf in zip(
        jax.tree.leaves(reference), jax.tree.leaves(candidate), strict=True
    ):
        ref_leaf = jnp.asarray(ref_leaf)
        cand_leaf = jnp.asarray(cand_leaf)
        assert ref_leaf.dtype == cand_leaf.dtype, (
            f"Tree dtype mismatch for {name}: {ref_leaf.dtype} != {cand_leaf.dtype}"
        )
        if (ref_leaf.shape == () and cand_leaf.shape == (1,)) or (
            ref_leaf.shape == (1,) and cand_leaf.shape == ()
        ):
            warnings.warn(
                f"Detected inconsistent scalar shape ({ref_leaf.shape} and {cand_leaf.shape})"
                f"This is often caused by a mismatch in the observation space and the actual"
                f"observation returned by step()/reset(). This will not necessarily cause issues,"
                f" but should likely be fixed in the environment implementation. "
            )
            continue
        assert ref_leaf.shape == cand_leaf.shape, (
            f"Tree shape mismatch for {name}: {ref_leaf.shape} != {cand_leaf.shape}"
        )


def _has_continuous_action_space(env: jym.Environment) -> bool:
    spaces = jax.tree.leaves(env.action_space)
    for space in spaces:
        if hasattr(space, "n") or hasattr(space, "nvec"):
            continue
        return True
    return False


def get_valid_test_algs(env: jym.Environment) -> list[type[RLAlgorithm]]:
    algs = [
        partial[PPO](PPO, **AGENT_MIN_CONFIG),
        partial[SAC](SAC, **AGENT_MIN_CONFIG),
    ]
    if not _has_continuous_action_space(env):
        NO_ACTOR_AGENT_MIN_CONFIG = {
            k: v for k, v in AGENT_MIN_CONFIG.items() if k != "actor_kwargs"
        }
        algs.extend(
            [
                partial[DQN](DQN, **NO_ACTOR_AGENT_MIN_CONFIG),
                partial[PQN](PQN, **NO_ACTOR_AGENT_MIN_CONFIG),
            ]
        )
    return algs


def _make_dummy_update_batch(
    alg: RLAlgorithm, env: jym.Environment, key: jax.Array
) -> Transition:
    obs_key, action_key, next_obs_key = jax.random.split(key, 3)

    observation = env.sample_observation(obs_key)
    next_observation = env.sample_observation(next_obs_key)
    action = alg.get_action(action_key, observation)

    if env.multi_agent:
        reward = jym.tree.map_one_level(lambda x: jnp.array(0.0), action)
    else:
        reward = jnp.array(0.0)

    batch = Transition(
        observation=observation,
        action=action,
        reward=reward,
        terminated=jym.tree.zeros_like(reward, dtype=bool),
        truncated=jym.tree.zeros_like(reward, dtype=bool),
        next_observation=next_observation,
        return_=jym.tree.zeros_like(reward),
        value=jym.tree.ones_like(reward),
        log_prob=jym.tree.zeros_like(reward),
        advantage=jym.tree.zeros_like(reward),
    )

    # the batch is now a single transition, for perhaps multiple agents
    # make it a batch:
    batch = jax.tree.map(lambda x: jnp.broadcast_to(x, (4, *x.shape)), batch)

    return batch


def _run_agent_update(
    alg: RLAlgorithm, batch: Transition, key: jax.Array
) -> RLAlgorithm:
    if isinstance(alg, SAC):
        agent = alg.agent.update_critics_params(key, batch, alg)
        agent = agent.update_actor_params(key, batch, alg)
    else:
        agent = alg.agent.update_params(batch, alg)
    return replace(alg, agent=agent)


def run_env_and_agent_env_test(
    env: jym.Environment | str,
    *,
    test_reset: bool,
    test_step: bool,
    flatten_obs: bool,
    test_train_runs: bool = False,
) -> None:
    if isinstance(env, str):
        env_id = env
        skip_envs = get_skip_envs(SKIP_ENVS)
        if env in SKIP_ENVS:
            pytest.skip(f"Skipping {env}: {skip_envs[env]}")
        try:
            env = jym.make(env)
        except ImportError as e:
            pytest.skip(f"Skipping {env} due to ImportError: {e}")
    else:
        env_id = "_"

    (reset_key, o_sample_key, a_sample_key, a_agent_key, step_key, init_key) = (
        jax.random.split(jax.random.PRNGKey(0), 6)
    )

    if flatten_obs:
        env = jym.wrappers.FlattenObservationWrapper(env)

    sampled_obs = env.sample_observation(o_sample_key)
    sampled_action = env.sample_action(a_sample_key)
    if test_reset:
        reset_obs, reset_state = env.reset(reset_key)
        # _assert_equal_pytrees(sampled_obs, reset_obs, "sampled_obs != reset_obs")
    if test_step:
        assert test_reset, "test_step requires test_reset"
        timestep, step_state = env.step(step_key, reset_state, sampled_action)  # type: ignore
        assert jym.ORIGINAL_OBSERVATION_KEY in timestep.info, (
            f"ORIGINAL_OBSERVATION_KEY not in timestep.info for {env_id}"
        )

    if env_id in get_skip_envs(SKIP_AGENT_ENVS):
        return

    algs = get_valid_test_algs(env)
    for alg_cls in algs:
        alg: RLAlgorithm = alg_cls()  # type: ignore
        alg = alg.init_agent(init_key, env)  # type: ignore
        agent_action = alg.get_action(a_agent_key, sampled_obs)
        _assert_equal_pytrees(
            sampled_action, agent_action, "sampled_action != agent_action"
        )
        if test_step:
            timestep, step_state = env.step(step_key, reset_state, agent_action)  # type: ignore
            assert jym.ORIGINAL_OBSERVATION_KEY in timestep.info, (
                f"ORIGINAL_OBSERVATION_KEY not in timestep.info for {env_id}"
            )
        if test_train_runs:
            batch = _make_dummy_update_batch(alg, env, a_sample_key)
            alg = _run_agent_update(alg, batch, a_agent_key)
