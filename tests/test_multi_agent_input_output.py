"""Structural input/output tests for the algorithm networks. These obs/action spaces
cover the possible paths that we want supported in terms of possible input / output networks
and multi-agent support.
"""

from typing import Any, Callable

import pytest
from _proxy_test_envs import (
    MULTI_AGENT_ACT_SPACES,
    MULTI_AGENT_OBS_SPACES,
    SINGLE_AGENT_ACT_SPACES,
    SINGLE_AGENT_OBS_SPACES,
    make_proxy_env,
    mirror_agent_structure,
    obs_box_vector,
)
from _test_utils import run_env_and_agent_env_test

default_obs_space = obs_box_vector

INPUT_SPACE_CASES = [
    *[pytest.param(fn, False, id=name) for name, fn in SINGLE_AGENT_OBS_SPACES.items()],
    *[pytest.param(fn, True, id=name) for name, fn in MULTI_AGENT_OBS_SPACES.items()],
]


@pytest.mark.parametrize("obs_space_fn, multi_agent", INPUT_SPACE_CASES)
def test_various_input_spaces(obs_space_fn: Callable[[], Any], multi_agent: bool):
    env = make_proxy_env(obs_space_fn, multi_agent=multi_agent)
    run_env_and_agent_env_test(
        env, test_reset=True, test_step=True, flatten_obs=False, test_train_runs=True
    )


OUTPUT_SPACE_CASES = [
    *[pytest.param(fn, False, id=name) for name, fn in SINGLE_AGENT_ACT_SPACES.items()],
    *[pytest.param(fn, True, id=name) for name, fn in MULTI_AGENT_ACT_SPACES.items()],
]


@pytest.mark.parametrize("action_space_fn, multi_agent", OUTPUT_SPACE_CASES)
def test_various_output_spaces(action_space_fn: Callable[[], Any], multi_agent: bool):
    if multi_agent:
        obs_space = mirror_agent_structure(action_space_fn(), default_obs_space)
        obs_space_fn: Callable[[], Any] = lambda: obs_space
    else:
        obs_space_fn = default_obs_space

    env = make_proxy_env(obs_space_fn, action_space_fn, multi_agent=multi_agent)
    run_env_and_agent_env_test(
        env, test_reset=True, test_step=True, flatten_obs=False, test_train_runs=True
    )
