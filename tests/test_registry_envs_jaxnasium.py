import pytest
from _test_utils import registry_envs_for_package, run_env_and_agent_env_test

import jaxnasium  # noqa: F401

JAXNASIUM_ENVS = registry_envs_for_package("jaxnasium")


@pytest.mark.parametrize("env_id", JAXNASIUM_ENVS)
def test_jaxnasium_env_smoke(env_id: str) -> None:
    run_env_and_agent_env_test(
        env_id,
        test_reset=True,
        test_step=True,
        flatten_obs=True,
    )
