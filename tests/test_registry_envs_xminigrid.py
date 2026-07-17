import pytest
from _test_utils import registry_envs_for_package, run_env_and_agent_env_test

pytest.importorskip("xminigrid")


XMINIGRID_ENVS = registry_envs_for_package("xminigrid")


@pytest.mark.parametrize("env_id", XMINIGRID_ENVS)
def test_xminigrid_env_smoke(env_id: str) -> None:
    run_env_and_agent_env_test(
        env_id,
        test_reset=False,
        test_step=False,
        flatten_obs=True,
    )
