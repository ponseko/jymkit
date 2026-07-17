import pytest
from _test_utils import registry_envs_for_package, run_env_and_agent_env_test

# Only run these in "external" mode
pytestmark = pytest.mark.external

pytest.importorskip("gymnax")


GYMNAX_ENVS = registry_envs_for_package("gymnax")


@pytest.mark.parametrize("env_id", GYMNAX_ENVS)
def test_gymnax_env_smoke(env_id: str) -> None:
    run_env_and_agent_env_test(
        env_id,
        test_reset=False,
        test_step=False,
        flatten_obs=True,
    )
