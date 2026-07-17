import pytest
from _test_utils import registry_envs_for_package, run_env_and_agent_env_test

# Only run these in "external" mode
pytestmark = pytest.mark.external

pytest.importorskip("pgx")


PGX_ENVS = registry_envs_for_package("pgx")


@pytest.mark.parametrize("env_id", PGX_ENVS)
def test_pgx_env_smoke(env_id: str) -> None:
    run_env_and_agent_env_test(
        env_id,
        test_reset=False,
        test_step=False,
        flatten_obs=True,
    )
