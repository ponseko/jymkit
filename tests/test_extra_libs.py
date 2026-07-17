import pytest
from _test_utils import run_env_and_agent_env_test

# Only run these in "external" mode
pytestmark = pytest.mark.external

pytest.importorskip("econojax")
pytest.importorskip("rice_jax")
pytest.importorskip("chargax")


def test_econojax_env():
    from econojax import EconoJax  # type: ignore

    env = EconoJax(num_population=4)
    run_env_and_agent_env_test(env, test_reset=True, test_step=True, flatten_obs=True)


@pytest.mark.skip(reason="Rice JAX is undergoing an update")
def test_rice_jax_env():
    from rice_jax import Rice  # type: ignore
    from rice_jax.util import load_region_yamls  # type: ignore

    region_yamls = load_region_yamls(3)
    env = Rice(region_yamls)

    run_env_and_agent_env_test(env, test_reset=True, test_step=True, flatten_obs=True)


def test_chargax_env():
    from chargax import Chargax, ChargingStation  # type: ignore

    station = ChargingStation.init_default_station()
    env = Chargax(station=station)

    run_env_and_agent_env_test(env, test_reset=True, test_step=True, flatten_obs=True)


# def test_econojax_env():
#     try:
#         from econojax import EconoJax  # pyright: ignore
#     except ImportError:
#         pytest.skip("Econojax is not installed.")

#     env = EconoJax(num_population=4)
#     agent = jaxnasium.algorithms.PPO(**TEST_CONSTS.PPO_MIN_CONFIG)
#     agent = agent.train(jax.random.PRNGKey(1), env)


# def test_rice_jax_env():
#     try:
#         from rice_jax import Rice  # pyright: ignore
#         from rice_jax.util import load_region_yamls  # pyright: ignore
#     except ImportError:
#         pytest.skip("rice_jax is not installed.")

#     region_yamls = load_region_yamls(3)
#     env = Rice(region_yamls)
#     agent = jaxnasium.algorithms.PPO(**TEST_CONSTS.PPO_MIN_CONFIG)
#     agent = agent.train(jax.random.PRNGKey(1), env)


# def test_chargax_env():
#     try:
#         from chargax import Chargax, get_electricity_prices
#     except ImportError:
#         pytest.skip("chargax is not installed.")

#     env = Chargax(
#         elec_grid_buy_price=get_electricity_prices("2023_NL"),
#         elec_grid_sell_price=get_electricity_prices("2023_NL") - 0.02,
#     )
#     agent = jaxnasium.algorithms.PPO(**TEST_CONSTS.PPO_MIN_CONFIG)
#     agent = agent.train(jax.random.PRNGKey(1), env)  # type: ignore
