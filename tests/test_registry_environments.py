import jax
import pytest

import jymkit as jym
from jymkit._environment import ORIGINAL_OBSERVATION_KEY
from jymkit.algorithms import DQN, PPO, PQN, SAC

TEST_ENVS = (  # All included envs + arbitrary subset of external envs
    list(jym.registry._environments.keys()) + list(jym.registry._aliases.keys())[::8]
)
TEST_ENVS_SMALL = (  # All included envs + arbitrary subset of external envs
    list(jym.registry._environments.keys()) + list(jym.registry._aliases.keys())[::16]
)
TEST_SET_SMALL_ONLY_DISCRETE = [
    "MountainCar-v0",
    "Acrobot-v1",
    "gymnax:Acrobot-v1",
    "Breakout-MinAtar",
    "Catch-bsuite",
    "FourRooms-misc",
    "ant",
    "halfcheetah",
    "Snake-v1",
    "Game2048-v1",
    "chess",
    "go_9x9",
    "coin_game",
    "jaxnav",
    "Navix-FourRooms-v0",
    "XLand-MiniGrid-R1-9x9",
]

# Skip certain environments that might have special requirements or known issues
SKIP_ENVS = {
    "SimpleBandit-bsuite": "Known issue with this environment  (https://github.com/RobertTLange/gymnax/pull/106)",
    # "some_problematic_env": "Known issue with this environment"
}


def run_env_3_steps(env_id):
    """Creates an environment, resets it, and performs 3 steps with sampled actions."""
    try:
        env = jym.make(env_id)
    except ImportError as e:
        pytest.skip(f"Skipping {env_id} due to ImportError: {e}")

    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key)

    # Do 3 steps (with sampled actions)
    for i in range(3):
        key = jax.random.PRNGKey(i)
        # env.action_space may give back a pytree of spaces, which cannot be sampled without a jax.map
        # env.sample_action always applies the jax.map and should therefore always work
        action = env.sample_action(jax.random.PRNGKey(1))
        timestep, state = env.step(key, state, action)

    assert ORIGINAL_OBSERVATION_KEY in timestep.info  # type: ignore


@pytest.mark.parametrize("env_id", TEST_ENVS)
def test_subset_registered_environments_run(env_id):
    if env_id in SKIP_ENVS:
        pytest.skip(f"Skipping {env_id}: {SKIP_ENVS[env_id]}")
    run_env_3_steps(env_id)


@pytest.mark.parametrize("env_id", TEST_ENVS)
@pytest.mark.parametrize("alg", [PPO, SAC])
def test_subset_registered_environments_train(env_id, alg):
    if env_id in SKIP_ENVS:
        pytest.skip(f"Skipping {env_id}: {SKIP_ENVS[env_id]}")
    env = jym.make(env_id)
    config = {"total_timesteps": 500, "log_function": None, "num_envs": 2}
    if alg == PPO or alg == PQN:
        config = {**config, "num_epochs": 1, "num_minibatches": 1}
    agent = alg(**config)
    agent = agent.train(jax.random.PRNGKey(0), env)
    agent.evaluate(jax.random.PRNGKey(1), env, num_eval_episodes=2)


@pytest.mark.parametrize("env_id", TEST_SET_SMALL_ONLY_DISCRETE)
@pytest.mark.parametrize("alg", [PQN, DQN])
def test_subset_registered_environments_train_discrete(env_id, alg):
    if env_id in SKIP_ENVS:
        pytest.skip(f"Skipping {env_id}: {SKIP_ENVS[env_id]}")
    env = jym.make(env_id)
    config = {"total_timesteps": 1000, "log_function": None, "num_envs": 2}
    if alg == PPO or alg == PQN:
        config = {**config, "num_epochs": 1, "num_minibatches": 1}
    agent = alg(**config)
    agent = agent.train(jax.random.PRNGKey(0), env)
    agent.evaluate(jax.random.PRNGKey(1), env, num_eval_episodes=2)


@pytest.mark.skip(reason="Slow test; we run subset of environments instead")
def test_all_registered_environments_run():
    """Test that all registered environments can be loaded, reset, and stepped."""

    all_env_ids = jym.registry.registered_envs

    # Track results
    successful_envs = []
    failed_envs = []

    print(f"\nTesting {len(all_env_ids)} registered environments...")

    for env_id in all_env_ids:
        if env_id in SKIP_ENVS:
            print(f"⏭️  Skipping {env_id}: {SKIP_ENVS[env_id]}")
            continue

        try:
            print(f"🧪 Testing {env_id}...")
            run_env_3_steps(env_id)
            successful_envs.append(env_id)

        except Exception as e:
            failed_envs.append((env_id, str(e)))
            print(f"   ❌ {env_id} failed: {e}")

    # Assert that all non-skipped environments passed
    assert len(failed_envs) == 0, (
        f"Some environments failed ({len(failed_envs)}): {failed_envs}"
    )
