from jymkit.algorithms import PPO, SAC

DISCRETE_ALGS = [PPO, SAC]
CONTINUOUS_ALGS = [PPO, SAC]

PPO_MIN_CONFIG = {
    "num_envs": 2,
    "num_epochs": 1,
    "num_minibatches": 1,
    "total_timesteps": 1_000,
    "log_function": None,
}

SAC_CONTINUOUS_CONFIG = {
    "total_timesteps": 1_000_000,
    "num_envs": 8,
    "learning_rate": 0.001,
    "update_every": 8,
    "batch_size": 256,
    "target_entropy_scale": 1.5,
    "anneal_entropy_scale": 0.2,
    "replay_buffer_size": 500_000,
    "normalize_rew": False,
    "normalize_obs": False,
    "policy_kwargs": {
        "actor_features": [128, 128],
        "critic_features": [128, 128],
    },
    "log_function": None,
}

CLASSIC_CONTROL_ENVS = [
    "CartPole-v1",
    "MountainCar-v0",
    "Acrobot-v1",
    "Pendulum-v1",
    "ContinuousMountainCar-v0",
]

GYMNAX_TEST_ENVS = [
    "gymnax:CartPole-v1",
    "gymnax:Acrobot-v1",
    "Breakout-MinAtar",
    "Catch-bsuite",
    "FourRooms-misc",
]
JUMANJI_TEST_ENVS = ["Snake-v1", "Game2048-v1", "Cleaner-v0", "Maze-v0"]
BRAX_TEST_ENVS = [
    "ant",
    "halfcheetah",
    "humanoid",
    "inverted_double_pendulum",
    "walker2d",
]
