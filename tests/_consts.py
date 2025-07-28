from jymkit.algorithms import PPO, SAC

DISCRETE_ALGS = [PPO, SAC]
CONTINUOUS_ALGS = [PPO]

PPO_MIN_CONFIG = {
    "num_envs": 2,
    "num_epochs": 1,
    "num_minibatches": 1,
    "total_timesteps": 1_000,
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
