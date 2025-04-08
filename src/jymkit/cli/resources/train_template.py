import jax
from jaxtyping import PRNGKeyArray

import jymkit as jym
from jymkit.algorithms import PPOAgent

# from <PROJECTNAME> import ExampleEnv


def do_random_evaluation(
    key: PRNGKeyArray, env: jym.Environment, num_repitions: int = 10
):
    """Perform some random steps to set a baseline for the environment."""
    rewards = 0.0
    for _ in range(num_repitions):
        obs, env_state = env.reset(key)
        while True:
            key, key = jax.random.split(key)
            action = env.action_space.sample(key)
            (obs, reward, terminated, truncated, info), env_state = env.step(
                key, env_state, action
            )
            rewards += reward
            if terminated or truncated:
                break
    return rewards / num_repitions


if __name__ == "__main__":
    env = ExampleEnv()
    rng = jax.random.PRNGKey(0)

    random_rewards = do_random_evaluation(rng, env)
    print(f"Random Agent average reward: {random_rewards}")

    # RL Training with PPO
    agent = PPOAgent(
        total_timesteps=50000,
        num_steps=32,
        ent_coef=0.0,
        debug=True,  # Log rewards during training
        learning_rate=2.5e-3,
    )
    agent = agent.train(rng, env)
