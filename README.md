
# JymKit

JymKit is your lightweight utility library for reinforcement learning projects in JAX. 

📖 [Documentation](https://ponseko.github.io/jymkit/)

## 🚀 Getting started

JymKit lets you bootstrap your new reinforcement learning projects directly from the command line. As such, for new projects, the easiest way to get started is via [uv](https://docs.astral.sh/uv/getting-started/installation/):

> ```bash
> uvx jymkit <projectname>
> uv run example_train.py
> 
> # ... or via pipx
> pipx run jymkit <projectname>
> # activate a virtual environment in your preferred way, e.g. conda
> python example_train.py
> ```

For existing projects, you can simply install JymKit via `pip` and import the required functionality.

> ```bash
> pip install jymkit
> ```

> ```python
> import jax
> import jymkit as jym
> from jymkit.algorithms import PPO
> 
> env = jym.make("CartPole")
> env = jymkit.LogWrapper(env)
> rng = jax.random.PRNGKey(0)
> agent = PPO(total_timesteps=5e5, learning_rate=2.5e-3)
> agent = agent.train(rng, env)
> ```

## 🏠 Environments

JymKit is not aimed at delivering a full environment suite. However, it does come equipped with a `jym.make(...)` command to import environments from existing suites and wrap them appropriately to the JymKit API standard. For example, using environments from Gymnax:

```python
import jymkit as jym
from jymkit.algorithms import PPO
import jax

env = jym.make("Breakout-MinAtar")
env = jym.FlattenObservationWrapper(env)
env = jym.LogWrapper(env)

agent = PPO(**some_good_hyperparameters)
agent = agent.train(jax.random.PRNGKey(0), env)

# > Using an environment from Gymnax via gymnax.make(Breakout-MinAtar).
# > Wrapping Gymnax environment with GymnaxWrapper
# >  Disable this behavior by passing wrapper=False
# > Wrapping environment in VecEnvWrapper
# > ... training results
```

> For convenience, JymKit will include the 5 [classic-control environments](https://gymnasium.farama.org/environments/classic_control/) (2/5 currently).


> Currently, only Gymnax environments can be imported directly. More libraries are coming up. Note that some Gymnax environments no longer work properly with newer versions of JAX. 

### Environment API

The JymKit API stays close to the *somewhat* established [Gymnax](https://github.com/RobertTLange/gymnax) API for the `reset()` and `step()` functions, but allows for truncated episodes in a manner closer to [Gymnasium](https://gymnasium.farama.org/).

```python
env = jym.make(...)

obs, env_state = env.reset(key) # <-- Mirroring Gymnax
(obs, reward, terminated, truncated, info), env_state = env.step(key, state, action) # <-- Gymnasium Timestep tuple with state information
```

## 🤖 Algorithms

Algorithms in `jymkit.algorithms` are built following a near-single-file implementation philosophy in mind. In contrast to implementations in [CleanRL](https://github.com/vwxyzjn/cleanrl) or [PureJaxRL](https://github.com/luchris429/purejaxrl), JymKit algorithms are built in Equinox and follow a class-based design with a familiar [Stable-Baselines](https://github.com/DLR-RM/stable-baselines3) API. 

Each algorithm supports both discrete- and continuous action/observation space -- adjusting based on the provided environment `observation_space` and `action_space`. Additionally, the implementations support multi-agent environments out of the box.

> Currently, only a `PPO` implementation is implemented. More will be included in the near future, but the general idea is not to recreate every algorithm.

```python
from jymkit.algorithms import PPO
import jax

env = ...
agent = PPO(**some_good_hyperparameters)
agent = agent.train(jax.random.PRNGKey(0), env)
```