
# <center>JymKit</center>

JymKit is your lightweight utility library for building reinforcement learning projects in JAX. 


> JymKit is in an early stage of development. Functionality is limited and may change at any moment. More features will be added in the near future.

JymKit revolves around three main pillars which complement each other but may be used separately and combined with different libraries or your own code.

- An environment API, sticking close to [Gymnax](https://github.com/RobertTLange/gymnax), but properly handling truncation when required and allowing for variable discounting. JymKit Environments are built as [Equinox](https://docs.kidger.site/equinox/) Modules, neatly integrating them into the JAX ecosystem.
- JymKit provides general algorithms that train according to the [PureJaxRL](https://github.com/luchris429/purejaxrl) pattern for maximum performance, but are built as [Equinox](https://docs.kidger.site/equinox/) Modules. The algorithms are built as single-file implementations, follow a class-based approach and their API resembles that of the stable-baselines.
- JymKit can bootstrap your project by providing some template code for building your reinforcement learning project via `pipx run jymkit <projectname>`.

## Getting started

For new projects, the easiest way to get started is via [uv](https://docs.astral.sh/uv/getting-started/installation/):

> ```bash
> uvx jymkit <projectname>
> uv run example_train.py
> 
> # ... or via pipx
> pipx run jymkit <projectname>
> # ... active a virtual environment in your prefered way
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
> from jymkit.envs import CartPole
> 
> env = CartPole()
> rng = jax.random.PRNGKey(0)
> agent = PPO(total_timesteps=5e5, debug=True, learning_rate=2.5e-3)
> agent = agent.train(rng, env)
> ```
