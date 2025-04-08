try:
    from .ppo import PPOAgent

except ImportError:
    raise ImportError(
        """Trying to import jymkit.algorithms without jymkit[algs] installed,
        please install it with pip install jymkit[algs]"""
    )
