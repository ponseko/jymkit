try:
    # Weird import for proper copy from the CLI
    from jymkit.algorithms._algorithm import RLAlgorithm as RLAlgorithm

    from ._networks import (
        ActorNetwork as ActorNetwork,
        CriticNetwork as CriticNetwork,
    )
    from ._ppo import PPO as PPO

except ImportError:
    raise ImportError(
        """Trying to import jymkit.algorithms without jymkit[algs] installed,
        please install it with pip install jymkit[algs]"""
    )
