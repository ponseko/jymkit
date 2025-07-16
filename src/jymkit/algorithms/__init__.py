try:
    # Weird import for proper copy from the CLI
    from jymkit.algorithms._algorithm import RLAlgorithm as RLAlgorithm

    from ._networks import (
        ActorNetwork as ActorNetwork,
        QValueNetwork as QValueNetwork,
        ValueNetwork as ValueNetwork,
    )
    from ._ppo import PPO as PPO
    from ._sac import SAC as SAC

except ImportError:
    raise ImportError(
        """Trying to import jymkit.algorithms without jymkit[algs] installed,
        please install it with pip install jymkit[algs]"""
    )
