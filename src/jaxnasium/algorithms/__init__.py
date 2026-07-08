try:
    print("importing jaxnasium.algorithms")
    from ._algorithm import RLAgent as RLAgent, RLAlgorithm as RLAlgorithm
    from ._architectures import (
        CNN as CNN,
        MLP as MLP,
        BroNet as BroNet,
        Identity as Identity,
    )
    from ._core import (
        Normalizer as Normalizer,
        PrioritizedTransitionBuffer as PrioritizedTransitionBuffer,
        PyTreeObsSpaceNetwork as PyTreeObsSpaceNetwork,
        PyTreeOutputNetwork as PyTreeOutputNetwork,
        Schedule as Schedule,
        Transition as Transition,
        TransitionBuffer as TransitionBuffer,
        pretty_print_network as pretty_print_network,
        scan_callback as scan_callback,
        set_weight_bias as set_weight_bias,
    )

    # isort: off
    from ._btr import BTR as BTR
    from ._dqn import DQN as DQN
    from ._ppo import PPO as PPO
    from ._pqn import PQN as PQN
    from ._sac import SAC as SAC

    # isort: on
    from .agent_networks import (
        ActorNetwork as ActorNetwork,
        AdvantageNetwork as AdvantageNetwork,
        QValueNetwork as QValueNetwork,
        ValueNetwork as ValueNetwork,
    )

    print("importing jaxnasium.algorithms")

except ImportError:
    raise ImportError(
        """Trying to import jaxnasium.algorithms without jaxnasium[algs] installed,
        please install it with pip install jaxnasium[algs]"""
    )
