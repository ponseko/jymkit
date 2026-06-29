try:
    # Weird import for proper copy from the CLI
    from jaxnasium.algorithms._algorithm import (
        RLAgent as RLAgent,
        RLAlgorithm as RLAlgorithm,
    )  # noqa: I001

    from ._core import (
        CNN as CNN,
        MLP as MLP,
        AutoAgentObservationNet as AutoAgentObservationNet,
        AutoAgentOutputNet as AutoAgentOutputNet,
        BroNet as BroNet,
        Identity as Identity,
        Normalizer as Normalizer,
        PrioritizedTransitionBuffer as PrioritizedTransitionBuffer,
        Schedule as Schedule,
        Transition as Transition,
        TransitionBuffer as TransitionBuffer,
        pretty_print_network as pretty_print_network,
        rl_initialization as rl_initialization,
        scan_callback as scan_callback,
    )
    from ._dqn import DQN as DQN
    from ._ppo import PPO as PPO
    from ._pqn import PQN as PQN
    from ._sac import SAC as SAC
    from .agent_networks import (
        ActorNetwork as ActorNetwork,
        AdvantageCriticNetwork as AdvantageCriticNetwork,
        QValueNetwork as QValueNetwork,
        ValueNetwork as ValueNetwork,
    )

except ImportError:
    raise ImportError(
        """Trying to import jaxnasium.algorithms without jaxnasium[algs] installed,
        please install it with pip install jaxnasium[algs]"""
    )
