"""Temporary stub for backwards compatibility."""

import warnings

from jaxnasium.algorithms._core._buffer import TransitionBuffer as TransitionBuffer
from jaxnasium.algorithms._core._distributions import (
    DistraxContainer as DistraxContainer,
    TanhNormalFactory as TanhNormalFactory,
)
from jaxnasium.algorithms._core._initialization import (
    rl_initialization as rl_initialization,
)
from jaxnasium.algorithms._core._logging import (
    pretty_print_network as pretty_print_network,
    scan_callback as scan_callback,
)
from jaxnasium.algorithms._core._multi_agent import (
    MultiAgentWrapper as MultiAgentWrapper,
)
from jaxnasium.algorithms._core._normalization import (
    Normalizer as Normalizer,
    RunningStatisticsState as RunningStatisticsState,
)
from jaxnasium.algorithms._core._scan import scan_transitions as scan_transitions
from jaxnasium.algorithms._core._schedule import Schedule as Schedule
from jaxnasium.algorithms._core._transition import Transition as Transition

warnings.warn(
    "importing from jaxnasium.algorithms.utils is deprecated."
    "You can now import directly from jaxnasium.algorithms",
    DeprecationWarning,
    stacklevel=2,
)
