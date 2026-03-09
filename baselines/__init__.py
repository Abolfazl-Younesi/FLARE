"""
FLARE Baselines Package
=======================
Modular, plug-in FL aggregation strategies compatible with Flower 1.20.0.

Available strategies
--------------------
aggr_strategies  : FedAvgBaseline, KrumStrategy, MultiKrumStrategy, TrimmedMeanStrategy
fltrust          : FLTrustStrategy
flame            : FLAMEStrategy
brea             : BREAStrategy
repunet          : RepuNetStrategy
"""

from baselines.aggr_strategies import (
    FedAvgBaseline,
    KrumStrategy,
    MultiKrumStrategy,
    TrimmedMeanStrategy,
)
from baselines.fltrust import FLTrustStrategy
from baselines.flame import FLAMEStrategy
from baselines.brea import BREAStrategy
from baselines.repunet import RepuNetStrategy

__all__ = [
    "FedAvgBaseline",
    "KrumStrategy",
    "MultiKrumStrategy",
    "TrimmedMeanStrategy",
    "FLTrustStrategy",
    "FLAMEStrategy",
    "BREAStrategy",
    "RepuNetStrategy",
]
