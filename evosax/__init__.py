from .strategy import Strategy, EvoState, EvoParams
from .strategies import (
    CMA_ES,
    OpenES,
    Sep_CMA_ES,
    DiffusionEvolution,
    PolarizedCBO,
    OptimizationViaIntegration,
    ClusteredCBO,
    AdaptivePcboOvi,
    AdaptiveCcboOvi,
)
from .core import FitnessShaper, ParameterReshaper
from .utils import ESLog
from .networks import NetworkMapper
from .problems import ProblemMapper


Strategies = {
    "CMA_ES": CMA_ES,
    "OpenES": OpenES,
    "Sep_CMA_ES": Sep_CMA_ES,
    "DiffusionEvolution": DiffusionEvolution,
    "PolarizedCBO": PolarizedCBO,
    "OptimizationViaIntegration": OptimizationViaIntegration,
    'ClusteredCBO': ClusteredCBO,
    'AdaptivePcboOvi': AdaptivePcboOvi,
    'AdaptiveCcboOvi': AdaptiveCcboOvi
}

