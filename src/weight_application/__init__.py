"""Physiology-informed simulation adapters for v1.1."""

from .adapter import SyntheticWeightAdapter
from .constraints import SYNTHETIC_E0_CONSTRAINTS
from .decisions import SYNTHETIC_E0_DECISIONS
from .model_roles import (
    E0_SYNTHETIC_PLANNING_BINDING,
    V11_MQ1_EVALUATION_PARAMETER_BINDING,
    V11_MQ1_PLANNING_BINDING,
)
from .objectives import SYNTHETIC_E0_OBJECTIVES
from .state import (
    SyntheticWeightModel,
    SyntheticWeightProjection,
    SyntheticWeightState,
    WeightStateError,
)

__all__ = [
    "SyntheticWeightAdapter",
    "SYNTHETIC_E0_CONSTRAINTS",
    "SYNTHETIC_E0_DECISIONS",
    "E0_SYNTHETIC_PLANNING_BINDING",
    "V11_MQ1_EVALUATION_PARAMETER_BINDING",
    "V11_MQ1_PLANNING_BINDING",
    "SYNTHETIC_E0_OBJECTIVES",
    "SyntheticWeightModel",
    "SyntheticWeightProjection",
    "SyntheticWeightState",
    "WeightStateError",
]
from .illustrative_adapter import (
    IllustrativeHallEngineeringAdapter,
    R6_E3_SCENARIOS,
)
from .formal_e3_adapter import FormalHallE3Adapter
from e3_inputs.contract import (
    E3_SCENARIOS,
    E3InputContractError,
    E3SubjectParameters,
    generate_subject_parameters,
    target_mass_kg,
)

__all__ += [
    "IllustrativeHallEngineeringAdapter",
    "FormalHallE3Adapter",
    "R6_E3_SCENARIOS",
    "E3_SCENARIOS",
    "E3InputContractError",
    "E3SubjectParameters",
    "generate_subject_parameters",
    "target_mass_kg",
]
