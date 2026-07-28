"""F08 model-role and information-time boundary for v1.1.

Provenance disposition:
    FORMAL_V1/dt_ramde_formal/model_roles.py -> F08 CONDITIONAL_PORT
    Source SHA-256 2d639443e8f81b20536e5877b73ba5029f91f5854aeed9e312ab73f1eeca65e8

The E0 fixture remains unqualified. R3 adds explicit bindings for the F09
scientific implementation while keeping empirical V11-MQ1 qualification
separate from mathematical and numerical correctness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


class RoleViolation(RuntimeError):
    """A model is used outside its frozen information or qualification role."""


class ModelRole(str, Enum):
    PLANNING = "M_P"
    EVALUATION_PARAMETER = "M_E_par"
    EVALUATION_FORM = "M_E_form"
    OBSERVATION = "M_O"
    ADHERENCE_STRESS = "M_A"


@dataclass(frozen=True)
class ModelRoleBinding:
    binding_id: str
    model_id: str
    role: ModelRole
    qualification_status: str
    allowed_scope: str
    participant_data_allowed: bool
    effect_estimation_allowed: bool
    scientific_model_gate: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["role"] = self.role.value
        return payload


E0_SYNTHETIC_PLANNING_BINDING = ModelRoleBinding(
    binding_id="WGT-V11-E0-ROLE-01",
    model_id="WGT-E0-LINEAR-ENERGY-MASS-FIXTURE",
    role=ModelRole.PLANNING,
    qualification_status="NOT_QUALIFIED_E0_CORRECTNESS_ONLY",
    allowed_scope="unit_test_fixture",
    participant_data_allowed=False,
    effect_estimation_allowed=False,
    scientific_model_gate="F09_BLOCKED_PENDING_R3_QUALIFICATION",
)

V11_MQ1_EVALUATION_PARAMETER_BINDING = ModelRoleBinding(
    binding_id="WGT-V11-MQ1-M-E-PAR-01",
    model_id="HALL2011_REDUCED_LONG_TERM_ADULT_WEIGHT_V11",
    role=ModelRole.EVALUATION_PARAMETER,
    qualification_status="PENDING_V11_MQ1_EMPIRICAL_DECISION",
    allowed_scope="A1_exposure_conditioned_visit_level_model_qualification",
    participant_data_allowed=True,
    effect_estimation_allowed=False,
    scientific_model_gate="R3_AUTHORIZED_EXACT_A1_COMMAND_PENDING",
)

V11_MQ1_PLANNING_BINDING = ModelRoleBinding(
    binding_id="WGT-V11-MQ1-M-P-01",
    model_id="HALL2011_REDUCED_LONG_TERM_ADULT_WEIGHT_V11",
    role=ModelRole.PLANNING,
    qualification_status="PENDING_V11_MQ1_EMPIRICAL_DECISION",
    allowed_scope="synthetic_known_answer_and_future_case_branch_only",
    participant_data_allowed=False,
    effect_estimation_allowed=False,
    scientific_model_gate="R3_AUTHORIZED_NOT_R4_OR_EFFECT_AUTHORITY",
)

R6_ILLUSTRATIVE_PLANNING_BINDING = ModelRoleBinding(
    binding_id="WGT-V11-R6-ILLUSTRATIVE-M-P-01",
    model_id="HALL2011_REDUCED_LONG_TERM_ADULT_WEIGHT_V11",
    role=ModelRole.PLANNING,
    qualification_status=(
        "V11_MQ1_POINT_MODEL_FAILED__ILLUSTRATIVE_ONLY"
    ),
    allowed_scope="isolated_nonformal_result_blind_engineering_pilot",
    participant_data_allowed=False,
    effect_estimation_allowed=False,
    scientific_model_gate=(
        "R6_ENGINEERING_ONLY__R7_BLOCKED_BY_R5A_SUBJECT_GENERATOR_TARGET_FREEZE"
    ),
)

R8_FORMAL_PUBLIC_E3_PLANNING_BINDING = ModelRoleBinding(
    binding_id="WGT-V11-R8-FORMAL-PUBLIC-E3-M-P-01",
    model_id="HALL2011_REDUCED_LONG_TERM_ADULT_WEIGHT_V11",
    role=ModelRole.PLANNING,
    qualification_status=(
        "V11_MQ1_POINT_MODEL_FAILED__PUBLIC_SYNTHETIC_BENCHMARK_ONLY"
    ),
    allowed_scope="r8_frozen_public_synthetic_e3_benchmark",
    participant_data_allowed=False,
    effect_estimation_allowed=True,
    scientific_model_gate=(
        "R7_RESULT_BLIND_AUTHORIZATION__NO_PARTICIPANT_OR_CLINICAL_EFFECT_CLAIM"
    ),
)


@dataclass(frozen=True)
class OuterClock:
    t_week: int
    event_k: int

    def __post_init__(self) -> None:
        if self.t_week < 0 or self.event_k < 0:
            raise RoleViolation("outer indices must be nonnegative")


@dataclass(frozen=True)
class InnerClock:
    generation_g: int
    horizon_h: int

    def __post_init__(self) -> None:
        if self.generation_g < 0 or self.horizon_h < 0:
            raise RoleViolation("inner indices must be nonnegative")


@dataclass(frozen=True)
class StateVector:
    values: tuple[float, ...]
    units: tuple[str, ...]
    observed_at_week: int

    def validate(self) -> None:
        if not self.values or len(self.values) != len(self.units):
            raise RoleViolation("state values and units must be nonempty and aligned")
        if self.observed_at_week < 0 or not np.all(np.isfinite(self.values)):
            raise RoleViolation("state must be finite and time stamped")


@dataclass(frozen=True)
class DecisionVector:
    energy_intake_kcal_day: float
    cardio_minutes_week: float
    strength_minutes_week: float

    @property
    def units(self) -> Mapping[str, str]:
        return {
            "energy_intake_kcal_day": "kcal/day",
            "cardio_minutes_week": "min/week",
            "strength_minutes_week": "min/week",
        }

    def validate_finite_only(self) -> None:
        if not np.all(
            np.isfinite(
                [
                    self.energy_intake_kcal_day,
                    self.cardio_minutes_week,
                    self.strength_minutes_week,
                ]
            )
        ):
            raise RoleViolation("decision values must be finite")


class StateTransitionModel(Protocol):
    role: ModelRole
    binding_id: str
    equation_registry_ids: Sequence[str]

    def step_week(
        self,
        state: StateVector,
        decision: DecisionVector,
        outer: OuterClock,
    ) -> StateVector: ...


@dataclass
class PlanningStateGuard:
    """Keep the observed outer state immutable throughout inner generations."""

    outer_clock: OuterClock
    initial_state: StateVector

    def __post_init__(self) -> None:
        self.initial_state.validate()
        self._fingerprint = (self.outer_clock, self.initial_state)

    def assert_inner_loop_immutable(
        self,
        outer_clock: OuterClock,
        state: StateVector,
    ) -> None:
        if (outer_clock, state) != self._fingerprint:
            raise RoleViolation(
                "inner generation changed outer time or physiological state"
            )

    def evaluate_horizon(
        self,
        model: StateTransitionModel,
        decisions: Sequence[DecisionVector],
        inner_generation: int,
    ) -> list[StateVector]:
        self.assert_inner_loop_immutable(self.outer_clock, self.initial_state)
        predicted: list[StateVector] = []
        state = self.initial_state
        for horizon, decision in enumerate(decisions):
            InnerClock(inner_generation, horizon)
            decision.validate_finite_only()
            state = model.step_week(
                state,
                decision,
                OuterClock(
                    self.outer_clock.t_week + horizon,
                    self.outer_clock.event_k,
                ),
            )
            state.validate()
            predicted.append(state)
        self.assert_inner_loop_immutable(self.outer_clock, self.initial_state)
        return predicted


def assert_optimizer_role_access(role: ModelRole) -> None:
    if role is not ModelRole.PLANNING:
        raise RoleViolation("optimizer may access the planning model role only")


def assert_information_release(
    observation_event: int,
    decision_event: int,
) -> None:
    if observation_event < 0 or decision_event < 0:
        raise RoleViolation("event indices must be nonnegative")
    if observation_event > decision_event:
        raise RoleViolation("future observation cannot be released to the optimizer")
