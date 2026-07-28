"""N01 event adapter for an explicitly synthetic R2 weight fixture.

Provenance disposition:
    NEW_V11/N01 NEW_IMPLEMENTATION
    BUILD_FROM_EVENT_INTERFACE_AND_MODEL_ROLE_CONTRACTS

No legacy metabolic implementation, participant data, virtual cohort, or
unqualified scientific model is imported by this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any

import numpy as np

from evaluation.contracts import EvaluationResult
from evaluation.evaluator import SharedEvaluator
from evaluation.firewall import (
    PROHIBITED_FIELDS,
    InformationBoundaryError,
    InformationField,
    InformationSnapshot,
    freeze_information,
)
from evaluation.ledger import EvaluationLedger

from .constraints import (
    SYNTHETIC_E0_CONSTRAINTS,
    evaluate_weight_constraints,
)
from .decisions import (
    DecisionContractError,
    SYNTHETIC_E0_DECISIONS,
)
from .model_roles import E0_SYNTHETIC_PLANNING_BINDING
from .objectives import (
    SYNTHETIC_E0_OBJECTIVES,
    evaluate_weight_objectives,
)
from .state import (
    SyntheticWeightModel,
    SyntheticWeightState,
    WeightStateError,
)


def _find_prohibited_key(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key)
            if name in PROHIBITED_FIELDS:
                return name
            nested = _find_prohibited_key(item)
            if nested is not None:
                return nested
    elif isinstance(value, list | tuple):
        for item in value:
            nested = _find_prohibited_key(item)
            if nested is not None:
                return nested
    return None


class SyntheticWeightAdapter:
    """Same event interface as public benchmarks, restricted to E0 fixtures."""

    adapter_id = "WGT-V11-SYNTHETIC-E0"
    adapter_version = "1.1.0-r2-fixture"
    decision_contract = SYNTHETIC_E0_DECISIONS
    model_role_binding = E0_SYNTHETIC_PLANNING_BINDING
    decision_dimension = decision_contract.dimension
    atomic_steps_per_evaluation = 1
    lower_bounds = decision_contract.lower_bounds
    upper_bounds = decision_contract.upper_bounds
    objective_names = SYNTHETIC_E0_OBJECTIVES.names
    constraint_names = SYNTHETIC_E0_CONSTRAINTS.names
    constraint_scales = (1.0, 1.0)

    def __init__(
        self,
        *,
        initial_state: SyntheticWeightState,
        target_mass_kg: float,
        model: SyntheticWeightModel,
        minimum_body_mass_kg: float = 40.0,
        maximum_daily_energy_imbalance_kcal: float = 1500.0,
    ) -> None:
        if not isinstance(initial_state, SyntheticWeightState):
            raise WeightStateError(
                "initial_state must be a SyntheticWeightState"
            )
        if not isinstance(model, SyntheticWeightModel):
            raise WeightStateError("model must be a SyntheticWeightModel")
        numeric = (
            target_mass_kg,
            minimum_body_mass_kg,
            maximum_daily_energy_imbalance_kcal,
        )
        if not all(isfinite(value) and value > 0.0 for value in numeric):
            raise WeightStateError(
                "weight target and safety thresholds must be positive and finite"
            )
        if target_mass_kg < minimum_body_mass_kg:
            raise WeightStateError(
                "target mass must not be below the minimum body mass"
            )

        self.state = initial_state
        self.target_mass_kg = float(target_mass_kg)
        self.model = model
        self.minimum_body_mass_kg = float(minimum_body_mass_kg)
        self.maximum_daily_energy_imbalance_kcal = float(
            maximum_daily_energy_imbalance_kcal
        )
        self.execution_loss_scale_kg = (
            self.model.event_days
            * self.maximum_daily_energy_imbalance_kcal
            / self.model.energy_density_kcal_per_kg
        )
        self.constraint_scales = (
            1.0,
            self.maximum_daily_energy_imbalance_kcal,
            1.0,
            1.0,
        )
        self._information: InformationSnapshot | None = None
        self._frozen_state: SyntheticWeightState | None = None
        self._evaluator = SharedEvaluator(
            objective_names=self.objective_names,
            constraint_names=self.constraint_names,
            evaluate_joint=self._evaluate_joint,
        )

    def identity(self) -> Mapping[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "role": "supportive_synthetic_E0_correctness_fixture",
            "participant_data_used": False,
            "virtual_human_claim": False,
            "effect_evidence": False,
            "model": dict(self.model.identity()),
            "model_role": self.model_role_binding.to_dict(),
        }

    def freeze_information(
        self, event_id: int, feedback: Mapping[str, Any] | None
    ) -> InformationSnapshot:
        if event_id != self.state.event_id:
            raise WeightStateError(
                "event_id must match the current state event"
            )
        prohibited = _find_prohibited_key(feedback)
        if prohibited is not None:
            raise InformationBoundaryError(
                f"prohibited information field: {prohibited}"
            )
        if event_id == 0 and feedback is not None:
            raise WeightStateError(
                "the initial event cannot receive prior feedback"
            )
        if feedback is not None:
            released_at = feedback.get("released_at")
            if type(released_at) is not int or released_at != event_id:
                raise InformationBoundaryError(
                    "prior feedback must be explicitly released at the "
                    "current integer decision time"
                )
        fields = {
            "current_synthetic_weight_state": InformationField(
                available_at=event_id,
                value=self.state.to_dict(),
            ),
            "current_target_mass_kg": InformationField(
                available_at=event_id,
                value=self.target_mass_kg,
            ),
            "frozen_model_role_and_units": InformationField(
                available_at=0,
                value={
                    "model": dict(self.model.identity()),
                    "role_binding": self.model_role_binding.to_dict(),
                },
            ),
            "safety_thresholds": InformationField(
                available_at=0,
                value={
                    "minimum_body_mass_kg": self.minimum_body_mass_kg,
                    "maximum_daily_energy_imbalance_kcal": (
                        self.maximum_daily_energy_imbalance_kcal
                    ),
                },
            ),
            "execution_credit_contract": InformationField(
                available_at=0,
                value={
                    "loss_unit": "kg",
                    "ell_ref_definition": (
                        "absolute_pre_execution_target_mass_error_kg"
                    ),
                    "ell_exec_definition": (
                        "absolute_first_post_execution_target_mass_error_kg"
                    ),
                    "s_exec_definition": (
                        "maximum_allowed_single_event_mass_change_kg"
                    ),
                    "s_exec_kg": self.execution_loss_scale_kg,
                },
            ),
        }
        if feedback is not None:
            fields["prior_execution_feedback"] = InformationField(
                available_at=event_id,
                value=dict(feedback),
            )
        self._information = freeze_information(
            decision_time=event_id,
            fields=fields,
        )
        self._frozen_state = self.state
        return self._information

    def _evaluate_joint(
        self,
        vector: Sequence[float],
        information: InformationSnapshot,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if information is not self._information:
            raise WeightStateError(
                "joint evaluator received an unbound information snapshot"
            )
        frozen = information.fields[
            "current_synthetic_weight_state"
        ].value
        state = SyntheticWeightState.from_mapping(frozen)
        predicted = self.model.project(state, vector)
        intake_adjustment, activity_adjustment = (
            float(value) for value in vector
        )
        daily_energy_imbalance = (
            intake_adjustment - activity_adjustment
        )
        return (
            evaluate_weight_objectives(
                predicted_body_mass_kg=predicted.body_mass_kg,
                target_mass_kg=self.target_mass_kg,
                intake_adjustment_kcal_per_day=intake_adjustment,
                activity_expenditure_adjustment_kcal_per_day=(
                    activity_adjustment
                ),
            ),
            evaluate_weight_constraints(
                predicted_body_mass_kg=predicted.body_mass_kg,
                predicted_fat_mass_kg=predicted.fat_mass_kg,
                predicted_lean_mass_kg=predicted.lean_mass_kg,
                minimum_body_mass_kg=self.minimum_body_mass_kg,
                daily_energy_imbalance_kcal=daily_energy_imbalance,
                maximum_daily_energy_imbalance_kcal=(
                    self.maximum_daily_energy_imbalance_kcal
                ),
            ),
        )

    def evaluate(
        self,
        vector: Sequence[float],
        event_id: int,
        ledger: EvaluationLedger,
        candidate_id: str,
    ) -> EvaluationResult:
        information = self._information
        if (
            information is None
            or information.decision_time != event_id
            or self._frozen_state is None
        ):
            raise WeightStateError(
                "freeze_information must bind the current event before evaluate"
            )
        if self.state != self._frozen_state:
            raise WeightStateError(
                "outer state changed during inner-generation evaluation"
            )
        return self._evaluator.evaluate(
            vector=vector,
            event_id=event_id,
            candidate_id=candidate_id,
            information=information,
            ledger=ledger,
            atomic_steps=self.atomic_steps_per_evaluation,
            origin="synthetic_weight_joint_evaluator",
        )

    @staticmethod
    def safety_filter(result: EvaluationResult, event_id: int) -> bool:
        del event_id
        return result.feasible

    def project_state(
        self,
        action: Sequence[float],
        *,
        state: SyntheticWeightState | None = None,
    ) -> SyntheticWeightState:
        try:
            values = self.decision_contract.validate(action)
        except DecisionContractError as exc:
            raise WeightStateError(str(exc)) from exc
        return self.model.transition(state or self.state, values)

    def shift_solution(self, vector: Sequence[float]) -> np.ndarray:
        try:
            values = self.decision_contract._array(vector)
        except DecisionContractError as exc:
            raise WeightStateError("shift input has wrong shape") from exc
        return values.copy()

    def first_action(self, vector: Sequence[float]) -> np.ndarray:
        return self.shift_solution(vector)

    def fallback_action(self, event_id: int) -> np.ndarray:
        if event_id != self.state.event_id:
            raise WeightStateError(
                "fallback event_id must match the current state"
            )
        return self.decision_contract.neutral_action()

    def execute(
        self,
        action: Sequence[float],
        event_id: int,
        committed: bool,
        ledger: EvaluationLedger,
    ) -> Mapping[str, Any]:
        if event_id != self.state.event_id:
            raise WeightStateError(
                "execution event_id must match the current state"
            )
        before = self.state
        after = self.project_state(action, state=before)
        intake_adjustment, activity_adjustment = (
            float(value) for value in action
        )
        daily_energy_imbalance = (
            intake_adjustment - activity_adjustment
        )
        energy_imbalance = self.model.event_days * daily_energy_imbalance
        hard_violation = (
            after.body_mass_kg < self.minimum_body_mass_kg
            or abs(daily_energy_imbalance)
            > self.maximum_daily_energy_imbalance_kcal
        )
        self.state = after
        self._information = None
        self._frozen_state = None
        ledger.record_execution()
        return {
            "available": True,
            "committed": bool(committed),
            "ell_exec": abs(after.body_mass_kg - self.target_mass_kg),
            "ell_ref": abs(before.body_mass_kg - self.target_mass_kg),
            "s_exec": self.execution_loss_scale_kg,
            "hard_constraint_violation": hard_violation,
            "released_at": event_id + 1,
            "energy_imbalance_kcal": energy_imbalance,
            "state_after": after.to_dict(),
            "fixture_role": "E0_CORRECTNESS_ONLY",
        }
