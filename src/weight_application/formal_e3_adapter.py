"""Frozen public-synthetic E3 adapter for the R8 formal execution.

All planning evaluations use the same nominal nonlinear Hall model.  The R5a
scenario rules affect only their frozen layer: initial state, observation,
feasibility, execution transform, or evaluation transition.  No participant
or hidden-instance data enter this adapter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

import numpy as np

from e3_inputs.contract import (
    E3_SCENARIOS,
    E3SubjectParameters,
    OOD_ADAPTIVE_THERMOGENESIS_KCAL_DAY,
    OOD_FAT_MASS_KG,
    OOD_LEAN_MASS_KG,
    apply_execution_transform,
    observation_is_missing,
    paired_observation_noise_kg,
    parameter_mismatch_ee_offset_kcal_day,
    required_intake_deficit_constraint,
    target_mass_kg,
)
from evaluation.contracts import EvaluationResult
from evaluation.evaluator import SharedEvaluator
from evaluation.firewall import (
    InformationBoundaryError,
    InformationField,
    InformationSnapshot,
    freeze_information,
)
from evaluation.ledger import EvaluationLedger

from .decisions import DecisionContractError, SYNTHETIC_E0_DECISIONS
from .model_roles import (
    DecisionVector,
    ModelRole,
    OuterClock,
    R8_FORMAL_PUBLIC_E3_PLANNING_BINDING,
    StateVector,
)
from .objectives import evaluate_weight_objectives
from .scientific_models import (
    ActivityEnergyMap,
    DirectEnergyExposure,
    HallLinearizedFormModel,
    HallLongTermModel,
)
from .state import WeightStateError


def _retime(state: StateVector, week: int) -> StateVector:
    return StateVector(tuple(state.values), tuple(state.units), week)


class FormalHallE3Adapter:
    """Two-action, six-week-horizon adapter bound to the R5a E3 contract."""

    adapter_id = "WGT-V11-R8-FORMAL-PUBLIC-HALL-E3"
    adapter_version = "1.0.0-r7-frozen"
    decision_contract = SYNTHETIC_E0_DECISIONS
    decision_dimension = decision_contract.dimension
    lower_bounds = decision_contract.lower_bounds
    upper_bounds = decision_contract.upper_bounds
    atomic_steps_per_evaluation = 6
    objective_names = (
        "formal_public_target_mass_error_kg",
        "formal_public_intervention_burden_fraction",
    )
    constraint_names = (
        "minimum_body_mass",
        "maximum_daily_energy_imbalance",
        "positive_fat_mass",
        "positive_lean_mass",
        "required_intake_deficit_gap",
    )
    constraint_scales = (1.0, 2000.0, 1.0, 1.0, 1500.0)

    def __init__(
        self,
        *,
        subject: E3SubjectParameters,
        scenario: str,
        replicate_index: int,
        paired_master_seed_u64: int,
    ) -> None:
        subject.validate()
        if scenario not in E3_SCENARIOS:
            raise WeightStateError("unknown frozen E3 scenario")
        if type(replicate_index) is not int or replicate_index not in range(3):
            raise WeightStateError("E3 replicate_index must be in 0..2")
        if type(paired_master_seed_u64) is not int or not (
            0 <= paired_master_seed_u64 < 1 << 64
        ):
            raise WeightStateError(
                "paired_master_seed_u64 must be an unsigned 64-bit integer"
            )

        self.subject = subject
        self.scenario = scenario
        self.replicate_index = replicate_index
        self.paired_master_seed_u64 = paired_master_seed_u64
        self.baseline = subject.to_baseline()
        activity_map = ActivityEnergyMap(
            cardio_net_met=5.0,
            strength_net_met=3.5,
            evidence_id="R3_FROZEN_DIRECT_ENERGY_ACTION_MAP",
        )
        self.planning_model = HallLongTermModel(
            ModelRole.PLANNING,
            self.baseline,
            activity_map,
        )
        self.evaluation_parameter_model = HallLongTermModel(
            ModelRole.EVALUATION_PARAMETER,
            self.baseline,
            activity_map,
        )
        self.evaluation_form_model = HallLinearizedFormModel(self.baseline)
        if scenario == "OUT_OF_DOMAIN_STATE_FAT_50KG_LEAN_35KG":
            self.state = StateVector(
                (
                    float(OOD_FAT_MASS_KG),
                    float(OOD_LEAN_MASS_KG),
                    float(OOD_ADAPTIVE_THERMOGENESIS_KCAL_DAY),
                ),
                self.planning_model.initial_state().units,
                0,
            )
        else:
            self.state = self.planning_model.initial_state()
        self.target_mass_kg = float(target_mass_kg(subject, scenario))
        self.minimum_body_mass_kg = 40.0
        self.maximum_daily_energy_imbalance_kcal = 2000.0
        self._information: InformationSnapshot | None = None
        self._frozen_state: StateVector | None = None
        self._last_observed_state: StateVector | None = None
        self._last_observed_week: int | None = None
        self._evaluator = SharedEvaluator(
            objective_names=self.objective_names,
            constraint_names=self.constraint_names,
            evaluate_joint=self._evaluate_joint,
        )

    def identity(self) -> Mapping[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "role": "frozen_formal_public_synthetic_e3_benchmark",
            "subject_id": self.subject.subject_id,
            "subject_seed_u64": str(self.subject.seed_u64),
            "scenario": self.scenario,
            "replicate_index": self.replicate_index,
            "paired_master_seed_u64": str(self.paired_master_seed_u64),
            "participant_data_used": False,
            "formal_subject_generator_used": True,
            "hidden_instance_used": False,
            "benchmark_effect_evidence": True,
            "participant_or_clinical_effect_evidence": False,
            "horizon_atomic_week_steps": self.atomic_steps_per_evaluation,
            "outer_execution_week_steps": 1,
            "planning_model_binding": self.planning_model.binding_id,
            "evaluation_parameter_model_binding": (
                self.evaluation_parameter_model.binding_id
            ),
            "evaluation_form_model_binding": (
                self.evaluation_form_model.binding_id
            ),
            "model_role": R8_FORMAL_PUBLIC_E3_PLANNING_BINDING.to_dict(),
            "planning_scenario_transformations": "PROHIBITED",
            "r5a_scenario_layering": "ENFORCED",
        }

    @staticmethod
    def _weight(state: StateVector) -> float:
        return HallLongTermModel.weight_kg(state)

    def _observed_state(self, event_id: int) -> tuple[StateVector, int, bool]:
        true_state = self.state
        if observation_is_missing(self.scenario, event_id):
            if (
                self._last_observed_state is None
                or self._last_observed_week is None
            ):
                raise WeightStateError("missingness has no prior observation")
            return (
                _retime(self._last_observed_state, event_id),
                self._last_observed_week,
                False,
            )

        observed = true_state
        if self.scenario == "OBSERVATION_NOISE_GAUSSIAN_SD_0_5_KG":
            noise = paired_observation_noise_kg(
                paired_master_seed_u64=self.paired_master_seed_u64,
                subject_seed_u64=self.subject.seed_u64,
                scenario_id=self.scenario,
                replicate_index=self.replicate_index,
                decision_week=event_id,
            )
            fat, lean, adaptive = true_state.values
            total = fat + lean
            observed_total = total + float(noise)
            if observed_total <= 0.0:
                raise WeightStateError("noisy formal observation is nonpositive")
            observed = StateVector(
                (
                    observed_total * fat / total,
                    observed_total * lean / total,
                    adaptive,
                ),
                true_state.units,
                event_id,
            )
        self._last_observed_state = observed
        self._last_observed_week = event_id
        return observed, event_id, True

    def freeze_information(
        self,
        event_id: int,
        feedback: Mapping[str, Any] | None,
    ) -> InformationSnapshot:
        if event_id != self.state.observed_at_week or event_id not in range(26):
            raise WeightStateError(
                "event_id must be the current frozen E3 decision week"
            )
        if event_id == 0 and feedback is not None:
            raise InformationBoundaryError(
                "initial E3 event cannot receive prior feedback"
            )
        if feedback is not None:
            released_at = feedback.get("released_at")
            if type(released_at) is not int or released_at != event_id:
                raise InformationBoundaryError(
                    "feedback must be released at the current decision week"
                )
        observed, source_week, available = self._observed_state(event_id)
        fields = {
            "current_public_synthetic_observation": InformationField(
                available_at=event_id,
                value={
                    "values": tuple(observed.values),
                    "units": tuple(observed.units),
                    "source_week": source_week,
                    "observation_available": available,
                },
            ),
            "formal_target_mass_kg": InformationField(
                available_at=0,
                value=self.target_mass_kg,
            ),
            "formal_scenario_contract": InformationField(
                available_at=0,
                value={
                    "scenario": self.scenario,
                    "planning_model_layer": (
                        "M_P_HALL_NONLINEAR_NOMINAL"
                    ),
                    "participant_data_used": False,
                    "hidden_instance_used": False,
                },
            ),
            "execution_credit_contract": InformationField(
                available_at=0,
                value={
                    "loss_unit": "kg",
                    "ell_ref_definition": (
                        "absolute_pre_execution_formal_target_error"
                    ),
                    "ell_exec_definition": (
                        "absolute_post_execution_formal_target_error"
                    ),
                    "s_exec_definition": "formal_scale_kg",
                    "s_exec_kg": 1.0,
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
        self._frozen_state = observed
        return self._information

    def _validated_action(
        self,
        vector: Sequence[float],
    ) -> tuple[float, float]:
        try:
            return self.decision_contract.validate(vector)
        except DecisionContractError as exc:
            raise WeightStateError(str(exc)) from exc

    def _planning_project(
        self,
        state: StateVector,
        vector: Sequence[float],
        *,
        weeks: int,
    ) -> StateVector:
        """Project candidates with nominal M_P only, for every scenario."""

        intake_adjustment, activity_adjustment = self._validated_action(vector)
        projected = state
        exposure = DirectEnergyExposure(
            energy_intake_kcal_day=(
                self.planning_model.baseline_energy_intake_kcal_day
                + intake_adjustment
            ),
            activity_change_kcal_day=activity_adjustment,
        )
        for _ in range(weeks):
            advanced = self.planning_model.advance_days_exposure(
                projected,
                exposure,
                7.0,
            )
            projected = _retime(
                advanced,
                projected.observed_at_week + 1,
            )
        return projected

    def _evaluation_step(
        self,
        state: StateVector,
        *,
        intake_adjustment: float,
        activity_adjustment: float,
    ) -> StateVector:
        if self.scenario == (
            "MODEL_FORM_MISMATCH_HALL_LINEARIZED_EVALUATION"
        ):
            linear_state = StateVector(
                (self._weight(state),),
                self.evaluation_form_model.units,
                state.observed_at_week,
            )
            effective_intake = (
                self.evaluation_form_model.baseline_energy_intake_kcal_day
                + intake_adjustment
                - activity_adjustment
            )
            linear_after = self.evaluation_form_model.step_week(
                linear_state,
                DecisionVector(effective_intake, 0.0, 0.0),
                OuterClock(state.observed_at_week, 0),
            )
            fat, lean, adaptive = state.values
            fat_fraction = fat / (fat + lean)
            total = linear_after.values[0]
            return StateVector(
                (
                    total * fat_fraction,
                    total * (1.0 - fat_fraction),
                    adaptive,
                ),
                state.units,
                state.observed_at_week + 1,
            )

        parameter_offset = float(
            parameter_mismatch_ee_offset_kcal_day(
                self.subject,
                self.scenario,
            )
        )
        exposure = DirectEnergyExposure(
            energy_intake_kcal_day=(
                self.evaluation_parameter_model.baseline_energy_intake_kcal_day
                + intake_adjustment
            ),
            activity_change_kcal_day=(
                activity_adjustment + parameter_offset
            ),
        )
        after = self.evaluation_parameter_model.advance_days_exposure(
            state,
            exposure,
            7.0,
        )
        return _retime(after, state.observed_at_week + 1)

    def _evaluate_joint(
        self,
        vector: Sequence[float],
        information: InformationSnapshot,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if information is not self._information or self._frozen_state is None:
            raise WeightStateError("joint evaluator is not bound to this event")
        predicted = self._planning_project(
            self._frozen_state,
            vector,
            weeks=self.atomic_steps_per_evaluation,
        )
        intake_adjustment, activity_adjustment = self._validated_action(vector)
        body_mass = self._weight(predicted)
        required_gap = float(
            required_intake_deficit_constraint(
                self.scenario,
                Decimal(str(intake_adjustment)),
            )
        )
        return (
            evaluate_weight_objectives(
                predicted_body_mass_kg=body_mass,
                target_mass_kg=self.target_mass_kg,
                intake_adjustment_kcal_per_day=intake_adjustment,
                activity_expenditure_adjustment_kcal_per_day=(
                    activity_adjustment
                ),
            ),
            (
                self.minimum_body_mass_kg - body_mass,
                abs(intake_adjustment - activity_adjustment)
                - self.maximum_daily_energy_imbalance_kcal,
                -predicted.values[0],
                -predicted.values[1],
                required_gap,
            ),
        )

    def evaluate(
        self,
        vector: Sequence[float],
        event_id: int,
        ledger: EvaluationLedger,
        candidate_id: str,
    ) -> EvaluationResult:
        if (
            self._information is None
            or self._information.decision_time != event_id
            or self._frozen_state is None
        ):
            raise WeightStateError("freeze_information must precede evaluate")
        return self._evaluator.evaluate(
            vector=vector,
            event_id=event_id,
            candidate_id=candidate_id,
            information=self._information,
            ledger=ledger,
            atomic_steps=self.atomic_steps_per_evaluation,
            origin="r8_formal_public_e3_joint_evaluator",
        )

    @staticmethod
    def safety_filter(result: EvaluationResult, event_id: int) -> bool:
        del event_id
        return result.feasible

    def shift_solution(self, vector: Sequence[float]) -> np.ndarray:
        try:
            return self.decision_contract._array(vector).copy()
        except DecisionContractError as exc:
            raise WeightStateError(str(exc)) from exc

    @staticmethod
    def select_candidate(candidates: Sequence[Any]) -> Any:
        values = tuple(candidates)
        if not values:
            raise WeightStateError("formal E3 selector requires candidates")
        return min(
            values,
            key=lambda candidate: (
                candidate.evaluation.objectives[0],
                candidate.evaluation.objectives[1],
                candidate.candidate_id,
            ),
        )

    def first_action(self, vector: Sequence[float]) -> np.ndarray:
        return self.shift_solution(vector)

    def fallback_action(self, event_id: int) -> np.ndarray:
        if event_id != self.state.observed_at_week:
            raise WeightStateError("fallback event differs from outer week")
        return self.decision_contract.neutral_action()

    def execute(
        self,
        action: Sequence[float],
        event_id: int,
        committed: bool,
        ledger: EvaluationLedger,
    ) -> Mapping[str, Any]:
        if event_id != self.state.observed_at_week:
            raise WeightStateError("execution event differs from outer week")
        planned_intake, planned_activity = self._validated_action(action)
        executed_intake, executed_activity = apply_execution_transform(
            self.scenario,
            intake_adjustment_kcal_day=Decimal(str(planned_intake)),
            activity_adjustment_kcal_day=Decimal(str(planned_activity)),
        )
        before = self.state
        after = self._evaluation_step(
            before,
            intake_adjustment=float(executed_intake),
            activity_adjustment=float(executed_activity),
        )
        self.state = after
        self._information = None
        self._frozen_state = None
        ledger.record_execution()
        hard_violation = (
            self._weight(after) < self.minimum_body_mass_kg
            or abs(float(executed_intake) - float(executed_activity))
            > self.maximum_daily_energy_imbalance_kcal
            or after.values[0] <= 0.0
            or after.values[1] <= 0.0
        )
        return {
            "available": True,
            "committed": bool(committed),
            "ell_exec": abs(self._weight(after) - self.target_mass_kg),
            "ell_ref": abs(self._weight(before) - self.target_mass_kg),
            "s_exec": 1.0,
            "hard_constraint_violation": hard_violation,
            "released_at": event_id + 1,
            "formal_state_after": {
                "values": tuple(after.values),
                "units": tuple(after.units),
                "observed_at_week": after.observed_at_week,
            },
            "planned_action": {
                "intake_adjustment_kcal_day": planned_intake,
                "activity_adjustment_kcal_day": planned_activity,
            },
            "executed_action": {
                "intake_adjustment_kcal_day": float(executed_intake),
                "activity_adjustment_kcal_day": float(executed_activity),
            },
            "scenario": self.scenario,
            "fixture_role": "R8_FORMAL_PUBLIC_SYNTHETIC_E3",
        }
