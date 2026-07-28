"""Result-blind R6 engineering adapter for the illustrative E3 branch.

This adapter exists to exercise interfaces, scenario routing, ledgers, state
transitions, and deterministic replay.  Its single static baseline and target
are nonformal development fixtures.  They are not the R5 E3 subject generator,
not participant data, and not effect evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from math import isfinite
from typing import Any

import numpy as np

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
    R6_ILLUSTRATIVE_PLANNING_BINDING,
    StateVector,
)
from .objectives import evaluate_weight_objectives
from .scientific_models import (
    ActivityEnergyMap,
    AdultFemaleBaseline,
    DirectEnergyExposure,
    HallLinearizedFormModel,
    HallLongTermModel,
)
from .state import WeightStateError


R6_E3_SCENARIOS = (
    "NOMINAL",
    "PARAMETER_MISMATCH_EVAL_EE_PLUS_10_PERCENT",
    "MODEL_FORM_MISMATCH_HALL_LINEARIZED_EVALUATION",
    "OBSERVATION_NOISE_GAUSSIAN_SD_0_5_KG",
    "MISSINGNESS_EVERY_FOURTH_POSTBASELINE_WEEK",
    "IMPLEMENTATION_DEVIATION_75_PERCENT_INTAKE_ACTIVITY_FREQUENCY",
    "ENERGY_SURPLUS_PLUS_250_KCAL_DAY",
    "OUT_OF_DOMAIN_STATE_FAT_50KG_LEAN_35KG",
    "INFEASIBLE_REQUIRED_DEFICIT_1500_OVER_ACTION_CAP_1000_KCAL_DAY",
)

R6_DEVELOPMENT_BASELINE = AdultFemaleBaseline(
    age_year=55.0,
    height_cm=165.0,
    weight_kg=90.0,
    background_pal=1.4,
    adult_nonpregnant_nonlactating=True,
)
R6_DEVELOPMENT_TARGET_MASS_KG = 85.0
R6_DEVELOPMENT_SEED = 6_202_607_25


def _normal_noise(seed: int, scenario: str, event_id: int) -> float:
    digest = sha256(
        f"WGT-V11-R6-NOISE-v1\0{seed}\0{scenario}\0{event_id}".encode("ascii")
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return float(rng.normal(0.0, 0.5))


def _retime(state: StateVector, week: int) -> StateVector:
    return StateVector(tuple(state.values), tuple(state.units), week)


class IllustrativeHallEngineeringAdapter:
    """Two-action, six-week-horizon adapter for isolated R6 engineering."""

    adapter_id = "WGT-V11-R6-ILLUSTRATIVE-HALL"
    adapter_version = "1.0.0-r6-engineering"
    decision_contract = SYNTHETIC_E0_DECISIONS
    decision_dimension = decision_contract.dimension
    lower_bounds = decision_contract.lower_bounds
    upper_bounds = decision_contract.upper_bounds
    atomic_steps_per_evaluation = 6
    objective_names = (
        "development_target_mass_error_kg",
        "development_intervention_burden_fraction",
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
        scenario: str,
        development_seed: int = R6_DEVELOPMENT_SEED,
        target_mass_kg: float = R6_DEVELOPMENT_TARGET_MASS_KG,
    ) -> None:
        if scenario not in R6_E3_SCENARIOS:
            raise WeightStateError("unknown R6 illustrative scenario")
        if type(development_seed) is not int or development_seed < 0:
            raise WeightStateError("development seed must be nonnegative")
        if not isfinite(target_mass_kg) or target_mass_kg <= 40.0:
            raise WeightStateError("development target must be finite and above 40 kg")

        self.scenario = scenario
        self.development_seed = development_seed
        self.target_mass_kg = float(target_mass_kg)
        self.baseline = R6_DEVELOPMENT_BASELINE
        self.model = HallLongTermModel(
            ModelRole.PLANNING,
            self.baseline,
            ActivityEnergyMap(
                cardio_net_met=5.0,
                strength_net_met=3.5,
                evidence_id="R6_ENGINEERING_DIRECT_ENERGY_MAP_NOT_EFFECT",
            ),
        )
        self.linearized_model = HallLinearizedFormModel(self.baseline)
        if scenario == "OUT_OF_DOMAIN_STATE_FAT_50KG_LEAN_35KG":
            self.state = StateVector(
                (50.0, 35.0, 0.0),
                self.model.initial_state().units,
                0,
            )
        else:
            self.state = self.model.initial_state()
        self.minimum_body_mass_kg = 40.0
        self.maximum_daily_energy_imbalance_kcal = 2000.0
        self._information: InformationSnapshot | None = None
        self._frozen_state: StateVector | None = None
        self._last_observed_state: StateVector | None = None
        self._evaluator = SharedEvaluator(
            objective_names=self.objective_names,
            constraint_names=self.constraint_names,
            evaluate_joint=self._evaluate_joint,
        )

    def identity(self) -> Mapping[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "role": "illustrative_nonformal_development_fixture",
            "scenario": self.scenario,
            "participant_data_used": False,
            "formal_subject_generator_used": False,
            "hidden_instance_used": False,
            "effect_evidence": False,
            "method_comparison_allowed": False,
            "horizon_atomic_week_steps": self.atomic_steps_per_evaluation,
            "outer_execution_week_steps": 1,
            "planning_model_binding": self.model.binding_id,
            "model_role": R6_ILLUSTRATIVE_PLANNING_BINDING.to_dict(),
            "r7_gate": (
                "BLOCKED_BY_R5A_E3_SUBJECT_GENERATOR_TARGET_FREEZE"
            ),
        }

    @staticmethod
    def _weight(state: StateVector) -> float:
        return HallLongTermModel.weight_kg(state)

    def _observed_state(self, event_id: int) -> tuple[StateVector, int, bool]:
        true_state = self.state
        if (
            self.scenario
            == "MISSINGNESS_EVERY_FOURTH_POSTBASELINE_WEEK"
            and event_id > 0
            and event_id % 4 == 0
        ):
            if self._last_observed_state is None:
                raise WeightStateError("missingness branch has no past observation")
            return _retime(self._last_observed_state, event_id), event_id - 1, False

        observed = true_state
        if self.scenario == "OBSERVATION_NOISE_GAUSSIAN_SD_0_5_KG":
            noise = _normal_noise(
                self.development_seed,
                self.scenario,
                event_id,
            )
            fat, lean, adaptive = true_state.values
            total = fat + lean
            observed_total = total + noise
            if observed_total <= 0.0:
                raise WeightStateError("noisy development observation is nonpositive")
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
        return observed, event_id, True

    def freeze_information(
        self, event_id: int, feedback: Mapping[str, Any] | None
    ) -> InformationSnapshot:
        if event_id != self.state.observed_at_week:
            raise WeightStateError("event_id must equal the current outer week")
        if event_id == 0 and feedback is not None:
            raise InformationBoundaryError(
                "initial R6 event cannot receive prior feedback"
            )
        if feedback is not None:
            released_at = feedback.get("released_at")
            if type(released_at) is not int or released_at != event_id:
                raise InformationBoundaryError(
                    "feedback must be released at the current decision week"
                )
        observed, source_week, available = self._observed_state(event_id)
        fields = {
            "current_development_observation": InformationField(
                available_at=event_id,
                value={
                    "values": tuple(observed.values),
                    "units": tuple(observed.units),
                    "source_week": source_week,
                    "observation_available": available,
                },
            ),
            "development_target_mass_kg": InformationField(
                available_at=0,
                value=self.target_mass_kg,
            ),
            "scenario_contract": InformationField(
                available_at=0,
                value={
                    "scenario": self.scenario,
                    "fixture_role": (
                        "illustrative_nonformal_development_fixture"
                    ),
                    "effect_evidence": False,
                },
            ),
            "execution_credit_contract": InformationField(
                available_at=0,
                value={
                    "loss_unit": "kg",
                    "ell_ref_definition": (
                        "absolute_pre_execution_development_target_error"
                    ),
                    "ell_exec_definition": (
                        "absolute_post_execution_development_target_error"
                    ),
                    "s_exec_definition": "development_scale_kg",
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

    def _scenario_action(self, vector: Sequence[float]) -> tuple[float, float]:
        try:
            intake_adjustment, activity_adjustment = (
                self.decision_contract.validate(vector)
            )
        except DecisionContractError as exc:
            raise WeightStateError(str(exc)) from exc
        if (
            self.scenario
            == "IMPLEMENTATION_DEVIATION_75_PERCENT_INTAKE_ACTIVITY_FREQUENCY"
        ):
            intake_adjustment *= 0.75
            activity_adjustment *= 0.75
        if self.scenario == "ENERGY_SURPLUS_PLUS_250_KCAL_DAY":
            intake_adjustment += 250.0
        if (
            self.scenario
            == "PARAMETER_MISMATCH_EVAL_EE_PLUS_10_PERCENT"
        ):
            activity_adjustment += (
                0.10 * self.model.baseline_energy_intake_kcal_day
            )
        return intake_adjustment, activity_adjustment

    def _project(
        self,
        state: StateVector,
        vector: Sequence[float],
        *,
        weeks: int,
    ) -> StateVector:
        intake_adjustment, activity_adjustment = self._scenario_action(vector)
        if (
            self.scenario
            == "MODEL_FORM_MISMATCH_HALL_LINEARIZED_EVALUATION"
        ):
            linear = StateVector(
                (self._weight(state),),
                self.linearized_model.units,
                state.observed_at_week,
            )
            for offset in range(weeks):
                effective_intake = (
                    self.linearized_model.baseline_energy_intake_kcal_day
                    + intake_adjustment
                    - activity_adjustment
                )
                linear = self.linearized_model.step_week(
                    linear,
                    DecisionVector(effective_intake, 0.0, 0.0),
                    OuterClock(state.observed_at_week + offset, 0),
                )
            fat, lean, adaptive = state.values
            ratio = fat / (fat + lean)
            total = linear.values[0]
            return StateVector(
                (total * ratio, total * (1.0 - ratio), adaptive),
                state.units,
                state.observed_at_week + weeks,
            )

        projected = state
        exposure = DirectEnergyExposure(
            energy_intake_kcal_day=(
                self.model.baseline_energy_intake_kcal_day
                + intake_adjustment
            ),
            activity_change_kcal_day=activity_adjustment,
        )
        for _ in range(weeks):
            advanced = self.model.advance_days_exposure(
                projected,
                exposure,
                7.0,
            )
            projected = _retime(
                advanced,
                projected.observed_at_week + 1,
            )
        return projected

    def _evaluate_joint(
        self,
        vector: Sequence[float],
        information: InformationSnapshot,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if information is not self._information or self._frozen_state is None:
            raise WeightStateError("joint evaluator is not bound to this event")
        predicted = self._project(
            self._frozen_state,
            vector,
            weeks=self.atomic_steps_per_evaluation,
        )
        intake_adjustment, activity_adjustment = self._scenario_action(vector)
        body_mass = self._weight(predicted)
        required_gap = (
            intake_adjustment + 1500.0
            if self.scenario
            == "INFEASIBLE_REQUIRED_DEFICIT_1500_OVER_ACTION_CAP_1000_KCAL_DAY"
            else -1.0
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
            origin="r6_illustrative_joint_evaluator",
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

    def first_action(self, vector: Sequence[float]) -> np.ndarray:
        return self.shift_solution(vector)

    def fallback_action(self, event_id: int) -> np.ndarray:
        if event_id != self.state.observed_at_week:
            raise WeightStateError("fallback event differs from the outer week")
        return self.decision_contract.neutral_action()

    def execute(
        self,
        action: Sequence[float],
        event_id: int,
        committed: bool,
        ledger: EvaluationLedger,
    ) -> Mapping[str, Any]:
        if event_id != self.state.observed_at_week:
            raise WeightStateError("execution event differs from the outer week")
        before = self.state
        after = self._project(before, action, weeks=1)
        self.state = after
        self._information = None
        self._frozen_state = None
        ledger.record_execution()
        intake_adjustment, activity_adjustment = self._scenario_action(action)
        hard_violation = (
            self._weight(after) < self.minimum_body_mass_kg
            or abs(intake_adjustment - activity_adjustment)
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
            "development_state_after": {
                "values": tuple(after.values),
                "units": tuple(after.units),
                "observed_at_week": after.observed_at_week,
            },
            "fixture_role": "R6_ENGINEERING_ONLY_NOT_EFFECT",
        }
