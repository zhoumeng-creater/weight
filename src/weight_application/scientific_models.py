"""R3-bound scientific models for long-term adult body-weight dynamics.

Provenance:
    FORMAL_V1/dt_ramde_formal/scientific_models.py -> F09 conditional port
    Source SHA-256 0074ba6e8dc48c7a760d4bc14d8dbad5396276c6a867e633efff0f79d1b49021

The implementation is a reduced Hall et al. (2011) model. Glycogen and
extracellular-fluid dynamics are deliberately held fixed, so it is for
multi-week energy-balance trajectories, not acute fasting/refeeding or
short-term fluid claims. All internal energy units are kcal and days.
Mathematical correctness does not imply V11-MQ1 empirical qualification.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log
from typing import Mapping, Sequence

import numpy as np

from .model_roles import (
    DecisionVector,
    ModelRole,
    OuterClock,
    RoleViolation,
    StateVector,
    assert_information_release,
)


KJ_PER_KCAL = 4.184
HALL_STATE_UNITS = ("kg", "kg", "kcal/day")


@dataclass(frozen=True)
class HallConstants:
    """Published Hall appendix constants converted once from kJ/MJ to kcal."""

    rho_f_kcal_kg: float = 39_500.0 / KJ_PER_KCAL
    rho_l_kcal_kg: float = 7_600.0 / KJ_PER_KCAL
    gamma_f_kcal_kg_day: float = 13.0 / KJ_PER_KCAL
    gamma_l_kcal_kg_day: float = 92.0 / KJ_PER_KCAL
    eta_f_kcal_kg: float = 750.0 / KJ_PER_KCAL
    eta_l_kcal_kg: float = 960.0 / KJ_PER_KCAL
    beta_tef: float = 0.10
    beta_at: float = 0.14
    tau_at_day: float = 14.0
    forbes_c_kg: float = 10.4

    def validate(self) -> None:
        positive = (
            self.rho_f_kcal_kg,
            self.rho_l_kcal_kg,
            self.gamma_f_kcal_kg_day,
            self.gamma_l_kcal_kg_day,
            self.eta_f_kcal_kg,
            self.eta_l_kcal_kg,
            self.tau_at_day,
            self.forbes_c_kg,
        )
        if not all(isfinite(value) and value > 0.0 for value in positive):
            raise RoleViolation("Hall constants must be finite and positive")
        if not (
            0.0 <= self.beta_tef < 1.0 and 0.0 <= self.beta_at < 1.0
        ):
            raise RoleViolation("thermic/adaptive fractions must be in [0, 1)")


@dataclass(frozen=True)
class AdultFemaleBaseline:
    """Explicit PRIDE-context baseline; PAL excludes decision exercise."""

    age_year: float
    height_cm: float
    weight_kg: float
    background_pal: float
    adult_nonpregnant_nonlactating: bool

    def validate(self) -> None:
        values = (
            self.age_year,
            self.height_cm,
            self.weight_kg,
            self.background_pal,
        )
        if not all(isfinite(value) for value in values):
            raise RoleViolation("baseline values must be finite")
        bmi = self.weight_kg / (self.height_cm / 100.0) ** 2
        if not (30.0 <= self.age_year <= 78.0 and 25.0 <= bmi <= 50.0):
            raise RoleViolation(
                "baseline is outside the frozen PRIDE context of use"
            )
        if not (
            130.0 <= self.height_cm <= 200.0
            and 45.0 <= self.weight_kg <= 250.0
        ):
            raise RoleViolation(
                "height/weight is outside the numerical safety envelope"
            )
        if not (1.0 <= self.background_pal <= 2.5):
            raise RoleViolation("background PAL must be in [1.0, 2.5]")
        if not self.adult_nonpregnant_nonlactating:
            raise RoleViolation("model excludes minors, pregnancy, and lactation")

    @property
    def bmi_kg_m2(self) -> float:
        return self.weight_kg / (self.height_cm / 100.0) ** 2

    @property
    def resting_metabolic_rate_kcal_day(self) -> float:
        # Mifflin et al. (1990), women.
        return (
            10.0 * self.weight_kg
            + 6.25 * self.height_cm
            - 5.0 * self.age_year
            - 161.0
        )

    @property
    def fat_mass_kg(self) -> float:
        # Hall et al. web appendix Eq. 4, female branch.
        fat_percent = (
            0.14 * self.age_year + 39.96 * log(self.bmi_kg_m2) - 102.01
        )
        fat = self.weight_kg * fat_percent / 100.0
        if not (0.0 < fat < self.weight_kg):
            raise RoleViolation("Eq. 4 produced an invalid baseline fat mass")
        return fat

    @property
    def lean_mass_kg(self) -> float:
        return self.weight_kg - self.fat_mass_kg


@dataclass(frozen=True)
class ActivityEnergyMap:
    """Explicit net-MET conversion; no undocumented intensity default exists."""

    cardio_net_met: float
    strength_net_met: float
    evidence_id: str

    def validate(self) -> None:
        if not self.evidence_id or self.evidence_id == "PENDING":
            raise RoleViolation(
                "activity intensity evidence id must be explicit"
            )
        if not all(
            isfinite(value) and value >= 0.0
            for value in (self.cardio_net_met, self.strength_net_met)
        ):
            raise RoleViolation("net MET values must be finite and nonnegative")

    def kcal_day(self, weight_kg: float, decision: DecisionVector) -> float:
        """ACSM MET identity: kcal/min = MET * 3.5 * kg / 200."""

        self.validate()
        decision.validate_finite_only()
        if (
            decision.cardio_minutes_week < 0.0
            or decision.strength_minutes_week < 0.0
        ):
            raise RoleViolation("exercise minutes cannot be negative")
        met_minutes_week = (
            self.cardio_net_met * decision.cardio_minutes_week
            + self.strength_net_met * decision.strength_minutes_week
        )
        return met_minutes_week * 3.5 * weight_kg / 200.0 / 7.0


@dataclass(frozen=True)
class DirectEnergyExposure:
    """Source-native interval exposure for result-blind model qualification."""

    energy_intake_kcal_day: float
    activity_change_kcal_day: float

    def validate(self) -> None:
        if (
            not isfinite(self.energy_intake_kcal_day)
            or self.energy_intake_kcal_day <= 0.0
        ):
            raise RoleViolation(
                "direct energy intake must be finite and positive"
            )
        if not isfinite(self.activity_change_kcal_day):
            raise RoleViolation("direct activity change must be finite")


class HallLongTermModel:
    """Nonlinear fat/lean/adaptive-thermogenesis Hall model (Eq. 3, 5--9)."""

    binding_id = "HALL2011_REDUCED_LONG_TERM_ADULT_WEIGHT_V11"
    equation_registry_ids = (
        "EQ-G2A-01",
        "EQ-G2A-02",
        "EQ-G2A-03",
        "EQ-G2A-04",
        "EQ-G2A-05",
        "EQ-G2A-06",
        "EQ-G2A-07",
    )

    def __init__(
        self,
        role: ModelRole,
        baseline: AdultFemaleBaseline,
        activity_map: ActivityEnergyMap,
        constants: HallConstants = HallConstants(),
        integration_step_day: float = 0.25,
    ) -> None:
        if role not in {ModelRole.PLANNING, ModelRole.EVALUATION_PARAMETER}:
            raise RoleViolation(
                "nonlinear Hall model may bind only M_P or M_E_par"
            )
        baseline.validate()
        constants.validate()
        activity_map.validate()
        subdivisions = 7.0 / integration_step_day
        if not isfinite(integration_step_day) or integration_step_day <= 0.0:
            raise RoleViolation("integration step must be positive")
        if abs(subdivisions - round(subdivisions)) > 1e-12:
            raise RoleViolation(
                "integration step must divide seven days exactly"
            )
        self.role = role
        self.baseline = baseline
        self.activity_map = activity_map
        self.constants = constants
        self.integration_step_day = integration_step_day
        self._steps_per_week = int(round(subdivisions))

        fat0, lean0 = baseline.fat_mass_kg, baseline.lean_mass_kg
        rmr0 = baseline.resting_metabolic_rate_kcal_day
        self.baseline_energy_intake_kcal_day = baseline.background_pal * rmr0
        self._k_kcal_day = (
            rmr0
            - constants.gamma_f_kcal_kg_day * fat0
            - constants.gamma_l_kcal_kg_day * lean0
        )
        self._delta_kcal_kg_day = (
            ((1.0 - constants.beta_tef) * baseline.background_pal - 1.0)
            * rmr0
            / baseline.weight_kg
        )

    def initial_state(self, observed_at_week: int = 0) -> StateVector:
        return StateVector(
            (self.baseline.fat_mass_kg, self.baseline.lean_mass_kg, 0.0),
            HALL_STATE_UNITS,
            observed_at_week,
        )

    @staticmethod
    def weight_kg(state: StateVector) -> float:
        HallLongTermModel._validate_state(state)
        return state.values[0] + state.values[1]

    @staticmethod
    def _validate_state(state: StateVector) -> None:
        state.validate()
        if state.units != HALL_STATE_UNITS or len(state.values) != 3:
            raise RoleViolation(
                "Hall state must be (fat kg, lean kg, AT kcal/day)"
            )
        if state.values[0] <= 0.0 or state.values[1] <= 0.0:
            raise RoleViolation("fat and lean mass must remain positive")

    def _derivative(
        self,
        y: np.ndarray,
        decision: DecisionVector,
        direct_activity_change_kcal_day: float | None = None,
    ) -> np.ndarray:
        fat, lean, adaptive_thermogenesis = (float(value) for value in y)
        if fat <= 0.0 or lean <= 0.0:
            raise RoleViolation(
                "integration left the positive body-composition domain"
            )
        constants = self.constants
        weight = fat + lean
        c_scaled = (
            constants.forbes_c_kg
            * constants.rho_l_kcal_kg
            / constants.rho_f_kcal_kg
        )
        partition = c_scaled / (c_scaled + fat)
        q_value = (
            partition
            * constants.eta_l_kcal_kg
            / constants.rho_l_kcal_kg
            + (1.0 - partition)
            * constants.eta_f_kcal_kg
            / constants.rho_f_kcal_kg
        )
        intake = decision.energy_intake_kcal_day
        if intake <= 0.0:
            raise RoleViolation("energy intake must be positive")
        activity = (
            self.activity_map.kcal_day(weight, decision)
            if direct_activity_change_kcal_day is None
            else direct_activity_change_kcal_day
        )
        if not isfinite(activity):
            raise RoleViolation("activity energy must be finite")
        numerator = (
            self._k_kcal_day
            + constants.gamma_f_kcal_kg_day * fat
            + constants.gamma_l_kcal_kg_day * lean
            + self._delta_kcal_kg_day * weight
            + constants.beta_tef * intake
            + adaptive_thermogenesis
            + activity
            + intake * q_value
        )
        expenditure = numerator / (1.0 + q_value)
        imbalance = intake - expenditure
        fat_change = (
            (1.0 - partition)
            * imbalance
            / constants.rho_f_kcal_kg
        )
        lean_change = (
            partition * imbalance / constants.rho_l_kcal_kg
        )
        adaptive_change = (
            constants.beta_at
            * (intake - self.baseline_energy_intake_kcal_day)
            - adaptive_thermogenesis
        ) / constants.tau_at_day
        return np.asarray(
            (fat_change, lean_change, adaptive_change),
            dtype=float,
        )

    def advance_days(
        self,
        state: StateVector,
        decision: DecisionVector,
        days: float,
    ) -> StateVector:
        self._validate_state(state)
        decision.validate_finite_only()
        if not isfinite(days) or days < 0.0:
            raise RoleViolation("days must be finite and nonnegative")
        n_float = days / self.integration_step_day
        if abs(n_float - round(n_float)) > 1e-12:
            raise RoleViolation(
                "days must be an integer multiple of the integration step"
            )
        values = np.asarray(state.values, dtype=float)
        step = self.integration_step_day
        for _ in range(int(round(n_float))):
            k1 = self._derivative(values, decision)
            k2 = self._derivative(values + 0.5 * step * k1, decision)
            k3 = self._derivative(values + 0.5 * step * k2, decision)
            k4 = self._derivative(values + step * k3, decision)
            values = values + step * (
                k1 + 2.0 * k2 + 2.0 * k3 + k4
            ) / 6.0
        if not np.all(np.isfinite(values)):
            raise RoleViolation("non-finite Hall state")
        return StateVector(
            tuple(float(value) for value in values),
            HALL_STATE_UNITS,
            state.observed_at_week,
        )

    def advance_days_exposure(
        self,
        state: StateVector,
        exposure: DirectEnergyExposure,
        days: float,
    ) -> StateVector:
        """Advance with measured diet and direct activity-energy change."""

        self._validate_state(state)
        exposure.validate()
        if not isfinite(days) or days < 0.0:
            raise RoleViolation("days must be finite and nonnegative")
        n_float = days / self.integration_step_day
        if abs(n_float - round(n_float)) > 1e-12:
            raise RoleViolation(
                "days must be an integer multiple of the integration step"
            )
        decision = DecisionVector(exposure.energy_intake_kcal_day, 0.0, 0.0)
        values = np.asarray(state.values, dtype=float)
        step = self.integration_step_day
        for _ in range(int(round(n_float))):
            k1 = self._derivative(
                values,
                decision,
                exposure.activity_change_kcal_day,
            )
            k2 = self._derivative(
                values + 0.5 * step * k1,
                decision,
                exposure.activity_change_kcal_day,
            )
            k3 = self._derivative(
                values + 0.5 * step * k2,
                decision,
                exposure.activity_change_kcal_day,
            )
            k4 = self._derivative(
                values + step * k3,
                decision,
                exposure.activity_change_kcal_day,
            )
            values = values + step * (
                k1 + 2.0 * k2 + 2.0 * k3 + k4
            ) / 6.0
        if not np.all(np.isfinite(values)):
            raise RoleViolation("non-finite Hall state")
        return StateVector(
            tuple(float(value) for value in values),
            HALL_STATE_UNITS,
            state.observed_at_week,
        )

    def step_week(
        self,
        state: StateVector,
        decision: DecisionVector,
        outer: OuterClock,
    ) -> StateVector:
        self._validate_state(state)
        if state.observed_at_week != outer.t_week:
            raise RoleViolation("state timestamp and outer clock disagree")
        advanced = self.advance_days(state, decision, 7.0)
        return StateVector(
            advanced.values,
            advanced.units,
            outer.t_week + 1,
        )


class HallLinearizedFormModel:
    """Independent structural form from Hall appendix Eq. 10--15."""

    role = ModelRole.EVALUATION_FORM
    binding_id = "HALL2011_LINEARIZED_FORM_V11"
    equation_registry_ids = ("EQ-G2A-08-LINEARIZED",)
    units = ("kg",)

    def __init__(
        self,
        baseline: AdultFemaleBaseline,
        constants: HallConstants = HallConstants(),
    ) -> None:
        baseline.validate()
        constants.validate()
        self.baseline = baseline
        self.constants = constants
        fat0 = baseline.fat_mass_kg
        alpha = constants.forbes_c_kg / fat0
        beta = constants.beta_at + constants.beta_tef
        self.rho_kcal_kg = (
            constants.eta_f_kcal_kg
            + constants.rho_f_kcal_kg
            + alpha
            * (constants.eta_l_kcal_kg + constants.rho_l_kcal_kg)
        ) / ((1.0 - beta) * (1.0 + alpha))
        rmr0 = baseline.resting_metabolic_rate_kcal_day
        delta = (
            ((1.0 - constants.beta_tef) * baseline.background_pal - 1.0)
            * rmr0
            / baseline.weight_kg
        )
        epsilon = (
            (
                constants.gamma_f_kcal_kg_day
                + alpha * constants.gamma_l_kcal_kg_day
            )
            / (1.0 + alpha)
            + delta
        ) / (1.0 - beta)
        self.tau_day = self.rho_kcal_kg / epsilon
        self.baseline_energy_intake_kcal_day = baseline.background_pal * rmr0

    def initial_state(self, observed_at_week: int = 0) -> StateVector:
        return StateVector(
            (self.baseline.weight_kg,),
            self.units,
            observed_at_week,
        )

    def step_week(
        self,
        state: StateVector,
        decision: DecisionVector,
        outer: OuterClock,
    ) -> StateVector:
        state.validate()
        decision.validate_finite_only()
        if state.units != self.units or len(state.values) != 1:
            raise RoleViolation(
                "linearized form state must contain body weight in kg"
            )
        if state.observed_at_week != outer.t_week:
            raise RoleViolation("state timestamp and outer clock disagree")
        if (
            decision.cardio_minutes_week != 0.0
            or decision.strength_minutes_week != 0.0
        ):
            raise RoleViolation(
                "linearized form accepts intake perturbations only"
            )
        energy_delta = (
            decision.energy_intake_kcal_day
            - self.baseline_energy_intake_kcal_day
        )
        equilibrium_delta = (
            energy_delta * self.tau_day / self.rho_kcal_kg
        )
        current_delta = state.values[0] - self.baseline.weight_kg
        next_delta = equilibrium_delta + (
            current_delta - equilibrium_delta
        ) * exp(-7.0 / self.tau_day)
        return StateVector(
            (self.baseline.weight_kg + next_delta,),
            self.units,
            outer.t_week + 1,
        )


@dataclass(frozen=True)
class ObservationRecord:
    observed_at_week: int
    weight_kg: float


class PastOnlyObservationModel:
    """M_O releases only measurements observed by the decision week."""

    role = ModelRole.OBSERVATION
    binding_id = "PAST_ONLY_OBSERVATION_V11"
    equation_registry_ids: tuple[str, ...] = ()

    def __init__(
        self,
        records: Mapping[str, Sequence[ObservationRecord]],
    ) -> None:
        self._records = {
            key: tuple(
                sorted(value, key=lambda item: item.observed_at_week)
            )
            for key, value in records.items()
        }

    def release(
        self,
        participant_id: str,
        decision_week: int,
    ) -> tuple[ObservationRecord, ...]:
        released = []
        for record in self._records.get(participant_id, ()):
            if record.observed_at_week <= decision_week:
                assert_information_release(
                    record.observed_at_week,
                    decision_week,
                )
                released.append(record)
        return tuple(released)


class DeterministicAdherenceStressModel:
    """M_A applies declared deterministic fractions; it is not a fit."""

    role = ModelRole.ADHERENCE_STRESS
    binding_id = "DETERMINISTIC_ADHERENCE_STRESS_V11"
    equation_registry_ids = ("EQ-G2A-09-ADHERENCE",)

    def __init__(
        self,
        baseline_intake_kcal_day: float,
        intake_fraction: float,
        cardio_fraction: float,
        strength_fraction: float,
    ) -> None:
        values = (
            baseline_intake_kcal_day,
            intake_fraction,
            cardio_fraction,
            strength_fraction,
        )
        if (
            not all(isfinite(value) for value in values)
            or baseline_intake_kcal_day <= 0.0
        ):
            raise RoleViolation(
                "adherence inputs must be finite and baseline intake positive"
            )
        if not all(0.0 <= value <= 1.0 for value in values[1:]):
            raise RoleViolation("adherence fractions must be in [0, 1]")
        self.baseline_intake_kcal_day = baseline_intake_kcal_day
        self.intake_fraction = intake_fraction
        self.cardio_fraction = cardio_fraction
        self.strength_fraction = strength_fraction

    def apply(self, decision: DecisionVector) -> DecisionVector:
        decision.validate_finite_only()
        intake = self.baseline_intake_kcal_day + self.intake_fraction * (
            decision.energy_intake_kcal_day
            - self.baseline_intake_kcal_day
        )
        return DecisionVector(
            intake,
            self.cardio_fraction * decision.cardio_minutes_week,
            self.strength_fraction * decision.strength_minutes_week,
        )
