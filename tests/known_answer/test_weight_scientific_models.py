"""F17 known-answer and failure tests for the R3 F09 port.

Provenance:
    FORMAL_V1/tests/test_g2a_scientific_models.py -> F17 conditional port
    Source SHA-256 81e22f9ec8b20b459e0f190c14223939a125795dd8c1b1080e9f938becaff775
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from weight_application.model_roles import (
    DecisionVector,
    ModelRole,
    OuterClock,
    RoleViolation,
)
from weight_application.scientific_models import (
    ActivityEnergyMap,
    AdultFemaleBaseline,
    DeterministicAdherenceStressModel,
    HallConstants,
    HallLinearizedFormModel,
    HallLongTermModel,
    ObservationRecord,
    PastOnlyObservationModel,
)


def baseline() -> AdultFemaleBaseline:
    return AdultFemaleBaseline(55.0, 165.0, 90.0, 1.4, True)


def activity() -> ActivityEnergyMap:
    return ActivityEnergyMap(3.0, 2.0, "SYNTHETIC_TEST_ONLY")


def model(
    step: float = 0.25,
    role: ModelRole = ModelRole.PLANNING,
) -> HallLongTermModel:
    return HallLongTermModel(
        role,
        baseline(),
        activity(),
        integration_step_day=step,
    )


def test_f17_mifflin_and_hall_female_initialization() -> None:
    subject = baseline()
    assert subject.resting_metabolic_rate_kcal_day == pytest.approx(
        1495.25,
        abs=1e-10,
    )
    expected_percent = (
        0.14 * 55.0 + 39.96 * np.log(90.0 / 1.65**2) - 102.01
    )
    assert subject.fat_mass_kg == pytest.approx(
        90.0 * expected_percent / 100.0,
        abs=1e-12,
    )
    assert subject.fat_mass_kg + subject.lean_mass_kg == pytest.approx(
        90.0,
        abs=1e-12,
    )


def test_f17_published_unit_conversions() -> None:
    constants = HallConstants()
    assert constants.rho_f_kcal_kg * 4.184 == pytest.approx(
        39_500.0,
        abs=1e-9,
    )
    assert constants.rho_l_kcal_kg * 4.184 == pytest.approx(
        7_600.0,
        abs=1e-9,
    )
    assert constants.gamma_l_kcal_kg_day * 4.184 == pytest.approx(
        92.0,
        abs=1e-12,
    )


def test_f17_net_met_day_week_conversion() -> None:
    decision = DecisionVector(1800.0, 140.0, 70.0)
    expected = (
        (3.0 * 140.0 + 2.0 * 70.0) * 3.5 * 90.0 / 200.0 / 7.0
    )
    assert activity().kcal_day(90.0, decision) == pytest.approx(
        expected,
        abs=1e-12,
    )


def test_f17_zero_perturbation_preserves_state() -> None:
    hall = model()
    initial = hall.initial_state()
    decision = DecisionVector(
        hall.baseline_energy_intake_kcal_day,
        0.0,
        0.0,
    )
    advanced = hall.step_week(initial, decision, OuterClock(0, 0))
    np.testing.assert_allclose(
        advanced.values,
        initial.values,
        rtol=0.0,
        atol=2e-12,
    )


def test_f17_deficit_and_surplus_have_opposite_signs() -> None:
    hall = model()
    initial = hall.initial_state()
    low = hall.step_week(
        initial,
        DecisionVector(
            hall.baseline_energy_intake_kcal_day - 500.0,
            0.0,
            0.0,
        ),
        OuterClock(0, 0),
    )
    high = hall.step_week(
        initial,
        DecisionVector(
            hall.baseline_energy_intake_kcal_day + 500.0,
            0.0,
            0.0,
        ),
        OuterClock(0, 0),
    )
    assert hall.weight_kg(low) < 90.0 < hall.weight_kg(high)
    assert low.values[0] < initial.values[0]
    assert low.values[1] < initial.values[1]
    assert high.values[0] > initial.values[0]
    assert high.values[1] > initial.values[1]


def test_f17_body_composition_conservation() -> None:
    hall = model()
    initial = hall.initial_state()
    advanced = hall.step_week(
        initial,
        DecisionVector(
            hall.baseline_energy_intake_kcal_day - 400.0,
            90.0,
            30.0,
        ),
        OuterClock(0, 0),
    )
    delta_weight = hall.weight_kg(advanced) - hall.weight_kg(initial)
    assert delta_weight == pytest.approx(
        (advanced.values[0] - initial.values[0])
        + (advanced.values[1] - initial.values[1]),
        abs=1e-13,
    )


def test_f17_day_week_time_equivalence() -> None:
    hall = model()
    decision = DecisionVector(
        hall.baseline_energy_intake_kcal_day - 300.0,
        60.0,
        30.0,
    )
    weekly = hall.step_week(
        hall.initial_state(),
        decision,
        OuterClock(0, 0),
    )
    daily = hall.initial_state()
    for _ in range(7):
        daily = hall.advance_days(daily, decision, 1.0)
    np.testing.assert_allclose(
        weekly.values,
        daily.values,
        rtol=0.0,
        atol=1e-12,
    )


def test_f17_intake_and_activity_monotonicity() -> None:
    hall = model()
    intakes = [
        hall.baseline_energy_intake_kcal_day + offset
        for offset in (-600.0, -300.0, 0.0, 300.0, 600.0)
    ]
    intake_weights = [
        hall.weight_kg(
            hall.step_week(
                hall.initial_state(),
                DecisionVector(intake, 0.0, 0.0),
                OuterClock(0, 0),
            )
        )
        for intake in intakes
    ]
    assert all(
        first < second
        for first, second in zip(
            intake_weights[:-1],
            intake_weights[1:],
            strict=True,
        )
    )
    activity_weights = [
        hall.weight_kg(
            hall.step_week(
                hall.initial_state(),
                DecisionVector(
                    hall.baseline_energy_intake_kcal_day,
                    minutes,
                    0.0,
                ),
                OuterClock(0, 0),
            )
        )
        for minutes in (0.0, 60.0, 120.0)
    ]
    assert activity_weights[0] > activity_weights[1] > activity_weights[2]


def test_f17_rk4_step_halving_converges() -> None:
    finals = []
    for step in (3.5, 1.75, 0.875):
        hall = model(step)
        state = hall.initial_state()
        decision = DecisionVector(
            hall.baseline_energy_intake_kcal_day - 450.0,
            100.0,
            40.0,
        )
        for week in range(104):
            state = hall.step_week(
                state,
                decision,
                OuterClock(week, 0),
            )
        finals.append(hall.weight_kg(state))
    assert abs(finals[2] - finals[1]) < abs(finals[1] - finals[0])
    assert abs(finals[2] - finals[1]) < 2e-9


def test_f17_independent_dop853_reference() -> None:
    hall = model(0.25)
    decision = DecisionVector(
        hall.baseline_energy_intake_kcal_day - 350.0,
        80.0,
        20.0,
    )
    state = hall.initial_state()
    for week in range(12):
        state = hall.step_week(
            state,
            decision,
            OuterClock(week, 0),
        )
    reference = solve_ivp(
        lambda _time, values: hall._derivative(values, decision),
        (0.0, 84.0),
        np.asarray(hall.initial_state().values),
        method="DOP853",
        rtol=1e-12,
        atol=1e-13,
    )
    assert reference.success
    np.testing.assert_allclose(
        state.values,
        reference.y[:, -1],
        rtol=0.0,
        atol=2e-8,
    )


def test_f17_parallel_order_invariance() -> None:
    offsets = (-500.0, -250.0, 0.0, 250.0)

    def run(offset: float) -> float:
        hall = model()
        decision = DecisionVector(
            hall.baseline_energy_intake_kcal_day + offset,
            50.0,
            20.0,
        )
        return hall.weight_kg(
            hall.step_week(
                hall.initial_state(),
                decision,
                OuterClock(0, 0),
            )
        )

    serial = [run(offset) for offset in offsets]
    with ThreadPoolExecutor(max_workers=4) as pool:
        parallel = list(pool.map(run, reversed(offsets)))[::-1]
    np.testing.assert_array_equal(serial, parallel)


def test_f17_small_perturbation_matches_independent_linearized_form() -> None:
    nonlinear = model(0.125)
    linear = HallLinearizedFormModel(baseline())
    nonlinear_decision = DecisionVector(
        nonlinear.baseline_energy_intake_kcal_day - 10.0,
        0.0,
        0.0,
    )
    linear_decision = DecisionVector(
        linear.baseline_energy_intake_kcal_day - 10.0,
        0.0,
        0.0,
    )
    nonlinear_state = nonlinear.initial_state()
    linear_state = linear.initial_state()
    for week in range(8):
        nonlinear_state = nonlinear.step_week(
            nonlinear_state,
            nonlinear_decision,
            OuterClock(week, 0),
        )
        linear_state = linear.step_week(
            linear_state,
            linear_decision,
            OuterClock(week, 0),
        )
    assert (
        abs(nonlinear.weight_kg(nonlinear_state) - linear_state.values[0])
        < 0.01
    )


def test_f17_roles_context_future_information_and_adherence_fail_closed() -> None:
    evaluated = model(role=ModelRole.EVALUATION_PARAMETER)
    assert evaluated.role is ModelRole.EVALUATION_PARAMETER
    with pytest.raises(RoleViolation):
        HallLongTermModel(
            ModelRole.EVALUATION_FORM,
            baseline(),
            activity(),
        )
    with pytest.raises(RoleViolation):
        AdultFemaleBaseline(17.0, 165.0, 60.0, 1.4, True).validate()
    with pytest.raises(RoleViolation):
        AdultFemaleBaseline(55.0, 165.0, 90.0, 1.4, False).validate()

    hall = model()
    state = hall.initial_state()
    bad_state = type(state)(
        state.values,
        ("lb", "kg", "kcal/day"),
        0,
    )
    with pytest.raises(RoleViolation):
        hall.step_week(
            bad_state,
            DecisionVector(1800.0, 0.0, 0.0),
            OuterClock(0, 0),
        )

    observations = PastOnlyObservationModel(
        {
            "p": [
                ObservationRecord(0, 90.0),
                ObservationRecord(6, 85.0),
            ]
        }
    )
    assert observations.release("p", 0) == (ObservationRecord(0, 90.0),)
    assert len(observations.release("p", 6)) == 2

    adherence = DeterministicAdherenceStressModel(
        2200.0,
        0.5,
        0.8,
        0.25,
    )
    assert adherence.apply(
        DecisionVector(1800.0, 100.0, 40.0)
    ) == DecisionVector(2000.0, 80.0, 10.0)
    with pytest.raises(RoleViolation):
        DeterministicAdherenceStressModel(2200.0, 1.1, 1.0, 1.0)
