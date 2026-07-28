from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytest

from benchmark_adapters.r4_evaluators import CDFEvaluator
from benchmark_adapters.r4_public import (
    make_r4_cdf_adapter,
    make_r4_lircmop_adapter,
)
from benchmark_adapters.r4_wgt_rr import WGTRRPublicAdapter
from comparators import (
    ConventionalRollingPlannerBaseline,
    FixedEnergyDeficitBaseline,
    JMetalComparator,
    MatchedParetoDE,
)
from dt_ramde_v11.core import Candidate
from dt_ramde_v11.interfaces import OptimizerOrComparator
from evaluation.contracts import EvaluationResult
from evaluation.evaluator import SharedEvaluator
from evaluation.firewall import (
    InformationField,
    InformationSnapshot,
    freeze_information,
)
from evaluation.ledger import EvaluationLedger
from weight_application.adapter import SyntheticWeightAdapter
from weight_application.state import (
    SyntheticWeightModel,
    SyntheticWeightState,
)
from tools.validate_r4_bindings import validate_r4


class _QuadraticAdapter:
    adapter_id = "R4-QUADRATIC-BRIDGE-FIXTURE"
    adapter_version = "1"
    decision_dimension = 3
    atomic_steps_per_evaluation = 2
    lower_bounds = (-1.0, -1.0, -1.0)
    upper_bounds = (1.0, 1.0, 1.0)
    objective_names = ("sphere", "shifted_sphere")
    constraint_names = ("always_feasible",)
    constraint_scales = (1.0,)

    def __init__(self) -> None:
        self._information: InformationSnapshot | None = None
        self._evaluator = SharedEvaluator(
            objective_names=self.objective_names,
            constraint_names=self.constraint_names,
            evaluate_joint=self._joint,
        )

    def identity(self) -> Mapping[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "role": "R4_bridge_correctness_only",
        }

    def freeze_information(
        self, event_id: int, feedback: Mapping[str, Any] | None
    ) -> InformationSnapshot:
        del feedback
        self._information = freeze_information(
            decision_time=event_id,
            fields={
                "current_event": InformationField(
                    available_at=event_id,
                    value=event_id,
                )
            },
        )
        return self._information

    @staticmethod
    def _joint(
        vector: Sequence[float], information: InformationSnapshot
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        values = np.asarray(vector, dtype=float)
        shift = information.decision_time / 10.0
        return (
            (
                float(np.sum(values**2)),
                float(np.sum((values - shift) ** 2)),
            ),
            (-1.0,),
        )

    def evaluate(
        self,
        vector: Sequence[float],
        event_id: int,
        ledger: EvaluationLedger,
        candidate_id: str,
    ) -> EvaluationResult:
        if self._information is None:
            raise RuntimeError("event is not frozen")
        return self._evaluator.evaluate(
            vector=vector,
            event_id=event_id,
            candidate_id=candidate_id,
            information=self._information,
            ledger=ledger,
            atomic_steps=self.atomic_steps_per_evaluation,
            origin="r4_quadratic_bridge_fixture",
        )

    @staticmethod
    def safety_filter(result: EvaluationResult, event_id: int) -> bool:
        del event_id
        return result.feasible

    @staticmethod
    def shift_solution(vector: Sequence[float]) -> np.ndarray:
        return np.asarray(vector, dtype=float).copy()

    @staticmethod
    def first_action(vector: Sequence[float]) -> np.ndarray:
        return np.asarray(vector, dtype=float).copy()

    @staticmethod
    def fallback_action(event_id: int) -> np.ndarray:
        del event_id
        return np.zeros(3)

    @staticmethod
    def execute(
        action: Sequence[float],
        event_id: int,
        committed: bool,
        ledger: EvaluationLedger,
    ) -> Mapping[str, Any]:
        del action, committed
        ledger.record_execution()
        return {
            "available": False,
            "ell_exec": None,
            "ell_ref": None,
            "s_exec": None,
            "hard_constraint_violation": None,
            "released_at": event_id + 1,
        }


def _weight_adapter() -> SyntheticWeightAdapter:
    return SyntheticWeightAdapter(
        initial_state=SyntheticWeightState(
            event_id=0,
            fat_mass_kg=24.0,
            lean_mass_kg=56.0,
            cumulative_energy_imbalance_kcal=0.0,
        ),
        target_mass_kg=77.0,
        model=SyntheticWeightModel(
            event_days=7.0,
            energy_density_kcal_per_kg=7700.0,
            fat_mass_change_fraction=0.75,
        ),
    )


def test_r4_public_adapter_identities_keep_effect_execution_closed() -> None:
    static = make_r4_lircmop_adapter(1)
    dynamic = make_r4_cdf_adapter(1, profile="CDF-MILD")
    rolling = WGTRRPublicAdapter.from_known_answer()

    for adapter in (static, dynamic, rolling):
        identity = adapter.identity()
        assert identity["registered_benchmark_evaluator"] is True
        assert identity["registered_effect_instance"] is False
        assert identity["formal_effect_execution_allowed"] is False


@pytest.mark.parametrize("problem_index", range(1, 16))
@pytest.mark.parametrize("profile", ["CDF-HARSH", "CDF-MILD"])
def test_cdf_full_registry_returns_finite_canonical_vectors(
    problem_index: int, profile: str
) -> None:
    evaluator = CDFEvaluator(
        problem_index=problem_index,
        profile=profile,
        environment_seed=17,
    )
    lower = np.asarray(evaluator.lower_bounds)
    upper = np.asarray(evaluator.upper_bounds)
    vector = lower + 0.25 * (upper - lower)
    objectives, constraints = evaluator(vector, event_id=7)

    assert len(objectives) == 2
    assert len(constraints) == len(evaluator.constraint_names)
    assert np.all(np.isfinite(objectives))
    assert np.all(np.isfinite(constraints))


def test_static_lircmop_bridge_flips_upstream_constraint_sign_once() -> None:
    adapter = make_r4_lircmop_adapter(1)
    adapter.freeze_information(0, None)
    ledger = EvaluationLedger(max_cfe=1)
    result = adapter.evaluate((0.5,) * 30, 0, ledger, "lir")

    from jmetal.problem.multiobjective.lircmop import LIRCMOP1

    upstream = LIRCMOP1()
    solution = upstream.create_solution()
    solution.variables = [0.5] * 30
    upstream.evaluate(solution)
    assert result.constraints == pytest.approx(
        tuple(-value for value in solution.constraints)
    )
    assert ledger.snapshot()["cfe"] == 1


def test_public_rolling_bridge_charges_six_atomic_steps_and_one_execution() -> None:
    adapter = WGTRRPublicAdapter.from_known_answer()
    snapshot = adapter.freeze_information(0, None)
    ledger = EvaluationLedger(max_cfe=1)
    result = adapter.evaluate((0.0,) * 12, 0, ledger, "rolling")
    feedback = adapter.execute((0.0, 0.0), 0, False, ledger)

    assert len(result.objectives) == 3
    assert len(result.constraints) == 30
    assert "disturbance_sequence" not in repr(snapshot.fields)
    assert feedback["released_at"] == 1
    assert ledger.snapshot()["atomic_model_steps"] == 6
    assert ledger.snapshot()["execution_transition_count"] == 1


@pytest.mark.parametrize("mode", ["fixed", "jde", "shade"])
def test_matched_de_categories_use_exact_shared_budget(mode: str) -> None:
    problem = _QuadraticAdapter()
    problem.freeze_information(0, None)
    ledger = EvaluationLedger(max_cfe=8)
    comparator = MatchedParetoDE(
        mode=mode,
        population_size=4,
        archive_capacity=8,
    )

    result = comparator.optimize(
        problem,
        event_id=0,
        budget=8,
        seed=23,
        ledger=ledger,
    )

    assert isinstance(comparator, OptimizerOrComparator)
    assert result.archive
    assert ledger.snapshot()["cfe"] == 8
    assert ledger.snapshot()["atomic_model_steps"] == 16


@pytest.mark.parametrize(
    "mode",
    ["gde3", "nsgaii_static", "nsgaii_dynamic_restart"],
)
def test_jmetal_categories_use_project_evaluator_and_exact_budget(
    mode: str,
) -> None:
    problem = _QuadraticAdapter()
    problem.freeze_information(0, None)
    ledger = EvaluationLedger(max_cfe=8)
    comparator = JMetalComparator(
        mode=mode,
        population_size=4,
        archive_capacity=8,
    )

    result = comparator.optimize(
        problem,
        event_id=0,
        budget=8,
        seed=29,
        ledger=ledger,
    )

    assert result.archive
    assert ledger.snapshot()["cfe"] == 8
    assert all(
        charge.origin == "r4_quadratic_bridge_fixture"
        for charge in ledger.evaluations
    )
    assert comparator.identity()["native_evaluator"] is False


def test_dynamic_nsgaii_restart_reuses_only_shifted_current_vectors() -> None:
    problem = _QuadraticAdapter()
    comparator = JMetalComparator(
        mode="nsgaii_dynamic_restart",
        population_size=4,
        archive_capacity=8,
    )
    for event_id in (0, 1):
        problem.freeze_information(event_id, None)
        ledger = EvaluationLedger(max_cfe=8)
        result = comparator.optimize(
            problem,
            event_id=event_id,
            budget=8,
            seed=31,
            ledger=ledger,
        )
        assert result.archive
        assert ledger.snapshot()["cfe"] == 8


def test_weight_domain_baselines_are_executable_but_not_effect_runs() -> None:
    fixed_problem = _weight_adapter()
    fixed_problem.freeze_information(0, None)
    fixed_ledger = EvaluationLedger(max_cfe=1)
    fixed = FixedEnergyDeficitBaseline()
    fixed_result = fixed.optimize(
        fixed_problem,
        event_id=0,
        budget=1,
        seed=0,
        ledger=fixed_ledger,
    )

    rolling_problem = _weight_adapter()
    rolling_problem.freeze_information(0, None)
    rolling_ledger = EvaluationLedger(max_cfe=8)
    rolling = ConventionalRollingPlannerBaseline(
        population_size=4,
        archive_capacity=8,
    )
    rolling_result = rolling.optimize(
        rolling_problem,
        event_id=0,
        budget=8,
        seed=37,
        ledger=rolling_ledger,
    )

    assert fixed_result.archive
    assert rolling_result.archive
    assert fixed_ledger.snapshot()["cfe"] == 1
    assert rolling_ledger.snapshot()["cfe"] == 8
    assert fixed.identity()["effect_execution_allowed"] is False
    assert rolling.identity()["effect_execution_allowed"] is False


@pytest.mark.parametrize(
    ("problem_index", "event_id", "expected_objectives", "expected_constraints"),
    [
        (
            1,
            0,
            (0.5073159294558478, 0.840128366705839),
            (0.0669872981077807, 0.0005382808662772187),
        ),
        (
            6,
            7,
            (16.363401266761816, 14.374554648812023),
            (0.4051838992327912, 0.5400839926159475),
        ),
        (
            9,
            7,
            (6.326017727512383, 6.8992604249388085),
            (0.38420659964791537, 0.5319841599961043),
        ),
        (
            13,
            7,
            (6.4049377857689915, 6.32783472958726),
            (-10.576928457076937,),
        ),
        (
            15,
            7,
            (3.998802267701416, 4.737499999999999),
            (-19.348509190143485,),
        ),
    ],
)
def test_cdf_frozen_known_answers(
    problem_index: int,
    event_id: int,
    expected_objectives: tuple[float, ...],
    expected_constraints: tuple[float, ...],
) -> None:
    evaluator = CDFEvaluator(
        problem_index=problem_index,
        profile="CDF-HARSH",
        environment_seed=17,
    )
    lower = np.asarray(evaluator.lower_bounds)
    upper = np.asarray(evaluator.upper_bounds)
    vector = lower + 0.25 * (upper - lower)
    objectives, constraints = evaluator(vector, event_id)

    assert objectives == pytest.approx(expected_objectives, rel=1e-12)
    assert constraints == pytest.approx(expected_constraints, rel=1e-12)
    assert evaluator.environment_schedule_commitment == (
        "30e1fa98204908835c73688141b3d3a6467b90d41d33d7726d7aa05645569b4a"
    )


def test_wgt_rr_uses_frozen_augmented_tchebycheff_selector() -> None:
    constraints = (-1.0,) * 30
    candidates = (
        Candidate(
            np.zeros(12),
            EvaluationResult(
                candidate_id="extreme",
                objectives=(0.1, 10.0, 10.0),
                objective_names=WGTRRPublicAdapter.objective_names,
                constraints=constraints,
                constraint_names=WGTRRPublicAdapter.constraint_names,
            ),
            "extreme",
        ),
        Candidate(
            np.ones(12),
            EvaluationResult(
                candidate_id="balanced",
                objectives=(1.0, 1.0, 1.0),
                objective_names=WGTRRPublicAdapter.objective_names,
                constraints=constraints,
                constraint_names=WGTRRPublicAdapter.constraint_names,
            ),
            "balanced",
        ),
    )

    selected = WGTRRPublicAdapter.select_candidate(candidates)

    assert selected.candidate_id == "balanced"


class _OneFailureAdapter(_QuadraticAdapter):
    def __init__(self) -> None:
        self.calls = 0
        super().__init__()

    def _joint(
        self,
        vector: Sequence[float],
        information: InformationSnapshot,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        self.calls += 1
        if self.calls == 1:
            raise FloatingPointError("intentional bridge fixture failure")
        return _QuadraticAdapter._joint(vector, information)


def test_jmetal_bridge_charges_and_excludes_numerical_failure() -> None:
    problem = _OneFailureAdapter()
    problem.freeze_information(0, None)
    ledger = EvaluationLedger(max_cfe=8)
    comparator = JMetalComparator(
        mode="nsgaii_static",
        population_size=4,
        archive_capacity=8,
    )

    result = comparator.optimize(
        problem,
        event_id=0,
        budget=8,
        seed=41,
        ledger=ledger,
    )

    assert result.archive
    assert ledger.snapshot()["cfe"] == 8
    assert ledger.snapshot()["evaluation_failures"] == 1


def test_r4_machine_manifests_and_all_executable_bindings_close() -> None:
    summary = validate_r4()

    assert summary == {
        "validator": "WGT-V11-R4-EXECUTABLE-BINDING-VALIDATOR-01",
        "status": "PASS",
        "method_categories_bound": 8,
        "static_problem_bindings_exercised": 14,
        "dynamic_profile_problem_bindings_exercised": 30,
        "rolling_known_answer_bindings_exercised": 1,
        "effect_estimation_performed": False,
        "participant_data_accessed": False,
        "hidden_instance_accessed_or_generated": False,
        "results_analysis_performed": False,
        "distribution_authorized": True,
    }
