from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pytest

import dt_ramde_v11.engine as engine_module
from dt_ramde_v11.contracts import (
    AlgorithmConfig,
    COMPACT_CHECKPOINT_AUDIT_MATERIALIZATION,
    ConfigurationError,
    ExecutionScope,
    R2ExecutionRequest,
)
from dt_ramde_v11.engine import DTRAMDE
from dt_ramde_v11.state import (
    MGState,
    MemoryState,
    ParameterAtom,
    StateIntegrityError,
)
from evaluation.contracts import EvaluationResult, TerminalCode
from evaluation.evaluator import SharedEvaluator
from evaluation.firewall import (
    InformationField,
    InformationSnapshot,
    freeze_information,
)
from evaluation.ledger import EvaluationLedger


class _Selector:
    selector_id = "fixture.lexicographic"
    selector_version = "1"

    def identity(self) -> Mapping[str, Any]:
        return {
            "selector_id": self.selector_id,
            "selector_version": self.selector_version,
        }

    def select(self, archive: Sequence[EvaluationResult]) -> str | None:
        return min((candidate.candidate_id for candidate in archive), default=None)


class _Problem:
    adapter_id = "fixture.problem"
    adapter_version = "1"
    decision_dimension = 2
    atomic_steps_per_evaluation = 2
    lower_bounds = np.asarray([0.0, 0.0])
    upper_bounds = np.asarray([1.0, 1.0])
    constraint_scales = (1.0,)

    def __init__(
        self,
        *,
        always_infeasible: bool = False,
        reject_safety: bool = False,
        numerical_failure: bool = False,
    ) -> None:
        self.outer_state = 0
        self.execution_log: list[tuple[int, bool, tuple[float, ...]]] = []
        self.evaluation_outer_states: list[tuple[int, int]] = []
        self._information = None
        self._always_infeasible = always_infeasible
        self._reject_safety = reject_safety
        self._numerical_failure = numerical_failure

        def joint(
            vector: Sequence[float], _information: object
        ) -> tuple[tuple[float, ...], tuple[float, ...]]:
            if self._numerical_failure:
                raise FloatingPointError("synthetic evaluator failure")
            x = np.asarray(vector, dtype=float)
            constraint = 1.0 if self._always_infeasible else -1.0
            return (
                (float(np.sum(x**2)), float(np.sum((x - 0.5) ** 2))),
                (constraint,),
            )

        self._evaluator = SharedEvaluator(
            objective_names=("distance_zero", "distance_half"),
            constraint_names=("fixture_feasibility",),
            evaluate_joint=joint,
        )

    def identity(self) -> Mapping[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "role": "synthetic_correctness_fixture",
        }

    def freeze_information(
        self, event_id: int, feedback: Mapping[str, Any] | None
    ) -> object:
        fields = {
            "current_state": InformationField(
                available_at=event_id, value={"outer_state": self.outer_state}
            )
        }
        if feedback is not None:
            fields["released_feedback"] = InformationField(
                available_at=event_id, value=dict(feedback)
            )
        self._information = freeze_information(
            decision_time=event_id,
            fields=fields,
        )
        return self._information

    def evaluate(
        self,
        vector: Sequence[float],
        event_id: int,
        ledger: EvaluationLedger,
        candidate_id: str,
    ) -> EvaluationResult:
        self.evaluation_outer_states.append((event_id, self.outer_state))
        return self._evaluator.evaluate(
            vector=vector,
            event_id=event_id,
            candidate_id=candidate_id,
            information=self._information,
            ledger=ledger,
            atomic_steps=self.atomic_steps_per_evaluation,
        )

    def safety_filter(self, result: EvaluationResult, event_id: int) -> bool:
        return result.feasible and not self._reject_safety

    def shift_solution(self, vector: Sequence[float]) -> np.ndarray:
        return np.asarray(vector, dtype=float)

    def execute(
        self,
        action: Sequence[float],
        event_id: int,
        committed: bool,
        ledger: EvaluationLedger,
    ) -> Mapping[str, Any]:
        values = tuple(float(value) for value in action)
        self.execution_log.append((event_id, committed, values))
        ledger.record_execution()
        if committed:
            self.outer_state += 1
        return {
            "available": committed,
            "ell_exec": 0.2,
            "ell_ref": 0.4,
            "s_exec": 0.5,
            "hard_constraint_violation": False,
        }

    def first_action(self, vector: Sequence[float]) -> np.ndarray:
        values = np.asarray(vector, dtype=float)
        return values[:1].copy()

    def fallback_action(self, event_id: int) -> np.ndarray:
        return np.zeros(1, dtype=float)


class _UnreadableExecutionFeedback(Mapping[str, Any]):
    def _raise(self, operation: str) -> None:
        raise AssertionError(f"execution feedback was observed via {operation}")

    def __iter__(self):
        self._raise("__iter__")

    def __len__(self) -> int:
        self._raise("__len__")

    def __getitem__(self, key: str) -> Any:
        del key
        self._raise("__getitem__")

    def items(self):
        self._raise("items")

    def keys(self):
        self._raise("keys")

    def values(self):
        self._raise("values")

    def get(self, key: str, default: Any = None) -> Any:
        del key, default
        self._raise("get")

    def dict(self):
        self._raise("dict")

    def __bool__(self) -> bool:
        self._raise("__bool__")


class _TripwireFeedbackProblem(_Problem):
    def execute(
        self,
        action: Sequence[float],
        event_id: int,
        committed: bool,
        ledger: EvaluationLedger,
    ) -> Mapping[str, Any]:
        super().execute(action, event_id, committed, ledger)
        return _UnreadableExecutionFeedback()


class _UnchargedFailureProblem(_Problem):
    def evaluate(
        self,
        vector: Sequence[float],
        event_id: int,
        ledger: EvaluationLedger,
        candidate_id: str,
    ) -> EvaluationResult:
        raise FloatingPointError("failure before ledger charge")


class _FutureSnapshotProblem(_Problem):
    def freeze_information(
        self, event_id: int, feedback: Mapping[str, Any] | None
    ) -> object:
        return freeze_information(
            decision_time=event_id + 1,
            fields={
                "future_but_self_consistent": InformationField(
                    available_at=event_id + 1,
                    value=42,
                )
            },
        )


class _ForgedSnapshotProblem(_Problem):
    def freeze_information(
        self, event_id: int, feedback: Mapping[str, Any] | None
    ) -> object:
        return InformationSnapshot(
            decision_time=event_id,
            fields={
                "future_trajectory": InformationField(
                    available_at=event_id,
                    value=[1, 2, 3],
                )
            },
            information_hash="0" * 64,
        )


class _NoTransitionChargeProblem(_Problem):
    def execute(
        self,
        action: Sequence[float],
        event_id: int,
        committed: bool,
        ledger: EvaluationLedger,
    ) -> Mapping[str, Any]:
        return {
            "available": committed,
            "ell_exec": 0.2,
            "ell_ref": 0.4,
            "s_exec": 0.5,
            "hard_constraint_violation": False,
        }


class _LateNumericalProblem(_Problem):
    def evaluate(
        self,
        vector: Sequence[float],
        event_id: int,
        ledger: EvaluationLedger,
        candidate_id: str,
    ) -> EvaluationResult:
        self._numerical_failure = ledger.snapshot()["cfe"] == 8
        return super().evaluate(vector, event_id, ledger, candidate_id)


def _config(**changes: Any) -> AlgorithmConfig:
    values: dict[str, Any] = {
        "variant": "FULL",
        "population_size": 4,
        "cfe_per_event": 8,
        "algorithm_seed": 17,
        "max_events": 2,
        "timing_mode": "TS2_fixed_periodic_replanning",
        "method_label": "DT-RAMDE_TS2_FULL",
        "adapter_id": "fixture.problem",
        "adapter_version": "1",
        "selector_id": "fixture.lexicographic",
        "selector_version": "1",
        "atomic_steps_per_evaluation": 2,
        "event_time_limit_seconds": 3600.0,
        "configuration_evidence_id": "UNIT_TEST_FIXTURE",
        "execution_request": R2ExecutionRequest(
            scope=ExecutionScope.UNIT_TEST_FIXTURE
        ),
    }
    values.update(changes)
    return AlgorithmConfig(**values)


def test_engine_is_deterministic_budget_exact_and_outer_state_is_immutable() -> None:
    first_problem = _Problem()
    first = DTRAMDE(_config()).run_sequence(first_problem, selector=_Selector())

    assert len(first.events) == 2
    assert [event.ledger["cfe"] for event in first.events] == [8, 8]
    assert [event.ledger["atomic_model_steps"] for event in first.events] == [
        16,
        16,
    ]
    assert first_problem.outer_state == 2
    assert first_problem.execution_log[0][1] is True
    assert first.events[1].warm_start_seed_count > 0
    assert {state for event, state in first_problem.evaluation_outer_states if event == 0} == {0}
    assert {state for event, state in first_problem.evaluation_outer_states if event == 1} == {1}
    assert len(first.events[0].state_transitions) == 9
    assert {
        "pending_credit",
        "bank",
        "atom_audit",
        "tau",
        "reset_log",
    }.issubset(first.events[0].memory_snapshot)
    assert first.events[0].lineage_records
    trial_lineage = next(
        item
        for item in first.events[0].lineage_records
        if item["generation"] >= 0
    )
    assert {
        "objectives",
        "constraints",
        "feasible",
        "normalized_cv",
        "survival",
        "archive_admission",
        "target_predecessor",
        "pbest_id",
        "r1_id",
        "r2_id",
    }.issubset(trial_lineage)
    trial_audit = first.events[0].trial_audit[0]
    assert {
        "operator_audit",
        "raw_vector",
        "repaired_vector",
        "pre_repair_hash",
        "post_repair_hash",
        "selection",
        "archive_admission",
    }.issubset(trial_audit)
    assert {
        "operator",
        "j_rand",
    } == set(trial_audit["operator_audit"]["rng"])
    assert first.events[0].archive_audit
    assert first.events[0].archive_audit[-1]["rng"]["tokens"]["substream"] == "archive"
    initialization = first.events[0].initialization_audit
    assert initialization["rng"]["tokens"]["stream"] == "initialization"
    assert initialization["rng"]["tokens"]["substream"] == "initialization"
    assert len(initialization["roots"]) == 4
    assert {
        "candidate_id",
        "source",
        "vector",
        "vector_hash",
        "evaluation_status",
    }.issubset(initialization["roots"][0])
    assert first.effect_estimation_performed is False
    assert first.hidden_seed_or_instance_generated is False

    second_problem = _Problem()
    second = DTRAMDE(_config()).run_sequence(second_problem, selector=_Selector())
    assert first.to_dict() == second.to_dict()


def test_rank_and_crowding_is_assigned_once_per_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = engine_module.assign_rank_and_crowding

    def counted(candidates: Sequence[Any]) -> None:
        nonlocal calls
        calls += 1
        original(candidates)

    monkeypatch.setattr(engine_module, "assign_rank_and_crowding", counted)
    result = DTRAMDE(_config(max_events=1)).run_sequence(
        _Problem(),
        selector=_Selector(),
    )

    assert result.events[0].ledger["cfe"] == 8
    assert len(result.events[0].trial_audit) == 4
    assert calls == 1


def test_no_execution_feedback_never_observes_or_serializes_adapter_return() -> None:
    problem = _TripwireFeedbackProblem()
    result = DTRAMDE(
        _config(
            variant="NO_EXECUTION_FEEDBACK",
            method_label="NO_EXECUTION_FEEDBACK",
        )
    ).run_sequence(problem, selector=_Selector())

    assert len(result.events) == 2
    assert problem.outer_state == 2
    assert [
        event.ledger["execution_transition_count"] for event in result.events
    ] == [1, 1]
    assert [event.execution_feedback for event in result.events] == [None, None]
    assert [
        event["execution_feedback"] for event in result.to_dict()["events"]
    ] == [None, None]


def test_full_variant_still_observes_required_execution_feedback() -> None:
    with pytest.raises(StateIntegrityError) as captured:
        DTRAMDE(_config()).run_sequence(
            _TripwireFeedbackProblem(),
            selector=_Selector(),
        )
    cause = captured.value.__cause__
    assert isinstance(cause, AssertionError)
    assert "execution feedback was observed" in str(cause)


def test_shade_only_runs_with_result_blind_success_audit_and_no_cross_memory() -> None:
    result = DTRAMDE(
        _config(
            variant="SHADE_ONLY",
            method_label="SHADE_ONLY",
        )
    ).run_sequence(_Problem(), selector=_Selector())

    assert result.config["variant_components"] == {
        "M_g": True,
        "M_g_mode": "WGT_SHADE_CMO_SUCCESS_01",
        "M_k": False,
        "warm_start": False,
        "execution_q": False,
        "rejection_q": False,
        "lineage": "off",
        "soft_reset": False,
    }
    assert [event.warm_start_seed_count for event in result.events] == [0, 0]
    assert all(event.memory_snapshot["bank"] == [] for event in result.events)

    required = {
        "paired_target_id",
        "mg_mode",
        "mg_success",
        "mg_success_reason",
        "mg_success_delta",
        "mg_success_weight",
        "inferior_parent_archive_admission",
    }
    allowed_success_reasons = {
        "INFEASIBLE_TO_FEASIBLE",
        "INFEASIBLE_CV_REDUCTION",
        "FEASIBLE_PARETO_DOMINANCE",
    }
    for event in result.events:
        audited = [
            row
            for row in event.trial_audit
            if row.get("evaluation_status") == "completed"
            and row["selection"] in {"survived", "discarded"}
        ]
        assert audited
        assert all(required.issubset(row) for row in audited)
        for row in audited:
            assert row["mg_mode"] == "WGT_SHADE_CMO_SUCCESS_01"
            if row["mg_success"]:
                assert row["mg_success_reason"] in allowed_success_reasons
                assert row["mg_success_delta"] > 0.0
                assert row["mg_success_weight"] > 0.0
                assert row["inferior_parent_archive_admission"] is True
            else:
                assert row["mg_success_delta"] == 0.0
                assert row["mg_success_weight"] == 0.0

        generations: dict[str, list[Mapping[str, Any]]] = {}
        for row in audited:
            generation = row["node_id"].split(":")[1]
            generations.setdefault(generation, []).append(row)
        for rows in generations.values():
            successful = [row for row in rows if row["mg_success"]]
            if successful:
                assert sum(
                    row["mg_success_weight"] for row in successful
                ) == pytest.approx(1.0)


def test_engine_uses_independent_algorithm_substreams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str | None] = []
    original = engine_module.derive_rng

    def recording_derive_rng(*args: Any, **kwargs: Any) -> Any:
        observed.append(kwargs.get("substream"))
        return original(*args, **kwargs)

    monkeypatch.setattr(engine_module, "derive_rng", recording_derive_rng)
    DTRAMDE(_config(max_events=1)).run_sequence(
        _Problem(),
        selector=_Selector(),
    )

    assert {
        "initialization",
        "parameter",
        "operator",
        "j_rand",
        "archive",
    }.issubset(set(observed))


def test_parameter_source_decision_precedes_conditional_atom_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = MemoryState(
        bank=[
            ParameterAtom(
                f=0.6,
                cr=0.7,
                signed_credit=0.5,
                source_event=0,
                lineage_node_id="prior-node",
            )
        ],
        tau=0.0,
    )
    engine = DTRAMDE(_config(max_events=1), memory)
    calls = 0
    original = engine_module.sample_atom

    def recording_sample_atom(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(engine_module, "sample_atom", recording_sample_atom)
    _f, _cr, source, _audit = engine._parameter_source(
        problem=_Problem(),
        event_id=0,
        generation=0,
        target_index=0,
        mg=MGState.initialize(),
    )

    assert calls == 0
    assert source.startswith("M_g:")


def test_event_mg_is_fresh_and_mk_sampling_does_not_overwrite_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ParameterRng:
        def random(self) -> float:
            return 0.0

        def choice(self, _count: int, *, p: np.ndarray) -> int:
            assert p.tolist() == [1.0]
            return 0

        def standard_cauchy(self) -> float:
            return 0.0

        def normal(self, center: float, _scale: float) -> float:
            return center

    monkeypatch.setattr(
        engine_module,
        "derive_rng",
        lambda *args, **kwargs: (ParameterRng(), {"tokens": {}}),
    )
    memory = MemoryState(
        bank=[
            ParameterAtom(
                f=0.8,
                cr=0.2,
                signed_credit=0.5,
                source_event=0,
                lineage_node_id="prior",
            )
        ],
        tau=1.0,
    )
    mg = MGState.initialize()
    engine = DTRAMDE(_config(max_events=1), memory)
    _f, _cr, source, audit = engine._parameter_source(
        problem=_Problem(),
        event_id=1,
        generation=0,
        target_index=0,
        mg=mg,
    )

    assert source == "M_k:0:prior"
    assert (audit["mu_f"], audit["mu_cr"]) == (0.8, 0.2)
    assert mg.memory_f == [0.5] * 10
    assert mg.memory_cr == [0.5] * 10
    assert mg.pointer == 0


def test_no_feasible_and_safety_paths_are_typed_and_not_deleted() -> None:
    infeasible_problem = _Problem(always_infeasible=True)
    infeasible = DTRAMDE(_config(max_events=1)).run_sequence(
        infeasible_problem, selector=_Selector()
    )
    assert infeasible.events[0].terminal.code is (
        TerminalCode.REJECT_BUDGET_NO_FEASIBLE
    )
    assert infeasible.events[0].ledger["cfe"] == 8
    assert infeasible.events[0].ledger["execution_transition_count"] == 1
    assert infeasible_problem.execution_log[0][2] == (0.0,)

    unsafe = DTRAMDE(_config(max_events=1)).run_sequence(
        _Problem(reject_safety=True), selector=_Selector()
    )
    assert unsafe.events[0].terminal.code is TerminalCode.REJECT_SAFETY_FILTER
    assert unsafe.events[0].terminal.candidate_id is not None


def test_numerical_failure_is_charged_and_typed() -> None:
    result = DTRAMDE(_config(max_events=1)).run_sequence(
        _Problem(numerical_failure=True), selector=_Selector()
    )
    event = result.events[0]
    assert event.terminal.code is TerminalCode.REJECT_NUMERICAL
    assert event.ledger["cfe"] == 1
    assert event.ledger["evaluation_failures"] == 1


def test_late_numerical_rejection_keeps_the_predecessor_credit_chain() -> None:
    result = DTRAMDE(
        _config(max_events=1, cfe_per_event=9)
    ).run_sequence(_LateNumericalProblem(), selector=_Selector())
    event = result.events[0]
    assert event.terminal.code is TerminalCode.REJECT_NUMERICAL
    failed = next(
        item
        for item in event.lineage_records
        if item["node_id"] == event.terminal.candidate_id
    )
    predecessor_id = failed["target_predecessor"]
    predecessor = next(
        item
        for item in event.lineage_records
        if item["node_id"] == predecessor_id
    )
    assert predecessor["survival"] is True
    assert predecessor["f"] is not None
    pending = event.memory_snapshot["pending_credit"]
    assert pending["lineage_weights"]
    assert pending["lineage_weights"][0][0] == predecessor_id


def test_adapter_cannot_hide_an_uncharged_evaluation_failure() -> None:
    with pytest.raises(StateIntegrityError, match="exactly one joint CFE"):
        DTRAMDE(_config(max_events=1)).run_sequence(
            _UnchargedFailureProblem(),
            selector=_Selector(),
        )


def test_ts2_adapter_must_record_the_fallback_or_committed_transition() -> None:
    engine = DTRAMDE(_config(max_events=2))
    with pytest.raises(StateIntegrityError, match="execution transition"):
        engine.run_sequence(
            _NoTransitionChargeProblem(always_infeasible=True),
            selector=_Selector(),
        )
    assert engine.memory.event_index == -1
    assert engine.memory.pending_credit is None
    assert engine.memory.invalidated is True
    with pytest.raises(StateIntegrityError, match="invalidated"):
        engine.run_sequence(_Problem(), selector=_Selector())
    persisted_invalid = MemoryState.from_dict(engine.memory.to_dict())
    with pytest.raises(StateIntegrityError, match="invalidated"):
        DTRAMDE(_config(max_events=2), persisted_invalid)


def test_information_snapshot_must_match_the_current_event() -> None:
    with pytest.raises(StateIntegrityError, match="decision time"):
        DTRAMDE(_config(max_events=1)).run_sequence(
            _FutureSnapshotProblem(),
            selector=_Selector(),
        )


def test_information_snapshot_is_revalidated_at_the_engine_boundary() -> None:
    with pytest.raises(StateIntegrityError, match="snapshot integrity"):
        DTRAMDE(_config(max_events=1)).run_sequence(
            _ForgedSnapshotProblem(),
            selector=_Selector(),
        )


def test_charged_nonrecoverable_evaluator_error_is_typed() -> None:
    problem = _Problem()

    def fail_after_entry(
        vector: Sequence[float], information: object
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        raise RuntimeError("synthetic nonrecoverable evaluator error")

    problem._evaluator = SharedEvaluator(
        objective_names=("distance_zero", "distance_half"),
        constraint_names=("fixture_feasibility",),
        evaluate_joint=fail_after_entry,
    )
    result = DTRAMDE(_config(max_events=1)).run_sequence(
        problem,
        selector=_Selector(),
    )

    assert result.events[0].terminal.code is TerminalCode.REJECT_NUMERICAL
    assert result.events[0].ledger["cfe"] == 1
    assert result.events[0].ledger["evaluation_failures"] == 1


def test_common_event_resource_limit_produces_typed_timeout() -> None:
    elapsed = [0.0]

    class _TimeoutProblem(_Problem):
        def evaluate(
            self,
            vector: Sequence[float],
            event_id: int,
            ledger: EvaluationLedger,
            candidate_id: str,
        ) -> EvaluationResult:
            result = super().evaluate(
                vector,
                event_id,
                ledger,
                candidate_id,
            )
            elapsed[0] = 2.0
            return result

    result = DTRAMDE(
        _config(max_events=1, event_time_limit_seconds=1.0),
        clock=lambda: elapsed[0],
    ).run_sequence(_TimeoutProblem(), selector=_Selector())

    event = result.events[0]
    assert event.terminal.code is TerminalCode.REJECT_TIMEOUT
    assert event.terminal.candidate_id is not None
    assert event.ledger["cfe"] == 1
    assert event.ledger["evaluation_failures"] == 0


@pytest.mark.parametrize("limit", [0.0, -1.0, float("nan"), float("inf")])
def test_event_resource_limit_must_be_finite_and_positive(limit: float) -> None:
    with pytest.raises(ConfigurationError, match="event_time_limit_seconds"):
        DTRAMDE(_config(event_time_limit_seconds=limit))


def test_ts1_has_one_event_and_no_cross_event_state() -> None:
    result = DTRAMDE(
        _config(
            variant="NO_CROSS_EVENT_MEMORY",
            max_events=1,
            timing_mode="TS1_single_event",
            method_label="F22_MG_STATIC",
        )
    ).run_sequence(_Problem(), selector=_Selector())
    state = result.persistent_state
    assert state["bank"] == []
    assert state["solution_memory"] == []
    assert state["pending_credit"] is None


def test_prohibited_scope_is_rejected_before_adapter_use() -> None:
    with pytest.raises(ConfigurationError, match="R2 correctness scope"):
        DTRAMDE(
            _config(
                execution_request=R2ExecutionRequest(
                    scope=ExecutionScope.BENCHMARK_EFFECT
                )
            )
        )


def test_method_and_interface_identities_are_exactly_bound() -> None:
    with pytest.raises(ConfigurationError, match="method_label"):
        DTRAMDE(_config(method_label="renamed-full"))

    wrong_adapter = _Problem()
    wrong_adapter.adapter_version = "2"
    with pytest.raises(StateIntegrityError, match="adapter version"):
        DTRAMDE(_config(max_events=1)).run_sequence(
            wrong_adapter,
            selector=_Selector(),
        )

    wrong_selector = _Selector()
    wrong_selector.selector_version = "2"
    with pytest.raises(StateIntegrityError, match="selector identity"):
        DTRAMDE(_config(max_events=1)).run_sequence(
            _Problem(),
            selector=wrong_selector,
        )


def test_config_snapshot_contains_frozen_component_semantics() -> None:
    snapshot = DTRAMDE(_config(max_events=1)).config.to_dict()
    assert "audit_materialization" not in snapshot
    assert snapshot["variant_components"] == {
        "M_g": True,
        "M_g_mode": "F22_weighted_survivor",
        "M_k": True,
        "warm_start": True,
        "execution_q": True,
        "rejection_q": True,
        "lineage": "chain",
        "soft_reset": True,
    }


def test_compact_audit_mode_rejects_non_corrective_fixture_binding() -> None:
    with pytest.raises(
        ConfigurationError,
        match=r"restricted to corrective E1\+E2",
    ):
        DTRAMDE(
            _config(
                audit_materialization=(
                    COMPACT_CHECKPOINT_AUDIT_MATERIALIZATION
                )
            )
        )


@pytest.mark.parametrize(
    ("variant", "method_label"),
    [
        ("FULL", "DT-RAMDE_TS2_FULL"),
        ("SHADE_ONLY", "SHADE_ONLY"),
    ],
)
def test_interrupted_resume_matches_uninterrupted_and_rejects_config_drift(
    variant: str,
    method_label: str,
) -> None:
    config = _config(
        max_events=2,
        variant=variant,
        method_label=method_label,
    )
    uninterrupted = DTRAMDE(config).run_sequence(
        _Problem(),
        selector=_Selector(),
    )

    resumed_problem = _Problem()
    first_engine = DTRAMDE(config)
    first_event = first_engine.run_event(
        resumed_problem,
        selector=_Selector(),
        event_id=0,
        prior_feedback=None,
    )
    restored = MemoryState.from_dict(first_engine.memory.to_dict())
    resumed = DTRAMDE(config, restored).run_sequence(
        resumed_problem,
        selector=_Selector(),
        prior_feedback=first_event.execution_feedback,
    )

    assert len(resumed.events) == 1
    assert resumed.events[0].to_dict() == uninterrupted.events[1].to_dict()
    assert resumed.persistent_state == uninterrupted.persistent_state

    with pytest.raises(StateIntegrityError, match="binding"):
        DTRAMDE(
            _config(
                max_events=2,
                algorithm_seed=18,
                variant=variant,
                method_label=method_label,
            ),
            MemoryState.from_dict(first_engine.memory.to_dict()),
        )
    drifted = MemoryState.from_dict(first_engine.memory.to_dict())
    drifted.tau = 0.6
    with pytest.raises(StateIntegrityError, match="checkpoint checksum"):
        DTRAMDE(config, drifted)
