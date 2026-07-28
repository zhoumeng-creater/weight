from __future__ import annotations

from dataclasses import replace
import json
from math import inf, nan
from pathlib import Path
from typing import Any, Mapping, Sequence

import jsonschema
import pytest

from dt_ramde_v11.contracts import (
    ConfigurationError,
    ContractBindings,
    ExecutionScope,
    R2ExecutionRequest,
)
from dt_ramde_v11.interfaces import (
    EventProblemAdapter,
    OptimizationResult,
    OptimizerOrComparator,
)
from evaluation.contracts import (
    EvaluationContractError,
    EvaluationResult,
    TerminalCode,
    TerminalOutcome,
)
from evaluation.firewall import (
    InformationField,
    InformationSnapshot,
    freeze_information as freeze_information_snapshot,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
R2_REQUEST_SCHEMA = (
    PROJECT_ROOT / "config" / "schema" / "v11_r2_execution_request.schema.json"
)


def test_contract_bindings_are_rebound_to_the_current_v11_overlay() -> None:
    bindings = ContractBindings()
    bindings.validate()
    assert bindings.protocol_id == "WGT-JOURNAL-2026-01"
    assert bindings.protocol_version == "v1.1.8-r2-shade-success-frozen"
    assert bindings.r2_start_record_id == "WGT-V11-R2-START-20260724-01"
    assert (
        bindings.f22_f23_shade_amendment_id
        == "AMEND-V11-F22-F23-SHADE-20260724-01"
    )

    with pytest.raises(ConfigurationError, match="contract identity"):
        replace(bindings, protocol_version="v1.0-frozen-rq4-rolling-only").validate()


@pytest.mark.parametrize(
    "scope",
    [
        ExecutionScope.BENCHMARK_EFFECT,
        ExecutionScope.WEIGHT_EFFECT,
        ExecutionScope.HIDDEN,
        ExecutionScope.CONFIRMATORY,
    ],
)
def test_r2_execution_request_rejects_non_correctness_scopes(
    scope: ExecutionScope,
) -> None:
    with pytest.raises(ConfigurationError, match="R2 correctness scope"):
        R2ExecutionRequest(scope=scope).validate()


@pytest.mark.parametrize(
    "field",
    [
        "participant_data_requested",
        "effect_estimation_requested",
        "hidden_generation_requested",
        "results_writing_requested",
        "remote_git_mutation_requested",
        "release_or_distribution_requested",
    ],
)
def test_r2_execution_request_rejects_prohibited_permissions(field: str) -> None:
    request = R2ExecutionRequest(scope=ExecutionScope.UNIT_TEST_FIXTURE)
    with pytest.raises(ConfigurationError, match="prohibited permission"):
        replace(request, **{field: True}).validate()


def test_r2_execution_request_allows_only_correctness_fixtures() -> None:
    for scope in (
        ExecutionScope.UNIT_TEST_FIXTURE,
        ExecutionScope.PUBLIC_CORRECTNESS_FIXTURE,
    ):
        R2ExecutionRequest(scope=scope).validate()


def test_r2_execution_request_schema_accepts_only_the_golden_correctness_shape() -> None:
    schema = json.loads(R2_REQUEST_SCHEMA.read_text(encoding="utf-8"))
    jsonschema.validators.validator_for(schema).check_schema(schema)
    golden = {
        "protocol_id": "WGT-JOURNAL-2026-01",
        "protocol_version": "v1.1.8-r2-shade-success-frozen",
        "r2_start_record_id": "WGT-V11-R2-START-20260724-01",
        "f22_f23_shade_amendment_id": (
            "AMEND-V11-F22-F23-SHADE-20260724-01"
        ),
        "f22_shade_success_overlay_id": (
            "WGT-DT-RAMDE-F22-SHADE-SUCCESS-OVERLAY-01"
        ),
        "f23_shade_success_overlay_id": (
            "WGT-F23-SHADE-SUCCESS-BINDING-OVERLAY-01"
        ),
        "scope": "public_correctness_fixture",
        "permissions": {
            "participant_data_requested": False,
            "effect_estimation_requested": False,
            "hidden_generation_requested": False,
            "results_writing_requested": False,
            "remote_git_mutation_requested": False,
            "release_or_distribution_requested": False,
        },
    }
    jsonschema.validate(instance=golden, schema=schema)

    prohibited = {
        **golden,
        "scope": "benchmark_effect",
        "permissions": {
            **golden["permissions"],
            "effect_estimation_requested": True,
        },
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=prohibited, schema=schema)


def test_evaluation_result_uses_vector_objectives_and_c_le_zero_constraints() -> None:
    feasible = EvaluationResult(
        candidate_id="c-1",
        objectives=(1.0, 2.0),
        objective_names=("quality", "cost"),
        constraints=(-1.0, 0.0),
        constraint_names=("capacity", "boundary"),
    )
    assert feasible.feasible is True
    assert feasible.total_violation == 0.0

    infeasible = replace(feasible, candidate_id="c-2", constraints=(0.5, -2.0))
    assert infeasible.feasible is False
    assert infeasible.total_violation == 0.5


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("objectives", (nan,)),
        ("objectives", (inf,)),
        ("constraints", (nan,)),
        ("constraints", (inf,)),
    ],
)
def test_evaluation_result_rejects_nonfinite_values(
    field: str, value: tuple[float, ...]
) -> None:
    payload = {
        "candidate_id": "c-invalid",
        "objectives": (1.0,),
        "objective_names": ("quality",),
        "constraints": (-1.0,),
        "constraint_names": ("capacity",),
    }
    payload[field] = value
    with pytest.raises(EvaluationContractError, match="finite"):
        EvaluationResult(**payload)


def test_terminal_outcomes_are_typed_and_accepted_requires_a_candidate() -> None:
    accepted = TerminalOutcome(code=TerminalCode.ACCEPTED, candidate_id="c-1")
    assert accepted.candidate_id == "c-1"

    rejected = TerminalOutcome(
        code=TerminalCode.REJECT_NO_FEASIBLE,
        reason="synthetic fixture has no feasible point",
    )
    assert rejected.candidate_id is None

    with pytest.raises(EvaluationContractError, match="candidate_id"):
        TerminalOutcome(code=TerminalCode.ACCEPTED)


def test_optimization_result_keeps_archive_and_terminal_selection_consistent() -> None:
    candidate = EvaluationResult(
        candidate_id="c-1",
        objectives=(1.0,),
        objective_names=("quality",),
        constraints=(-1.0,),
        constraint_names=("capacity",),
    )
    result = OptimizationResult(
        terminal=TerminalOutcome(code=TerminalCode.ACCEPTED, candidate_id="c-1"),
        archive=(candidate,),
        selected_vector=(0.25,),
    )
    assert result.archive == (candidate,)
    assert result.selected_vector == (0.25,)

    with pytest.raises(EvaluationContractError, match="archive"):
        OptimizationResult(
            terminal=TerminalOutcome(
                code=TerminalCode.ACCEPTED, candidate_id="not-in-archive"
            ),
            archive=(candidate,),
            selected_vector=(0.25,),
        )
    with pytest.raises(EvaluationContractError, match="selected vector"):
        OptimizationResult(
            terminal=TerminalOutcome(
                code=TerminalCode.ACCEPTED,
                candidate_id="c-1",
            ),
            archive=(candidate,),
        )


class _ProblemFixture:
    adapter_id = "fixture.problem"
    adapter_version = "1"
    decision_dimension = 1
    atomic_steps_per_evaluation = 1
    lower_bounds = (0.0,)
    upper_bounds = (1.0,)
    constraint_scales = (1.0,)

    def identity(self) -> Mapping[str, Any]:
        return {"adapter_id": self.adapter_id}

    def freeze_information(
        self, event_id: int, feedback: Mapping[str, Any] | None
    ) -> InformationSnapshot:
        return freeze_information_snapshot(
            decision_time=event_id,
            fields={
                "event_id": InformationField(
                    available_at=event_id,
                    value=event_id,
                ),
                "feedback": InformationField(
                    available_at=event_id,
                    value=feedback,
                ),
            },
        )

    def evaluate(
        self,
        vector: Sequence[float],
        event_id: int,
        ledger: Any,
        candidate_id: str,
    ) -> EvaluationResult:
        return EvaluationResult(
            candidate_id=candidate_id,
            objectives=(float(vector[0]),),
            objective_names=("quality",),
            constraints=(-1.0,),
            constraint_names=("capacity",),
        )

    def safety_filter(self, result: EvaluationResult, event_id: int) -> bool:
        return result.feasible

    def shift_solution(self, vector: Sequence[float]) -> tuple[float, ...]:
        return tuple(vector)

    def execute(
        self,
        action: Sequence[float],
        event_id: int,
        committed: bool,
        ledger: Any,
    ) -> Mapping[str, Any]:
        return {"committed": committed}

    def first_action(self, vector: Sequence[float]) -> tuple[float, ...]:
        return tuple(vector[:1])

    def fallback_action(self, event_id: int) -> tuple[float, ...]:
        return (0.0,)


class _OptimizerFixture:
    method_id = "fixture.optimizer"
    method_version = "1"

    def identity(self) -> Mapping[str, Any]:
        return {"method_id": self.method_id}

    def optimize(
        self,
        problem: EventProblemAdapter,
        *,
        event_id: int,
        budget: int,
        seed: int,
        ledger: Any,
    ) -> OptimizationResult:
        return OptimizationResult(
            terminal=TerminalOutcome(code=TerminalCode.REJECT_NO_FEASIBLE),
            archive=(),
        )


def test_problem_and_algorithm_interfaces_are_distinct_runtime_contracts() -> None:
    problem = _ProblemFixture()
    optimizer = _OptimizerFixture()

    assert isinstance(problem, EventProblemAdapter)
    assert not isinstance(problem, OptimizerOrComparator)
    assert isinstance(optimizer, OptimizerOrComparator)
    assert not isinstance(optimizer, EventProblemAdapter)
