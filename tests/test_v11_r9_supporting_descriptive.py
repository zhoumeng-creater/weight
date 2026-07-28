from __future__ import annotations

from pathlib import Path

from jsonschema import Draft202012Validator
import json
import pytest

from analysis.r9_supporting_descriptive import (
    ANALYSIS_STATUS,
    OutcomeRow,
    _cost_set_membership,
    _csv_data_row_count,
    _geometric_mean,
    _practical_class,
)


ROOT = Path(__file__).resolve().parents[1]


def _outcome(
    *,
    failure_count: int = 0,
    charged_cfe: int = 100,
    charged_atomic: int = 600,
) -> OutcomeRow:
    return OutcomeRow(
        task_id="task",
        schedule_index=0,
        workload_id="E1_ROLLING",
        unit_id="RR-SMOOTH/0",
        method_id="DT-RAMDE_TS2_FULL",
        replicate_index=0,
        task_status="COMPLETE",
        outcome_class="",
        terminal_counts=(
            ("ACCEPTED", 20 - failure_count),
            ("REJECT_SAFETY_FILTER", failure_count),
        ),
        event_count=20,
        failure_count=failure_count,
        evaluation_failure_count=0,
        scheduled_cfe=100,
        charged_cfe=charged_cfe,
        unconsumed_cfe=100 - charged_cfe,
        scheduled_atomic_model_steps=600,
        charged_atomic_model_steps=charged_atomic,
        charged_work_exact=True,
        wall_seconds=10.0,
        cpu_seconds=9.0,
        peak_rss_bytes=100,
        output_bytes=1000,
        automatic_retries=0,
    )


def test_geometric_mean_known_answer() -> None:
    assert _geometric_mean([1.0, 4.0]) == pytest.approx(2.0)


def test_csv_data_row_count_known_answer() -> None:
    assert _csv_data_row_count(b"a,b\n1,2\n3,4\n") == 2


def test_practical_class_direction_for_lower_is_better_failure() -> None:
    assert _practical_class(0.03, 0.02) == (
        "POSITIVE_FAVORS_PROPOSED"
    )
    assert _practical_class(-0.03, 0.02) == (
        "NEGATIVE_FAVORS_COMPARATOR"
    )
    assert _practical_class(0.01, 0.02) == "SMALL_OR_NULL"


def test_cost_available_sets_are_explicit_and_noninterchangeable() -> None:
    proposed = _outcome(failure_count=1)
    comparator = _outcome(failure_count=0)

    membership = _cost_set_membership(proposed, comparator)

    assert membership["ALL_COMPLETED_TASK_PAIRS"] is True
    assert membership["EQUAL_CHARGED_WORK_TASK_PAIRS"] is True
    assert membership["BOTH_ALL_EVENTS_ACCEPTED_TASK_PAIRS"] is False


def test_equal_charged_work_requires_both_cfe_and_atomic_steps() -> None:
    proposed = _outcome(charged_cfe=100, charged_atomic=600)
    comparator = _outcome(charged_cfe=100, charged_atomic=599)

    assert _cost_set_membership(
        proposed,
        comparator,
    )["EQUAL_CHARGED_WORK_TASK_PAIRS"] is False


def test_supporting_contract_schema_and_no_r10_identity() -> None:
    contract = json.loads(
        (
            ROOT
            / "config"
            / "r9"
            / "r9_supporting_descriptive_implementation_v1_0_1.json"
        ).read_text(encoding="utf-8")
    )
    schema = json.loads(
        (
            ROOT
            / "config"
            / "r9"
            / "r9_supporting_descriptive_implementation_v1_0_1.schema.json"
        ).read_text(encoding="utf-8")
    )

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(contract)
    assert contract["analysis_status"] == ANALYSIS_STATUS
    assert contract["authorization"]["r10_authorized"] is False
    assert contract["procedure"]["new_confirmatory_hypotheses"] == 0
    assert contract["procedure"]["new_p_values"] == 0
    assert contract["procedure"]["new_holm_families"] == 0
