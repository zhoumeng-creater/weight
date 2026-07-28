from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.validate_r5_freeze import (
    CONTRACT_PATH,
    R5ValidationError,
    _expected_workloads,
    _validate_statistics_and_permissions,
    validate_r5,
)


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_r5_result_blind_freeze_validator_passes_without_effect_execution() -> None:
    assert validate_r5() == {
        "validator": "WGT-V11-R5-RESULT-BLIND-FREEZE-VALIDATOR-01",
        "status": "PASS",
        "paired_master_seed_count": 10,
        "rolling_public_instance_count": 32,
        "e3_public_subject_count": 32,
        "confirmatory_hypothesis_count": 30,
        "unique_method_sequences": 7046,
        "total_CFE": 1075647488,
        "total_atomic_model_steps": 3318884928,
        "effect_estimation_performed": False,
        "participant_data_accessed": False,
        "hidden_instance_accessed_or_generated": False,
        "results_analysis_performed": False,
        "r6_or_formal_execution_authorized": False,
    }


def test_r5_workload_is_recomputed_from_sample_method_and_budget_fields() -> None:
    contract = _contract()
    expected = _expected_workloads(contract)

    assert expected == contract["workload_budget"]["unique_workloads"]
    assert sum(item["method_sequences"] for item in expected) == 7046
    assert sum(item["CFE"] for item in expected) == 1075647488
    assert (
        sum(item["atomic_model_steps"] for item in expected)
        == 3318884928
    )


def test_r5_seed_prefixes_and_public_rolling_strata_are_exact() -> None:
    contract = _contract()
    seed_contract = contract["seed_contract"]
    rows = seed_contract["rolling_public_instances"]

    assert seed_contract["use_prefix_counts"] == {
        "E1_STATIC": 10,
        "E1_DYNAMIC": 5,
        "E1_ROLLING": 5,
        "E2_DYNAMIC": 5,
        "E2_ROLLING": 5,
        "E3": 3,
    }
    assert {
        row["template"] for row in rows
    } == {
        "RR-SMOOTH",
        "RR-SHOCK",
        "RR-REJECTION",
        "RR-INTERMITTENT",
    }
    for template in {row["template"] for row in rows}:
        assert sorted(
            row["index"] for row in rows if row["template"] == template
        ) == list(range(8))
    assert seed_contract["rolling_public_generator"]["hidden"] is False


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (
            ("permissions", "formal_effect_execution_allowed"),
            True,
            "permission differs",
        ),
        (
            (
                "endpoint_contract",
                "nhv",
                "observed_method_union_reference_allowed",
            ),
            True,
            "observed-method reference",
        ),
        (
            ("resource_budget", "scratch", "onedrive_path_allowed"),
            True,
            "OneDrive execution scratch",
        ),
    ],
)
def test_r5_semantic_guard_rejects_permission_reference_or_scratch_drift(
    path: tuple[str, ...],
    value: bool,
    match: str,
) -> None:
    contract = deepcopy(_contract())
    target = contract
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(R5ValidationError, match=match):
        _validate_statistics_and_permissions(contract)


def test_r5_validator_source_does_not_import_effect_adapters() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "validate_r5_freeze.py"
    ).read_text(encoding="utf-8")

    assert "benchmark_adapters" not in source
    assert "comparators" not in source
    assert "weight_application" not in source
    assert "run_v11_experiment" not in source
