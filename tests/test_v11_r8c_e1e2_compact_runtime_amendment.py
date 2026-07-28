from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from typing import Any

from jsonschema import Draft202012Validator, ValidationError
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "config" / "r8c_e1e2"
AMENDMENT_PATH = CONFIG_ROOT / "r8c_e1e2_compact_runtime_amendment.json"
AMENDMENT_SCHEMA_PATH = (
    CONFIG_ROOT / "r8c_e1e2_compact_runtime_amendment.schema.json"
)
PENDING_CONTRACT_PATH = (
    CONFIG_ROOT / "r8c_e1e2_formal_execution_contract.json"
)
FORMAL_SCHEMA_PATH = (
    CONFIG_ROOT / "r8c_e1e2_formal_execution_contract.schema.json"
)
QUALIFIED_SCHEMA_PATH = (
    CONFIG_ROOT / "r8c_e1e2_target_qualified_contract.schema.json"
)
AMENDMENT_ID = "WGT-V11-R8C-E1E2-COMPACT-RUNTIME-AMENDMENT-01"
AMENDMENT_RELATIVE_PATH = (
    "config/r8c_e1e2/r8c_e1e2_compact_runtime_amendment.json"
)
AMENDMENT_SCHEMA_RELATIVE_PATH = (
    "config/r8c_e1e2/"
    "r8c_e1e2_compact_runtime_amendment.schema.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _validator(path: Path) -> Draft202012Validator:
    schema = _read_json(path)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(
        schema,
        format_checker=Draft202012Validator.FORMAT_CHECKER,
    )


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _assert_no_effect_payload_keys(value: Any) -> None:
    prohibited = {
        "auc",
        "constraints",
        "nhv",
        "objectives",
        "terminal_candidate_id",
    }
    if isinstance(value, dict):
        assert prohibited.isdisjoint(
            str(key).casefold() for key in value
        )
        for item in value.values():
            _assert_no_effect_payload_keys(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_effect_payload_keys(item)


def test_compact_runtime_amendment_is_strict_result_blind_machine_evidence() -> None:
    amendment = _read_json(AMENDMENT_PATH)
    _validator(AMENDMENT_SCHEMA_PATH).validate(amendment)

    assert amendment["amendment_id"] == AMENDMENT_ID
    assert amendment["created_date"] == "2026-07-26"
    assert amendment["scope"]["e3_in_scope"] is False
    assert amendment["scope"]["research_endpoints_changed"] is False
    identities = amendment["runtime_identities"]
    assert identities["full"]["audit_materialization"] == "full"
    assert (
        identities["compact_checkpoint"]["audit_materialization"]
        == "compact_checkpoint"
    )
    assert (
        identities["full"]["legacy_config_serialization_field_present"]
        is False
    )

    equivalence = amendment["same_seed_equivalence"]
    assert [
        case["case_class"] for case in equivalence["cases"]
    ] == ["STATIC", "CDF", "ROLLING"]
    assert equivalence["all_three_case_classes_passed"] is True
    assert equivalence["byte_exact_claims"] == [
        "CHECKPOINT_FRONTS_CFE_FILE",
        "EVENT_SUMMARIES_JSONL",
    ]
    test_source = (
        PROJECT_ROOT / equivalence["test_file"]
    ).read_text(encoding="utf-8")
    assert f"def {equivalence['test_name']}(" in test_source

    verification = amendment["verification"]["focused_regression"]
    assert verification["status"] == "PASS"
    assert verification["test_count"] == 323
    assert all(
        (PROJECT_ROOT / path).is_file()
        for path in verification["test_paths"]
    )
    assert (
        amendment["result_blindness"]["effect_values_recorded"]
        is False
    )
    _assert_no_effect_payload_keys(amendment)


def test_compact_runtime_ab_samples_and_projection_recompute_exactly() -> None:
    amendment = _read_json(AMENDMENT_PATH)
    benchmark = amendment["local_ab_benchmark"]
    samples = benchmark["samples"]

    cdf = samples["CDF"]
    rolling = samples["ROLLING"]
    assert median(cdf["full_old_scales_wall_seconds"]) == pytest.approx(
        cdf["baseline_median_wall_seconds"],
        abs=1e-9,
    )
    assert median(cdf["compact_new_scales_wall_seconds"]) == pytest.approx(
        cdf["combined_median_wall_seconds"],
        abs=1e-9,
    )
    assert median(rolling["full_old_scales_wall_seconds"]) == pytest.approx(
        rolling["baseline_median_wall_seconds"],
        abs=1e-6,
    )
    assert median(
        rolling["compact_new_scales_wall_seconds"]
    ) == pytest.approx(
        rolling["combined_median_wall_seconds"],
        abs=1e-9,
    )
    for sample in (cdf, rolling):
        speedup = 100.0 * (
            sample["baseline_median_wall_seconds"]
            - sample["combined_median_wall_seconds"]
        ) / sample["baseline_median_wall_seconds"]
        assert round(speedup, 2) == sample["combined_speedup_percent"]

    projection = amendment["full_batch_projection"]
    weights = projection["formal_cfe_weights"]
    assert sum(
        (
            weights["dynamic_dt_ramde"],
            weights["rolling_dt_ramde"],
            weights["static_dt_ramde_assigned_zero_gain"],
            weights["comparators_assigned_zero_gain"],
        )
    ) == weights["total_e1_e2"] == 851_000_000
    weighted = 100.0 * (
        weights["dynamic_dt_ramde"]
        * cdf["combined_speedup_percent"]
        / 100.0
        + weights["rolling_dt_ramde"]
        * rolling["combined_speedup_percent"]
        / 100.0
    ) / weights["total_e1_e2"]
    assert round(weighted, 2) == (
        projection["median_cfe_weighted_speedup_percent"]
    )

    envelope = projection["conservative_observed_envelope"]
    conservative = 100.0 * (
        weights["dynamic_dt_ramde"]
        * envelope[
            "cdf_baseline_min_vs_combined_max_speedup_percent"
        ]
        / 100.0
        + weights["rolling_dt_ramde"]
        * envelope[
            "rolling_baseline_min_vs_combined_max_speedup_percent"
        ]
        / 100.0
    ) / weights["total_e1_e2"]
    assert round(conservative, 2) == (
        envelope["full_batch_cfe_weighted_speedup_percent"]
    )
    assert projection["decision_gate_passed"] is True
    assert (
        envelope["full_batch_cfe_weighted_speedup_percent"]
        > projection["decision_gate_percent"]
    )
    assert projection["is_target_host_wall_time_eta"] is False
    assert projection["may_replace_target_host_qualification"] is False


def test_pending_and_qualified_schemas_bind_exact_compact_amendment_hashes() -> None:
    pending = _read_json(PENDING_CONTRACT_PATH)
    _validator(FORMAL_SCHEMA_PATH).validate(pending)
    binding = pending["upstream"]["compact_runtime_amendment"]
    assert binding == {
        "amendment_id": AMENDMENT_ID,
        "path": AMENDMENT_RELATIVE_PATH,
        "sha256": _file_sha256(AMENDMENT_PATH),
        "schema_path": AMENDMENT_SCHEMA_RELATIVE_PATH,
        "schema_sha256": _file_sha256(AMENDMENT_SCHEMA_PATH),
    }

    expected_ref = {
        "$ref": "#/$defs/compact_runtime_amendment_upstream"
    }
    for schema_path in (
        FORMAL_SCHEMA_PATH,
        QUALIFIED_SCHEMA_PATH,
    ):
        schema = _read_json(schema_path)
        Draft202012Validator.check_schema(schema)
        upstream = schema["properties"]["upstream"]
        assert "compact_runtime_amendment" in upstream["required"]
        assert (
            upstream["properties"]["compact_runtime_amendment"]
            == expected_ref
        )
        bound = schema["$defs"][
            "compact_runtime_amendment_upstream"
        ]
        assert (
            bound["properties"]["amendment_id"]["const"]
            == AMENDMENT_ID
        )
        assert (
            bound["properties"]["path"]["const"]
            == AMENDMENT_RELATIVE_PATH
        )
        assert (
            bound["properties"]["sha256"]["const"]
            == _file_sha256(AMENDMENT_PATH)
        )
        assert (
            bound["properties"]["schema_path"]["const"]
            == AMENDMENT_SCHEMA_RELATIVE_PATH
        )
        assert (
            bound["properties"]["schema_sha256"]["const"]
            == _file_sha256(AMENDMENT_SCHEMA_PATH)
        )


def test_compact_runtime_schema_rejects_eta_authority_drift_and_extras() -> None:
    amendment = _read_json(AMENDMENT_PATH)
    validator = _validator(AMENDMENT_SCHEMA_PATH)

    eta_drift = deepcopy(amendment)
    eta_drift["full_batch_projection"][
        "is_target_host_wall_time_eta"
    ] = True
    with pytest.raises(ValidationError):
        validator.validate(eta_drift)

    speedup_drift = deepcopy(amendment)
    speedup_drift["local_ab_benchmark"]["samples"]["CDF"][
        "combined_speedup_percent"
    ] = 99.0
    with pytest.raises(ValidationError):
        validator.validate(speedup_drift)

    extra = deepcopy(amendment)
    extra["effect_result"] = 1.0
    with pytest.raises(ValidationError):
        validator.validate(extra)
