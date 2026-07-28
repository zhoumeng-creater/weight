from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from formal_execution.checkpoint_data import (
    EVENT_SUMMARY_MAX_RECORD_BYTES,
    WORKER_CONTROL_REPORT_MAX_BYTES,
    estimate_e1e2_checkpoint_storage,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "r8c_e1e2"
PERSISTENCE_PATH = (
    CONFIG / "r8c_e1e2_checkpoint_persistence_contract.json"
)
PERSISTENCE_SCHEMA_PATH = (
    CONFIG / "r8c_e1e2_checkpoint_persistence_contract.schema.json"
)
PENDING_PATH = CONFIG / "r8c_e1e2_formal_execution_contract.json"
PENDING_SCHEMA_PATH = (
    CONFIG / "r8c_e1e2_formal_execution_contract.schema.json"
)
QUALIFIED_SCHEMA_PATH = (
    CONFIG / "r8c_e1e2_target_qualified_contract.schema.json"
)
PERSISTENCE_ID = (
    "WGT-V11-R8C-E1E2-CHECKPOINT-PERSISTENCE-CONTRACT-01"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validator(path: Path) -> Draft202012Validator:
    schema = _read_json(path)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(
        schema,
        format_checker=Draft202012Validator.FORMAT_CHECKER,
    )


def test_checkpoint_persistence_contract_is_exact_and_implementation_bound() -> None:
    contract = _read_json(PERSISTENCE_PATH)
    _validator(PERSISTENCE_SCHEMA_PATH).validate(contract)

    estimate = estimate_e1e2_checkpoint_storage()
    frozen = contract["storage_estimate"]
    assert frozen["task_count"] == estimate.task_count == 5030
    assert frozen["event_count"] == estimate.event_count == 162640
    assert (
        frozen["checkpoint_record_count"]
        == estimate.checkpoint_record_count
        == 3_415_440
    )
    assert (
        frozen["objective_payload_bytes"]
        == estimate.objective_payload_bytes
        == 6_219_360_000
    )
    assert (
        frozen["front_max_constraint_payload_bytes"]
        == estimate.max_constraint_payload_bytes
        == 27_323_520
    )
    assert (
        frozen["fixed_record_overhead_bytes"]
        == estimate.fixed_record_overhead_bytes
        == 464_499_840
    )
    assert (
        frozen["file_header_upper_bound_bytes"]
        == estimate.file_header_upper_bound_bytes
        == 20_602_880
    )
    assert (
        frozen["conservative_total_upper_bound_bytes"]
        == estimate.conservative_total_upper_bound_bytes
        == 6_731_786_240
    )
    assert frozen["event_summary_upper_bound_bytes"] == (
        162_640 * EVENT_SUMMARY_MAX_RECORD_BYTES
    )
    assert frozen["worker_control_report_upper_bound_bytes"] == (
        5_030 * WORKER_CONTROL_REPORT_MAX_BYTES
    )
    assert frozen[
        "checkpoint_event_and_worker_control_upper_bound_bytes"
    ] == (
        estimate.conservative_total_upper_bound_bytes
        + frozen["event_summary_upper_bound_bytes"]
        + frozen["worker_control_report_upper_bound_bytes"]
    )
    assert frozen[
        "checkpoint_event_and_worker_control_strictly_less_than_12_gib"
    ] is True
    assert (
        frozen["checkpoint_event_and_worker_control_upper_bound_bytes"]
        < 12 * 1024**3
    )
    assert frozen["conservative_total_upper_bound_bytes"] < 7 * 1024**3


def test_checkpoint_semantics_preserve_r5_research_identity() -> None:
    contract = _read_json(PERSISTENCE_PATH)
    scope = contract["amendment_scope"]
    semantics = contract["checkpoint_semantics"]
    binary = contract["binary_format"]
    event_summary = contract["event_summary_control"]
    analysis = contract["analysis_boundary"]

    assert scope["e3_in_scope"] is False
    assert set(scope["r5_fields_replaced_for_e1_e2_only"]) == {
        "resource_budget.output.format",
        "resource_budget.output.max_total_gib",
        "resource_budget.output.raw_evaluations_required",
        "resource_budget.output.silent_truncation_allowed",
        "resource_budget.scratch.minimum_free_gib_at_r7_start",
        "resource_budget.scratch.stop_dispatch_below_free_gib",
    }
    assert all(scope["unchanged_research_identity"].values())
    assert scope["population_size"] == 100
    assert scope["archive_capacity"] == 100
    assert semantics["checkpoints_per_event"] == 21
    assert semantics["checkpoint_fractions"] == [
        index / 20 for index in range(21)
    ]
    assert semantics["previous_truncated_checkpoint_used_as_future_base"] is False
    assert semantics["checkpoint_zero"] == (
        "EMPTY_FRONT_BEFORE_ANY_CHARGED_CFE"
    )
    assert binary["byte_order"] == "LITTLE_ENDIAN"
    assert binary["floating_point"] == "IEEE_754_FLOAT64"
    assert binary["fixed_objective_slots_per_checkpoint"] == 100
    assert binary["valid_count_required"] is True
    assert binary["front_max_constraint"]["values_per_checkpoint"] == 1
    assert binary["evaluation_chain_sha256"]["prefix_bound"] is True
    assert binary["raw_evaluation_rows_persisted"] is False
    assert binary["terminal_candidate_identity_persisted"] is False
    assert binary["execution_observation_persisted_in_control_plane"] is True
    assert (
        binary[
            "algorithm_feedback_and_statistical_observation_channels_separated"
        ]
        is True
    )
    assert event_summary == {
        "filename": "event_summaries.jsonl",
        "encoding": "UTF-8_CANONICAL_JSONL_LF",
        "append_scope": "ONE_DURABLE_RECORD_PER_COMPLETED_EVENT",
        "maximum_rows": 162640,
        "maximum_canonical_record_bytes_including_lf": 8192,
        "maximum_opaque_trailing_fragment_bytes": 8191,
        "oversize_writer_action": "FAIL_BEFORE_WRITE",
        "reader_rejects_oversize_record_before_json_decode": True,
        "flush_after_each_event": True,
        "fsync_after_each_event": True,
        "terminal_code_enum_frozen": True,
        "execution_transition_count_required": True,
        "candidate_ids_persisted": False,
        "decision_vectors_persisted": False,
        "objective_fronts_persisted": False,
        "effect_endpoints_persisted": False,
        "partial_task_completed_event_prefix_preserved": True,
        "included_in_formal_12_gib_envelope": True,
        "strict_final_lf_required_for_complete_or_exact_task": True,
        "opaque_tail_exception_outcome_classes": [
            "TECHNICAL_SEQUENCE_TIMEOUT",
            "TECHNICAL_GLOBAL_TIMEOUT",
            "TECHNICAL_RESOURCE_TERMINATION",
        ],
        "opaque_tail_exception_requires_charged_work_exact_false": True,
        "whole_file_commitment_verified_before_tail_split": True,
        "trailing_fragment_interpreted_as_event": False,
        "trailing_fragment_metadata_reported": [
            "present",
            "bytes",
            "sha256",
        ],
        "top_level_field_count": 7,
        "terminal_field_count": 3,
        "ledger_field_count": 8,
        "execution_channel_field_counts_allowed": [0, 6, 7],
        "arbitrary_nested_fields_or_arrays_allowed": False,
        "failure_type_count_keys_bounded_by_event_cfe": True,
    }
    assert contract["worker_control_reports"] == {
        "applies_only_to_profile": "R8C_E1E2",
        "transport": "TASK_MANIFEST_COMMITTED_TASK_ARTIFACTS",
        "report_kind_to_filename": {
            "TASK_SUMMARY": "task_summary.json",
            "TASK_FAILURE": "task_failure.json",
            "SUPERVISOR_OUTCOME": "task_supervisor_outcome.json",
        },
        "maximum_canonical_report_bytes_including_lf": 65_536,
        "encoding": "UTF-8_CANONICAL_JSON_WITH_FINAL_LF",
        "exclusive_create_flush_and_fsync_required": True,
        "task_summary_statuses_allowed": [
            "COMPLETE",
            "INCOMPLETE_RESOURCE_CEILING",
        ],
        "only_complete_status_is_success": True,
        "failure_reason_code_equals_outcome_class": True,
        "failure_error_type_only_no_message": True,
        "recursive_forbidden_keys": [
            "candidate_id",
            "candidate_ids",
            "vector",
            "vectors",
            "objectives",
            "front_objectives",
            "constraints",
            "nhv",
            "auc",
            "negative_transfer",
            "effect_size",
            "p_value",
            "error",
            "message",
            "traceback",
        ],
        "root_relative_commitment_fields": [
            "kind",
            "path",
            "bytes",
            "sha256",
        ],
        "task_manifest_binding_required": True,
        "run_and_runtime_commitments_equal_and_cover_schedule": True,
        "launch_binding_transport_marker_required": True,
        "raw_worker_stdout_persisted": False,
        "raw_worker_stderr_persisted": False,
        "worker_logs_directory_created": False,
        "worker_log_commitments_allowed": False,
        "normal_success_additional_report_files": 0,
        "legacy_r8_worker_log_behavior_changed": False,
    }
    assert analysis["r8_computes_nHV"] is False
    assert analysis["r9_independently_reconstructs_nHV_and_AUC"] is True
    assert analysis["checkpoint_fronts_reconstruct_frozen_endpoints"] is True
    assert analysis["untruncated_front_or_internal_archive_reconstructable"] is False
    assert analysis["evaluation_stream_chain_reconstructs_raw_rows"] is False
    assert (
        analysis[
            "evaluation_stream_chain_verification_requires_deterministic_replay"
        ]
        is True
    )
    assert (
        contract["authority_boundary"][
            "clarifies_statistical_observation_archive"
        ]
        is True
    )


def test_pending_contract_binds_persistence_and_remains_no_go() -> None:
    pending = _read_json(PENDING_PATH)
    _validator(PENDING_SCHEMA_PATH).validate(pending)

    binding = pending["upstream"]["checkpoint_persistence"]
    assert binding == {
        "contract_id": PERSISTENCE_ID,
        "path": (
            "config/r8c_e1e2/"
            "r8c_e1e2_checkpoint_persistence_contract.json"
        ),
        "sha256": sha256(PERSISTENCE_PATH.read_bytes()).hexdigest(),
        "schema_path": (
            "config/r8c_e1e2/"
            "r8c_e1e2_checkpoint_persistence_contract.schema.json"
        ),
        "schema_sha256": sha256(
            PERSISTENCE_SCHEMA_PATH.read_bytes()
        ).hexdigest(),
    }
    assert pending["status"] == "TARGET_HOST_UNMEASURED"
    assert (
        pending["authorization"]["formal_effect_execution_authorized"]
        is False
    )
    assert pending["launch"]["command_executable_now"] is False
    assert pending["launch"]["formal_launch_prohibited"] is True
    assert (
        pending["fail_closed_gate"]["formal_launch_status"]
        == "PROHIBITED"
    )
    assert pending["resources"]["scratch"][
        "minimum_free_bytes_at_start"
    ] == 32 * 1024**3
    assert pending["resources"]["scratch"][
        "stop_dispatch_below_free_bytes"
    ] == 8 * 1024**3
    assert pending["resources"]["output"] == {
        "max_total_bytes": 12 * 1024**3,
        "control_plane_reserve_bytes": 64 * 1024**2,
        "max_inflight_write_bytes_per_worker": 8 * 1024**2,
        "raw_evaluations_required": False,
        "format": "WGT_CFE_CHECKPOINT_BINARY_V1_ENDPOINT_SUFFICIENT",
        "silent_truncation": False,
    }


def test_persistence_schema_rejects_scope_drift_raw_rows_and_extras() -> None:
    contract = _read_json(PERSISTENCE_PATH)
    validator = _validator(PERSISTENCE_SCHEMA_PATH)

    changed_endpoint = deepcopy(contract)
    changed_endpoint["amendment_scope"]["unchanged_research_identity"][
        "endpoints"
    ] = False
    assert list(validator.iter_errors(changed_endpoint))

    raw_rows = deepcopy(contract)
    raw_rows["formal_resource_envelope"][
        "raw_evaluations_required"
    ] = True
    assert list(validator.iter_errors(raw_rows))

    oversize_event = deepcopy(contract)
    oversize_event["event_summary_control"][
        "maximum_canonical_record_bytes_including_lf"
    ] = 16_384
    assert list(validator.iter_errors(oversize_event))

    raw_worker_log = deepcopy(contract)
    raw_worker_log["worker_control_reports"][
        "raw_worker_stdout_persisted"
    ] = True
    assert list(validator.iter_errors(raw_worker_log))

    extra = deepcopy(contract)
    extra["binary_format"]["unfrozen_payload"] = True
    assert list(validator.iter_errors(extra))


def test_qualified_schema_requires_exact_persistence_binding() -> None:
    schema = _read_json(QUALIFIED_SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    upstream = schema["properties"]["upstream"]
    assert "checkpoint_persistence" in upstream["required"]
    assert upstream["properties"]["checkpoint_persistence"] == {
        "$ref": "#/$defs/checkpoint_persistence_upstream"
    }
    binding = schema["$defs"]["checkpoint_persistence_upstream"]
    assert binding["additionalProperties"] is False
    assert binding["properties"]["contract_id"]["const"] == PERSISTENCE_ID
    assert binding["properties"]["sha256"]["const"] == sha256(
        PERSISTENCE_PATH.read_bytes()
    ).hexdigest()
    assert binding["properties"]["schema_path"]["const"] == (
        "config/r8c_e1e2/"
        "r8c_e1e2_checkpoint_persistence_contract.schema.json"
    )
    assert binding["properties"]["schema_sha256"]["const"] == sha256(
        PERSISTENCE_SCHEMA_PATH.read_bytes()
    ).hexdigest()
