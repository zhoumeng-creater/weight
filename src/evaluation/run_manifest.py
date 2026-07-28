"""Canonical R2 correctness-artifact manifest and raw-lock validation.

Provenance disposition:
    LEGACY_WEIGHT/result_schema.py -> L11 REWRITE_FROM_SPEC
    Source SHA-256 30aef49186224327b21c142579c20176251a11bb9fd874d8361930534230e3c9

No legacy visualization schema is copied. This module represents only
synthetic/public correctness artifacts and cannot authorize effect analysis.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

from dt_ramde_v11.contracts import (
    AlgorithmConfig,
    ConfigurationError,
    ContractBindings,
    ExecutionScope,
    R2ExecutionRequest,
)
from evaluation.contracts import TerminalCode


R2_MANIFEST_SCHEMA = "WGT-V11-R2-CORRECTNESS-MANIFEST-1.0.0"
R2_ARTIFACT_ROLE = "R2_CORRECTNESS_ARTIFACT_NOT_EFFECT_RESULT"
RAW_RESULT_NAME = "raw_result.json"
R2_DEPENDENCY_LOCK = {
    "path": "requirements-r2.lock",
    "sha256": (
        "e134d3baaedf570d2b44a4359c10bcf4b968bf10a6d87fbbf7776bc64eb44ee8"
    ),
    "target": "CPython_3.12_Windows_AMD64",
    "install_mode": "require_hashes_only_binary_no_index_verified",
}
R2_RUNTIME_PLATFORM = {
    "system": "Windows",
    "machine": "AMD64",
}
_DEPENDENCY_KEYS = frozenset(
    {
        "python_implementation",
        "python_version",
        "numpy_version",
        "dt_ramde_v11_version",
        "lock",
        "runtime_platform",
    }
)

R2_PERMISSION_KEYS = frozenset(
    {
        "participant_data_accessed",
        "effect_estimation",
        "hidden_generation",
        "confirmatory_execution",
        "results_writing",
        "effect_analysis",
        "remote_git_mutation",
        "release_or_distribution",
    }
)

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "artifact_role",
        "run_id",
        "run_binding_sha256",
        "protocol",
        "execution_scope",
        "fixture",
        "method",
        "adapter",
        "selector",
        "configuration",
        "dependencies",
        "randomness",
        "budget",
        "failures",
        "code",
        "permissions",
        "raw_artifact",
        "completion",
        "parent_run_id",
        "deviation_ids",
    }
)
_CODE_KEYS = frozenset(
    {
        "git_commit",
        "git_dirty",
        "source_files",
        "source_bundle_sha256",
    }
)
_REQUIRED_SOURCE_PATHS = frozenset(
    {
        "src/benchmark_adapters/public_cmop.py",
        "src/dt_ramde_v11/contracts.py",
        "src/dt_ramde_v11/engine.py",
        "src/evaluation/ledger.py",
        "src/evaluation/run_manifest.py",
        "src/weight_application/adapter.py",
        "tools/analyze_v11_results.py",
        "tools/run_v11_experiment.py",
    }
)
_ALLOWED_SOURCE_PACKAGES = frozenset(
    {
        "analysis",
        "benchmark_adapters",
        "comparators",
        "dt_ramde_v11",
        "e3_inputs",
        "evaluation",
        "formal_execution",
        "resource_pilot",
        "weight_application",
    }
)

_RAW_TOP_LEVEL_KEYS = frozenset(
    {"artifact_role", "fixture_id", "run_result"}
)
_RUN_RESULT_KEYS = frozenset(
    {
        "config",
        "adapter_identity",
        "selector_identity",
        "events",
        "persistent_state",
        "effect_estimation_performed",
        "hidden_seed_or_instance_generated",
        "confirmatory_execution",
    }
)
_EVENT_KEYS = frozenset(
    {
        "archive",
        "archive_audit",
        "credit_resolution_status",
        "event_id",
        "execution_feedback",
        "information_hash",
        "initialization_audit",
        "ledger",
        "lineage_records",
        "memory_snapshot",
        "mg_final",
        "reset_reason",
        "resolved_q",
        "state_transitions",
        "terminal",
        "trial_audit",
        "warm_start_seed_count",
    }
)
_LEDGER_KEYS = frozenset(
    {
        "cfe",
        "objective_calls",
        "constraint_calls",
        "scenario_evaluations",
        "atomic_model_steps",
        "execution_transition_count",
        "repair_failed",
        "evaluation_failures",
    }
)
_TERMINAL_KEYS = frozenset({"candidate_id", "code", "reason"})
_CONFIG_KEYS = frozenset(
    {
        field.name
        for field in fields(AlgorithmConfig)
        if field.name != "audit_materialization"
    }
    | {"variant_components"}
)
_REQUEST_KEYS = frozenset(
    field.name for field in fields(R2ExecutionRequest)
)
_CONTRACT_KEYS = frozenset(
    field.name for field in fields(ContractBindings)
)
_REQUEST_PERMISSION_KEYS = frozenset(
    {
        "participant_data_requested",
        "effect_estimation_requested",
        "hidden_generation_requested",
        "results_writing_requested",
        "remote_git_mutation_requested",
        "release_or_distribution_requested",
    }
)
_METHOD_KEYS = frozenset(
    {"method_id", "method_version", "role", "variant"}
)
_SELECTOR_KEYS = frozenset(
    {"selector_id", "selector_version", "role"}
)
_SYNTHETIC_ADAPTER_KEYS = frozenset(
    {
        "adapter_id",
        "adapter_version",
        "role",
        "participant_data_used",
        "virtual_human_claim",
        "effect_evidence",
        "model",
        "model_role",
    }
)
_PUBLIC_ADAPTER_KEYS = frozenset(
    {
        "target_suite_id",
        "target_problem_id",
        "split",
        "target_registered_split",
        "adapter_id",
        "adapter_version",
        "evaluator_interface_version",
        "fixture_evaluator_sha256",
        "bridge_role",
        "registered_effect_instance",
        "formal_effect_execution_allowed",
    }
)
_FIXTURE_BINDINGS = {
    "synthetic_weight_e0": {
        "scope": ExecutionScope.UNIT_TEST_FIXTURE,
        "evidence": "UNIT_TEST_FIXTURE",
        "adapter_id": "WGT-V11-SYNTHETIC-E0",
        "adapter_version": "1.1.0-r2-fixture",
    },
    "static_bridge_e0": {
        "scope": ExecutionScope.PUBLIC_CORRECTNESS_FIXTURE,
        "evidence": "PUBLIC_CORRECTNESS_FIXTURE",
        "adapter_id": "BIND-STATIC-CMOP-01/R2-CORRECTNESS-BRIDGE",
        "adapter_version": "1.0.0-r2-fixture",
    },
}
_FIXTURE_VALUES = {
    "synthetic_weight_e0": {
        "fixture_id": "synthetic_weight_e0",
        "role": "synthetic_energy_mass_correctness",
        "initial_state": {
            "event_id": 0,
            "fat_mass_kg": 24.0,
            "lean_mass_kg": 56.0,
            "body_mass_kg": 80.0,
            "cumulative_energy_imbalance_kcal": 0.0,
        },
        "target_mass_kg": 77.0,
        "model": {
            "model_id": "WGT-E0-LINEAR-ENERGY-MASS-FIXTURE",
            "model_version": "1.0.0",
            "qualification_status": "NOT_QUALIFIED_E0_CORRECTNESS_ONLY",
            "event_days": 7.0,
            "energy_density_kcal_per_kg": 7700.0,
            "fat_mass_change_fraction": 0.75,
            "action_units": [
                "intake_adjustment_kcal_per_day",
                "activity_expenditure_adjustment_kcal_per_day",
            ],
        },
    },
    "static_bridge_e0": {
        "fixture_id": "static_bridge_e0",
        "target_suite_id": "DAS-CMOP-PLATEMO-4.15",
        "target_problem_id": "DASCMOP1",
        "decision_dimension": 30,
        "equations": [
            "f1=sum(x_i^2)",
            "f2=sum((x_i-0.25)^2)",
            "g1=mean(x_i)-0.75<=0",
        ],
        "formal_public_instance": False,
    },
}
_E0_MODEL_ROLE = {
    "binding_id": "WGT-V11-E0-ROLE-01",
    "model_id": "WGT-E0-LINEAR-ENERGY-MASS-FIXTURE",
    "role": "M_P",
    "qualification_status": "NOT_QUALIFIED_E0_CORRECTNESS_ONLY",
    "allowed_scope": "unit_test_fixture",
    "participant_data_allowed": False,
    "effect_estimation_allowed": False,
    "scientific_model_gate": "F09_BLOCKED_PENDING_R3_QUALIFICATION",
}
_PROHIBITED_RAW_PAYLOAD_KEYS = frozenset(
    {
        "results",
        "effect_estimate",
        "effect_estimates",
        "participant_rows",
        "participant_records",
        "participant_data",
        "hidden_content",
        "other_method_results",
    }
)


class ManifestIntegrityError(RuntimeError):
    """A correctness manifest or its raw artifact failed authentication."""


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON with a single deterministic representation."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ManifestIntegrityError(
            "value is not canonical finite JSON"
        ) from error


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_value(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _run_binding_payload(
    *,
    protocol: Mapping[str, Any],
    execution_scope: str,
    fixture_sha256: str,
    method: Mapping[str, Any],
    adapter_sha256: str,
    selector_sha256: str,
    configuration_sha256: str,
    dependency_sha256: str,
    randomness: Mapping[str, Any],
    budget: Mapping[str, Any],
    failures: Mapping[str, Any],
    code: Mapping[str, Any],
    permissions: Mapping[str, Any],
    raw_artifact: Mapping[str, Any],
    completion: str,
    parent_run_id: str | None,
    deviation_ids: list[str],
) -> dict[str, Any]:
    return {
        "protocol": dict(protocol),
        "execution_scope": execution_scope,
        "fixture_sha256": fixture_sha256,
        "method": dict(method),
        "adapter_sha256": adapter_sha256,
        "selector_sha256": selector_sha256,
        "configuration_sha256": configuration_sha256,
        "dependency_sha256": dependency_sha256,
        "randomness": dict(randomness),
        "budget": dict(budget),
        "failures": dict(failures),
        "code": {
            "git_commit": code.get("git_commit"),
            "git_dirty": code.get("git_dirty"),
            "source_bundle_sha256": code.get("source_bundle_sha256"),
        },
        "permissions": dict(permissions),
        "raw_artifact": dict(raw_artifact),
        "completion": completion,
        "parent_run_id": parent_run_id,
        "deviation_ids": list(deviation_ids),
    }


def build_r2_manifest(
    *,
    execution_scope: ExecutionScope,
    fixture: Mapping[str, Any],
    method: Mapping[str, Any],
    adapter_identity: Mapping[str, Any],
    selector_identity: Mapping[str, Any],
    configuration: Mapping[str, Any],
    dependencies: Mapping[str, Any],
    randomness: Mapping[str, Any],
    budget: Mapping[str, Any],
    failures: Mapping[str, Any],
    code: Mapping[str, Any],
    raw_sha256: str,
    raw_bytes: int,
) -> dict[str, Any]:
    """Build a deterministic, explicitly non-effect R2 manifest."""

    if execution_scope not in {
        ExecutionScope.UNIT_TEST_FIXTURE,
        ExecutionScope.PUBLIC_CORRECTNESS_FIXTURE,
    }:
        raise ManifestIntegrityError(
            "manifest scope is outside R2 correctness fixtures"
        )
    fixture_value = dict(fixture)
    method_value = dict(method)
    adapter_value = dict(adapter_identity)
    selector_value = dict(selector_identity)
    config_value = dict(configuration)
    dependency_value = dict(dependencies)
    randomness_value = dict(randomness)
    budget_value = dict(budget)
    failure_value = dict(failures)
    code_value = dict(code)
    _validate_dependency_identity(dependency_value)

    fixture_block = {
        **fixture_value,
        "sha256": _hash_value(fixture_value),
    }
    adapter_block = {
        "identity": adapter_value,
        "sha256": _hash_value(adapter_value),
    }
    selector_block = {
        "identity": selector_value,
        "sha256": _hash_value(selector_value),
    }
    configuration_block = {
        "value": config_value,
        "sha256": _hash_value(config_value),
    }
    dependency_block = {
        "value": dependency_value,
        "sha256": _hash_value(dependency_value),
    }
    source_files = code_value.get("source_files")
    if not isinstance(source_files, Mapping) or not source_files:
        raise ManifestIntegrityError(
            "code identity requires source file hashes"
        )
    source_bundle_sha256 = _hash_value(dict(source_files))
    code_value["source_bundle_sha256"] = source_bundle_sha256
    _validate_code_identity(code_value)
    permissions = {
        key: False for key in sorted(R2_PERMISSION_KEYS)
    }
    raw_artifact = {
        "path": RAW_RESULT_NAME,
        "sha256": raw_sha256,
        "bytes": int(raw_bytes),
    }
    completion = "COMPLETE_R2_CORRECTNESS_FIXTURE"
    deviation_ids = ["DEV-R2-001"]
    binding = _run_binding_payload(
        protocol=ContractBindings().to_dict(),
        execution_scope=execution_scope.value,
        fixture_sha256=fixture_block["sha256"],
        method=method_value,
        adapter_sha256=adapter_block["sha256"],
        selector_sha256=selector_block["sha256"],
        configuration_sha256=configuration_block["sha256"],
        dependency_sha256=dependency_block["sha256"],
        randomness=randomness_value,
        budget=budget_value,
        failures=failure_value,
        code=code_value,
        permissions=permissions,
        raw_artifact=raw_artifact,
        completion=completion,
        parent_run_id=None,
        deviation_ids=deviation_ids,
    )
    binding_sha256 = _hash_value(binding)
    return {
        "schema_version": R2_MANIFEST_SCHEMA,
        "artifact_role": R2_ARTIFACT_ROLE,
        "run_id": f"r2-{binding_sha256[:24]}",
        "run_binding_sha256": binding_sha256,
        "protocol": ContractBindings().to_dict(),
        "execution_scope": execution_scope.value,
        "fixture": fixture_block,
        "method": method_value,
        "adapter": adapter_block,
        "selector": selector_block,
        "configuration": configuration_block,
        "dependencies": dependency_block,
        "randomness": randomness_value,
        "budget": budget_value,
        "failures": failure_value,
        "code": code_value,
        "permissions": permissions,
        "raw_artifact": raw_artifact,
        "completion": completion,
        "parent_run_id": None,
        "deviation_ids": deviation_ids,
    }


def _require_hash(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ManifestIntegrityError(f"{label} is not a lowercase SHA-256")
    return value


def _validate_dependency_identity(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != _DEPENDENCY_KEYS:
        raise ManifestIntegrityError(
            "dependency identity does not bind the complete R2 dependency lock"
        )
    if value["python_implementation"] != "CPython":
        raise ManifestIntegrityError(
            "dependency identity requires CPython"
        )
    python_version = value["python_version"]
    if (
        not isinstance(python_version, str)
        or not python_version.startswith("3.12.")
    ):
        raise ManifestIntegrityError(
            "dependency identity is outside the locked Python target"
        )
    if value["numpy_version"] != "1.26.4":
        raise ManifestIntegrityError(
            "dependency identity differs from the R2 dependency lock"
        )
    if value["dt_ramde_v11_version"] != "0.1.0.dev0":
        raise ManifestIntegrityError(
            "dependency identity has an unexpected package version"
        )
    if value["lock"] != R2_DEPENDENCY_LOCK:
        raise ManifestIntegrityError(
            "dependency lock binding is invalid"
        )
    if value["runtime_platform"] != R2_RUNTIME_PLATFORM:
        raise ManifestIntegrityError(
            "dependency runtime platform is outside the locked target"
        )


def _validate_hashed_block(
    block: Any,
    *,
    value_key: str,
    label: str,
) -> None:
    if not isinstance(block, Mapping):
        raise ManifestIntegrityError(f"{label} block is invalid")
    expected = _require_hash(block.get("sha256"), label=f"{label} hash")
    if expected != _hash_value(block.get(value_key)):
        raise ManifestIntegrityError(f"{label} hash differs from its value")


def _validate_code_identity(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != _CODE_KEYS:
        raise ManifestIntegrityError("manifest code block schema is invalid")
    commit = value.get("git_commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ManifestIntegrityError(
            "manifest git commit is not a lowercase 40-character identity"
        )
    if type(value.get("git_dirty")) is not bool:
        raise ManifestIntegrityError(
            "manifest git dirty flag is invalid"
        )
    source_files = value.get("source_files")
    if not isinstance(source_files, dict) or not source_files:
        raise ManifestIntegrityError(
            "manifest source-file identity map is invalid"
        )
    for source_path, source_sha256 in source_files.items():
        if not isinstance(source_path, str):
            raise ManifestIntegrityError(
                "manifest source path is not a string"
            )
        parsed = PurePosixPath(source_path)
        parts = parsed.parts
        normalized = parsed.as_posix()
        source_is_allowed = (
            len(parts) >= 3
            and parts[0] == "src"
            and parts[1] in _ALLOWED_SOURCE_PACKAGES
            and parsed.suffix == ".py"
        ) or source_path in {
            "tools/analyze_v11_results.py",
            "tools/run_v11_experiment.py",
        }
        if (
            source_path != normalized
            or source_path.startswith("/")
            or "\\" in source_path
            or any(part in {"", ".", ".."} for part in parts)
            or not source_is_allowed
        ):
            raise ManifestIntegrityError(
                f"manifest source path is invalid: {source_path!r}"
            )
        _require_hash(
            source_sha256,
            label=f"source file {source_path!r} hash",
        )
    if not _REQUIRED_SOURCE_PATHS.issubset(source_files):
        raise ManifestIntegrityError(
            "manifest source map omits an R2-critical source path"
        )
    if _require_hash(
        value.get("source_bundle_sha256"),
        label="source bundle hash",
    ) != _hash_value(source_files):
        raise ManifestIntegrityError("source bundle hash is invalid")


def _read_canonical_json(path: Path, *, label: str) -> Any:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ManifestIntegrityError(
            f"{label} is not readable JSON"
        ) from error
    if payload != canonical_json_bytes(value) + b"\n":
        raise ManifestIntegrityError(
            f"{label} bytes are not the canonical JSON representation"
        )
    return value


def _reject_prohibited_raw_payload(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if (
                isinstance(key, str)
                and key.casefold() in _PROHIBITED_RAW_PAYLOAD_KEYS
            ):
                raise ManifestIntegrityError(
                    f"raw artifact contains prohibited payload key {key!r}"
                )
            _reject_prohibited_raw_payload(item)
    elif isinstance(value, list):
        for item in value:
            _reject_prohibited_raw_payload(item)


def _reconstruct_config(value: Any) -> AlgorithmConfig:
    if not isinstance(value, dict) or set(value) != _CONFIG_KEYS:
        raise ManifestIntegrityError(
            "configuration schema is invalid"
        )
    request_value = value.get("execution_request")
    if (
        not isinstance(request_value, dict)
        or set(request_value) != _REQUEST_KEYS
    ):
        raise ManifestIntegrityError(
            "configuration execution-request schema is invalid"
        )
    if any(
        type(request_value.get(key)) is not bool
        for key in _REQUEST_PERMISSION_KEYS
    ):
        raise ManifestIntegrityError(
            "configuration permission flags must be booleans"
        )
    contract_value = request_value.get("contracts")
    if (
        not isinstance(contract_value, dict)
        or set(contract_value) != _CONTRACT_KEYS
        or any(
            not isinstance(item, str) or not item
            for item in contract_value.values()
        )
    ):
        raise ManifestIntegrityError(
            "configuration contract binding schema is invalid"
        )
    integer_fields = {
        "population_size",
        "cfe_per_event",
        "algorithm_seed",
        "max_events",
        "atomic_steps_per_evaluation",
    }
    string_fields = {
        "variant",
        "timing_mode",
        "method_label",
        "adapter_id",
        "adapter_version",
        "selector_id",
        "selector_version",
        "configuration_evidence_id",
    }
    if any(type(value.get(key)) is not int for key in integer_fields):
        raise ManifestIntegrityError(
            "configuration integer field type is invalid"
        )
    if any(
        not isinstance(value.get(key), str) or not value[key]
        for key in string_fields
    ):
        raise ManifestIntegrityError(
            "configuration identity field type is invalid"
        )
    event_limit = value.get("event_time_limit_seconds")
    if (
        type(event_limit) not in {int, float}
        or isinstance(event_limit, bool)
    ):
        raise ManifestIntegrityError(
            "configuration event-time field type is invalid"
        )
    try:
        contracts = ContractBindings(**contract_value)
        request = R2ExecutionRequest(
            scope=ExecutionScope(request_value["scope"]),
            contracts=contracts,
            **{
                key: request_value[key]
                for key in _REQUEST_PERMISSION_KEYS
            },
        )
        config = AlgorithmConfig(
            **{
                field.name: (
                    request
                    if field.name == "execution_request"
                    else value[field.name]
                )
                for field in fields(AlgorithmConfig)
                if field.name != "audit_materialization"
            }
        )
        config.validate()
    except (KeyError, TypeError, ValueError, ConfigurationError) as error:
        raise ManifestIntegrityError(
            "configuration requests a prohibited permission or is invalid: "
            f"{error}"
        ) from error
    if config.to_dict() != value:
        raise ManifestIntegrityError(
            "configuration differs from reconstructed canonical binding"
        )
    return config


def _validate_identity_bindings(
    *,
    manifest: Mapping[str, Any],
    config: AlgorithmConfig,
    fixture: Mapping[str, Any],
) -> None:
    fixture_id = fixture.get("fixture_id")
    binding = _FIXTURE_BINDINGS.get(fixture_id)
    if binding is None:
        raise ManifestIntegrityError("fixture identity is not registered for R2")
    fixture_value = {
        key: value for key, value in fixture.items() if key != "sha256"
    }
    if fixture_value != _FIXTURE_VALUES[fixture_id]:
        raise ManifestIntegrityError(
            "fixture schema or registered correctness identity is invalid"
        )
    if (
        manifest["execution_scope"] != config.execution_request.scope.value
        or config.execution_request.scope is not binding["scope"]
    ):
        raise ManifestIntegrityError(
            "fixture, manifest, and configuration scope bindings disagree"
        )
    if config.configuration_evidence_id != binding["evidence"]:
        raise ManifestIntegrityError(
            "fixture scope and configuration evidence binding disagree"
        )

    method = manifest["method"]
    if (
        not isinstance(method, dict)
        or set(method) != _METHOD_KEYS
        or method.get("method_id") != config.method_label
        or method.get("variant") != config.variant
        or method.get("role") != "R2_correctness_only"
        or not isinstance(method.get("method_version"), str)
        or not method["method_version"]
    ):
        raise ManifestIntegrityError(
            "method identity differs from configuration binding"
        )
    selector = manifest["selector"]["identity"]
    if (
        not isinstance(selector, dict)
        or set(selector) != _SELECTOR_KEYS
        or selector.get("selector_id") != config.selector_id
        or selector.get("selector_version") != config.selector_version
        or selector.get("role") != "R2_correctness_fixture_only"
    ):
        raise ManifestIntegrityError(
            "selector identity differs from configuration binding"
        )
    adapter = manifest["adapter"]["identity"]
    if not isinstance(adapter, dict):
        raise ManifestIntegrityError("adapter identity schema is invalid")
    if (
        adapter.get("adapter_id") != config.adapter_id
        or adapter.get("adapter_version") != config.adapter_version
        or config.adapter_id != binding["adapter_id"]
        or config.adapter_version != binding["adapter_version"]
    ):
        raise ManifestIntegrityError(
            "adapter identity differs from configuration binding"
        )
    if fixture_id == "synthetic_weight_e0":
        if (
            set(adapter) != _SYNTHETIC_ADAPTER_KEYS
            or adapter.get("role")
            != "supportive_synthetic_E0_correctness_fixture"
            or adapter.get("model") != fixture_value["model"]
            or adapter.get("model_role") != _E0_MODEL_ROLE
            or any(
                adapter.get(key) is not False
                for key in (
                    "participant_data_used",
                    "virtual_human_claim",
                    "effect_evidence",
                )
            )
        ):
            raise ManifestIntegrityError(
                "synthetic adapter model role contains prohibited effect, "
                "participant, or qualification claims"
            )
    elif (
        set(adapter) != _PUBLIC_ADAPTER_KEYS
        or adapter.get("split")
        != "r2_public_bridge_correctness_fixture"
        or adapter.get("target_suite_id")
        != fixture_value["target_suite_id"]
        or adapter.get("target_problem_id")
        != fixture_value["target_problem_id"]
        or adapter.get("fixture_evaluator_sha256")
        != fixture.get("sha256")
        or adapter.get("target_registered_split")
        != "public_fixed_confirmatory"
        or adapter.get("evaluator_interface_version")
        != "STATIC-CMOP-EVAL-1.0.0"
        or adapter.get("bridge_role")
        != "r2_result_blind_fixture_bridge"
        or any(
            adapter.get(key) is not False
            for key in (
                "registered_effect_instance",
                "formal_effect_execution_allowed",
            )
        )
    ):
        raise ManifestIntegrityError(
            "public adapter is not a fail-closed R2 correctness bridge"
        )


def _validate_execution_feedback(
    feedback: Any,
    *,
    event_id: int,
    config: AlgorithmConfig,
) -> None:
    if config.variant == "NO_EXECUTION_FEEDBACK":
        if feedback is not None:
            raise ManifestIntegrityError(
                "NO_EXECUTION_FEEDBACK must serialize execution feedback as null"
            )
        return
    if not isinstance(feedback, dict):
        raise ManifestIntegrityError(
            f"{config.variant} requires complete execution feedback"
        )
    required_keys = {
        "available",
        "ell_exec",
        "ell_ref",
        "s_exec",
        "hard_constraint_violation",
        "released_at",
    }
    if not required_keys.issubset(feedback):
        raise ManifestIntegrityError(
            f"{config.variant} requires complete execution feedback"
        )
    if (
        type(feedback["available"]) is not bool
        or type(feedback["released_at"]) is not int
        or feedback["released_at"] != event_id + 1
    ):
        raise ManifestIntegrityError(
            f"{config.variant} execution feedback timing is invalid"
        )
    if feedback["available"]:
        numeric_fields = ("ell_exec", "ell_ref", "s_exec")
        if any(
            isinstance(feedback[field], bool)
            or not isinstance(feedback[field], (int, float))
            or not math.isfinite(float(feedback[field]))
            for field in numeric_fields
        ):
            raise ManifestIntegrityError(
                f"{config.variant} execution feedback values are invalid"
            )
        if (
            float(feedback["s_exec"]) <= 0.0
            or type(feedback["hard_constraint_violation"]) is not bool
        ):
            raise ManifestIntegrityError(
                f"{config.variant} execution feedback values are invalid"
            )
    elif any(
        feedback[field] is not None
        for field in (
            "ell_exec",
            "ell_ref",
            "s_exec",
            "hard_constraint_violation",
        )
    ):
        raise ManifestIntegrityError(
            f"{config.variant} unavailable execution feedback is invalid"
        )


def _validate_event_ledgers(
    events: Any,
    *,
    config: AlgorithmConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(events, list) or len(events) != config.max_events:
        raise ManifestIntegrityError(
            "raw artifact event count differs from configuration budget"
        )
    per_event: list[dict[str, Any]] = []
    terminal_counts: Counter[str] = Counter()
    evaluation_failures = 0
    repair_failures = 0
    for expected_event_id, event in enumerate(events):
        if not isinstance(event, dict) or set(event) != _EVENT_KEYS:
            raise ManifestIntegrityError("raw artifact event schema is invalid")
        event_id = event.get("event_id")
        ledger = event.get("ledger")
        terminal = event.get("terminal")
        if type(event_id) is not int or event_id != expected_event_id:
            raise ManifestIntegrityError(
                "raw artifact event sequence is invalid"
            )
        _validate_execution_feedback(
            event["execution_feedback"],
            event_id=event_id,
            config=config,
        )
        if (
            not isinstance(ledger, dict)
            or set(ledger) != _LEDGER_KEYS
            or any(type(value) is not int or value < 0 for value in ledger.values())
        ):
            raise ManifestIntegrityError(
                "raw artifact budget ledger schema is invalid"
            )
        if not (
            ledger["cfe"]
            == ledger["objective_calls"]
            == ledger["constraint_calls"]
            == ledger["scenario_evaluations"]
        ):
            raise ManifestIntegrityError(
                "raw artifact joint objective/constraint budget ledgers disagree"
            )
        if ledger["cfe"] > config.cfe_per_event:
            raise ManifestIntegrityError(
                "raw artifact CFE exceeds the configured event budget"
            )
        expected_transitions = (
            1
            if config.timing_mode == "TS2_fixed_periodic_replanning"
            else 0
        )
        if ledger["execution_transition_count"] != expected_transitions:
            raise ManifestIntegrityError(
                "raw artifact execution-transition ledger differs from timing mode"
            )
        if ledger["evaluation_failures"] > ledger["cfe"]:
            raise ManifestIntegrityError(
                "raw artifact evaluation-failure count exceeds charged CFE"
            )
        if (
            ledger["atomic_model_steps"]
            != ledger["cfe"] * config.atomic_steps_per_evaluation
        ):
            raise ManifestIntegrityError(
                "raw artifact atomic-step ledger differs from configuration"
            )
        if (
            not isinstance(terminal, dict)
            or set(terminal) != _TERMINAL_KEYS
            or not isinstance(terminal.get("code"), str)
            or not terminal["code"]
        ):
            raise ManifestIntegrityError(
                "raw artifact terminal record is invalid"
            )
        try:
            terminal_code = TerminalCode(terminal["code"])
        except ValueError as error:
            raise ManifestIntegrityError(
                "raw artifact terminal code is not registered"
            ) from error
        if (
            terminal_code is TerminalCode.ACCEPTED
            and (
                not isinstance(terminal.get("candidate_id"), str)
                or not terminal["candidate_id"]
            )
        ):
            raise ManifestIntegrityError(
                "accepted raw artifact terminal lacks a candidate identity"
            )
        per_event.append({"event_id": event_id, **ledger})
        terminal_counts[terminal_code.value] += 1
        evaluation_failures += ledger["evaluation_failures"]
        repair_failures += ledger["repair_failed"]
    return (
        {
            "event_count": len(events),
            "per_event": per_event,
            "unused_budget_transfer": False,
        },
        {
            "terminal_counts": dict(sorted(terminal_counts.items())),
            "evaluation_failures": evaluation_failures,
            "repair_failures": repair_failures,
            "silent_retry": False,
        },
    )


def validate_r2_manifest(manifest_path: Path) -> dict[str, Any]:
    """Read and authenticate one R2 correctness manifest and raw artifact."""

    path = Path(manifest_path)
    if not path.is_file():
        raise ManifestIntegrityError("manifest file does not exist")
    manifest = _read_canonical_json(path, label="manifest")
    if not isinstance(manifest, dict) or set(manifest) != _TOP_LEVEL_KEYS:
        raise ManifestIntegrityError("manifest top-level schema is invalid")
    if manifest["schema_version"] != R2_MANIFEST_SCHEMA:
        raise ManifestIntegrityError("manifest schema version is invalid")
    if manifest["artifact_role"] != R2_ARTIFACT_ROLE:
        raise ManifestIntegrityError("manifest artifact role is invalid")
    if manifest["protocol"] != ContractBindings().to_dict():
        raise ManifestIntegrityError("manifest protocol binding is invalid")
    if manifest["execution_scope"] not in {
        ExecutionScope.UNIT_TEST_FIXTURE.value,
        ExecutionScope.PUBLIC_CORRECTNESS_FIXTURE.value,
    }:
        raise ManifestIntegrityError("manifest execution scope is invalid")

    permissions = manifest["permissions"]
    if (
        not isinstance(permissions, dict)
        or set(permissions) != R2_PERMISSION_KEYS
        or any(value is not False for value in permissions.values())
    ):
        raise ManifestIntegrityError(
            "manifest permission block is not fail-closed"
        )
    _validate_hashed_block(
        manifest["configuration"],
        value_key="value",
        label="configuration",
    )
    _validate_hashed_block(
        manifest["dependencies"],
        value_key="value",
        label="dependency",
    )
    _validate_dependency_identity(manifest["dependencies"]["value"])
    _validate_hashed_block(
        manifest["adapter"],
        value_key="identity",
        label="adapter",
    )
    _validate_hashed_block(
        manifest["selector"],
        value_key="identity",
        label="selector",
    )
    fixture = manifest["fixture"]
    if not isinstance(fixture, dict) or "fixture_id" not in fixture:
        raise ManifestIntegrityError("manifest fixture block is invalid")
    fixture_value = {
        key: value for key, value in fixture.items() if key != "sha256"
    }
    if _require_hash(
        fixture.get("sha256"), label="fixture hash"
    ) != _hash_value(fixture_value):
        raise ManifestIntegrityError("fixture hash differs from its value")

    code = manifest["code"]
    _validate_code_identity(code)

    raw = manifest["raw_artifact"]
    if not isinstance(raw, dict) or set(raw) != {
        "path",
        "sha256",
        "bytes",
    }:
        raise ManifestIntegrityError("raw artifact block is invalid")
    if raw["path"] != RAW_RESULT_NAME:
        raise ManifestIntegrityError("raw artifact path is invalid")
    raw_path = path.parent / RAW_RESULT_NAME
    if not raw_path.is_file():
        raise ManifestIntegrityError("raw artifact file does not exist")
    if (
        type(raw["bytes"]) is not int
        or raw["bytes"] < 1
        or raw_path.stat().st_size != raw["bytes"]
    ):
        raise ManifestIntegrityError("raw artifact byte count is invalid")
    if _require_hash(
        raw["sha256"], label="raw artifact hash"
    ) != sha256_file(raw_path):
        raise ManifestIntegrityError("raw artifact hash is invalid")
    raw_payload = _read_canonical_json(
        raw_path,
        label="raw artifact",
    )
    if (
        not isinstance(raw_payload, dict)
        or set(raw_payload) != _RAW_TOP_LEVEL_KEYS
        or raw_payload.get("artifact_role") != R2_ARTIFACT_ROLE
        or raw_payload.get("fixture_id") != fixture["fixture_id"]
    ):
        raise ManifestIntegrityError(
            "raw artifact schema or identity is invalid"
        )
    _reject_prohibited_raw_payload(raw_payload)
    run_result = raw_payload.get("run_result")
    if (
        not isinstance(run_result, dict)
        or set(run_result) != _RUN_RESULT_KEYS
    ):
        raise ManifestIntegrityError("raw artifact run-result schema is invalid")
    if any(
        run_result[name] is not False
        for name in (
            "effect_estimation_performed",
            "hidden_seed_or_instance_generated",
            "confirmatory_execution",
        )
    ):
        raise ManifestIntegrityError(
            "raw artifact contains prohibited execution flags"
        )
    if (
        run_result.get("config") != manifest["configuration"]["value"]
        or run_result.get("adapter_identity")
        != manifest["adapter"]["identity"]
        or run_result.get("selector_identity")
        != manifest["selector"]["identity"]
    ):
        raise ManifestIntegrityError(
            "raw artifact differs from manifest configuration binding"
        )
    config = _reconstruct_config(manifest["configuration"]["value"])
    _validate_identity_bindings(
        manifest=manifest,
        config=config,
        fixture=fixture,
    )
    expected_budget, expected_failures = _validate_event_ledgers(
        run_result["events"],
        config=config,
    )
    if manifest["budget"] != expected_budget:
        raise ManifestIntegrityError(
            "budget differs from raw artifact ledger binding"
        )
    if manifest["failures"] != expected_failures:
        raise ManifestIntegrityError(
            "failures differ from raw artifact ledger binding"
        )
    method = manifest["method"]
    randomness = manifest["randomness"]
    if (
        randomness.get("algorithm_seed") != config.algorithm_seed
        or randomness.get("paired_fixture_seed") != config.algorithm_seed
        or randomness.get("hidden_seed_used") is not False
    ):
        raise ManifestIntegrityError(
            "method/randomness differs from configuration binding"
        )
    if (
        manifest["completion"] != "COMPLETE_R2_CORRECTNESS_FIXTURE"
        or manifest["parent_run_id"] is not None
        or manifest["deviation_ids"] != ["DEV-R2-001"]
    ):
        raise ManifestIntegrityError("manifest completion binding is invalid")
    binding = _run_binding_payload(
        protocol=manifest["protocol"],
        execution_scope=manifest["execution_scope"],
        fixture_sha256=fixture["sha256"],
        method=method,
        adapter_sha256=manifest["adapter"]["sha256"],
        selector_sha256=manifest["selector"]["sha256"],
        configuration_sha256=manifest["configuration"]["sha256"],
        dependency_sha256=manifest["dependencies"]["sha256"],
        randomness=randomness,
        budget=manifest["budget"],
        failures=manifest["failures"],
        code=code,
        permissions=permissions,
        raw_artifact=raw,
        completion=manifest["completion"],
        parent_run_id=manifest["parent_run_id"],
        deviation_ids=manifest["deviation_ids"],
    )
    binding_sha256 = _hash_value(binding)
    if _require_hash(
        manifest["run_binding_sha256"],
        label="run binding hash",
    ) != binding_sha256:
        raise ManifestIntegrityError("run binding hash is invalid")
    if manifest["run_id"] != f"r2-{binding_sha256[:24]}":
        raise ManifestIntegrityError("run_id differs from run binding")
    return manifest
