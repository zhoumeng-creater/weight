"""One-time aggregate-only V11-MQ1 participant-data qualification runner.

Do not run this command without the separately confirmed exact request. The
runner consumes its execution identity before opening the SAS member and never
writes participant rows or raw identifiers.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator


EXECUTION_ID = "WGT-V11-MQ1-EXECUTION-20260724-01"
RESULT_ID = "WGT-V11-MQ1-RESULT-20260724-01"
CONSUMPTION_ID = "WGT-V11-MQ1-CONSUMPTION-20260724-01"
PROTOCOL_VERSION = "v1.2.0-r3-v11mq1-frozen"
CONTRACT_ID = "WGT-V11-MQ1-MODEL-QUALIFICATION-01"
NONPASS_CASE_NAME = "illustrative mechanistic simulation"
SOURCE_TABLE = "full0_18nih.sas7bdat"
UQ_STATUS = "NOT_QUALIFIED_NO_INDEPENDENT_CALIBRATION_SET"
CORRECTIVE_REQUEST_ID = "WGT-V11-MQ1-REQUEST-20260724-02"
REQUEST_SCHEMA_NAMES = {
    "WGT-V11-MQ1-REQUEST-20260724-01": (
        "v11_mq1_execution_request.schema.json"
    ),
    CORRECTIVE_REQUEST_ID: "v11_mq1_execution_request_02.schema.json",
}
CORRECTIVE_ENVIRONMENT_MANIFEST = (
    "v11_mq1_corrective_environment_manifest.json"
)
CORRECTIVE_ENVIRONMENT_SCHEMA = (
    "v11_mq1_corrective_environment_manifest.schema.json"
)
CORRECTIVE_AUTHORIZATION_RECORD = (
    "v11_mq1_launcher_corrective_authorization_record.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the one-time aggregate-only V11-MQ1 qualification"
    )
    parser.add_argument("--request", required=True)
    return parser


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("machine record must be a JSON object")
    return value


def _validate(value: Mapping[str, Any], schema_path: Path) -> None:
    schema = _read_json(schema_path)
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda error: list(error.path),
    )
    if errors:
        raise RuntimeError(
            f"{schema_path.name} validation failed: {errors[0].message}"
        )


def _request_schema_path(
    gate_root: Path,
    request: Mapping[str, Any],
) -> Path:
    request_id = request.get("request_id")
    if not isinstance(request_id, str):
        raise RuntimeError("request_id must be present before schema selection")
    schema_name = REQUEST_SCHEMA_NAMES.get(request_id)
    if schema_name is None:
        raise RuntimeError("unrecognized execution request identity")
    return gate_root / schema_name


def _absolute_file(raw: str, label: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        raise RuntimeError(f"{label} must be an absolute path")
    resolved = path.resolve()
    if not resolved.is_file():
        raise RuntimeError(f"{label} is missing")
    return resolved


def _same_path(first: Path, second: Path) -> bool:
    return os.path.normcase(str(first.resolve())) == os.path.normcase(
        str(second.resolve())
    )


def _validate_corrective_environment(
    request: Mapping[str, Any],
    gate_root: Path,
    project_root: Path,
    commit: str,
    tree: str,
    qualification_lock_sha256: str,
) -> None:
    if request["request_id"] != CORRECTIVE_REQUEST_ID:
        return

    manifest_path = gate_root / CORRECTIVE_ENVIRONMENT_MANIFEST
    manifest_schema = gate_root / CORRECTIVE_ENVIRONMENT_SCHEMA
    corrective_authorization = gate_root / CORRECTIVE_AUTHORIZATION_RECORD
    for path in (
        manifest_path,
        manifest_schema,
        corrective_authorization,
    ):
        if not path.is_file():
            raise RuntimeError(
                f"required corrective artifact is missing: {path.name}"
            )
    if _sha256(manifest_path) != request["environment_manifest_sha256"]:
        raise RuntimeError("corrective environment manifest SHA-256 mismatch")
    if (
        _sha256(corrective_authorization)
        != request["corrective_authorization_record_sha256"]
    ):
        raise RuntimeError(
            "launcher corrective authorization SHA-256 mismatch"
        )

    manifest = _read_json(manifest_path)
    _validate(manifest, manifest_schema)
    expected_python = _absolute_file(
        manifest["python"]["executable"],
        "corrective Python executable",
    )
    current_python = Path(sys.executable).resolve()
    if not _same_path(current_python, expected_python):
        raise RuntimeError("launcher Python executable mismatch")
    if _sha256(current_python) != manifest["python"]["sha256"]:
        raise RuntimeError("launcher Python executable SHA-256 mismatch")

    project_wheel = _absolute_file(
        manifest["project_wheel"]["path"],
        "corrective project wheel",
    )
    if _sha256(project_wheel) != manifest["project_wheel"]["sha256"]:
        raise RuntimeError("corrective project wheel SHA-256 mismatch")
    if manifest["implementation"] != {
        "commit": commit,
        "tree": tree,
        "qualification_lock_sha256": qualification_lock_sha256,
    }:
        raise RuntimeError(
            "corrective environment implementation identity mismatch"
        )
    for distribution_name, expected_version in manifest[
        "installed_distributions"
    ].items():
        try:
            installed_version = metadata.version(distribution_name)
        except metadata.PackageNotFoundError as error:
            raise RuntimeError(
                f"required distribution is missing: {distribution_name}"
            ) from error
        if installed_version != expected_version:
            raise RuntimeError(
                f"distribution version mismatch: {distribution_name}"
            )


def _absolute_new_file(raw: str, label: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        raise RuntimeError(f"{label} must be an absolute path")
    resolved = path.resolve()
    if resolved.exists():
        raise RuntimeError(f"{label} already exists; silent retry is prohibited")
    if not resolved.parent.is_dir():
        raise RuntimeError(f"{label} parent directory is missing")
    return resolved


def _git_identity(project_root: Path) -> tuple[str, str, bool]:
    commit = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD^{tree}"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        [
            "git",
            "-C",
            str(project_root),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    return commit, tree, bool(status.strip())


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise RuntimeError("stale atomic-write temporary file exists")
    temporary.write_bytes(_canonical_json_bytes(value) + b"\n")
    temporary.replace(path)


def _zero_audit() -> dict[str, Any]:
    return {
        "source_rows": 0,
        "source_participants": 0,
        "eligible_participants": 0,
        "eligible_postbaseline_records": 0,
        "exclusion_counts": {},
        "postbaseline_outcome_used_for_prediction": False,
        "calibration_performed": False,
        "model_selection_performed": False,
        "threshold_changed": False,
        "raw_identifier_serialized": False,
        "prediction_interval_status": UQ_STATUS,
    }


def run(request_path: Path) -> dict[str, Any]:
    if not request_path.is_absolute():
        raise RuntimeError("request path must be absolute")
    request_path = request_path.resolve()
    request = _read_json(request_path)
    gate_root = request_path.parent
    workspace_root = gate_root.parent
    request_schema = _request_schema_path(gate_root, request)
    result_schema = gate_root / "v11_mq1_qualification_result.schema.json"
    consumption_schema = (
        gate_root / "v11_mq1_consumption_record.schema.json"
    )
    contract_path = (
        workspace_root
        / "04_机器协议与登记表"
        / "v11_mq1_model_qualification_contract.yaml"
    )
    authorization_path = gate_root / "v11_r3_authorization_record.json"
    for path in (
        request_schema,
        result_schema,
        consumption_schema,
        contract_path,
        authorization_path,
    ):
        if not path.is_file():
            raise RuntimeError(f"required frozen artifact is missing: {path.name}")

    _validate(request, request_schema)
    if request["execution_id"] != EXECUTION_ID:
        raise RuntimeError("execution identity mismatch")
    if request["contract_id"] != CONTRACT_ID:
        raise RuntimeError("contract identity mismatch")
    if request["protocol_version"] != PROTOCOL_VERSION:
        raise RuntimeError("protocol version mismatch")
    if _sha256(contract_path) != request["contract_sha256"]:
        raise RuntimeError("contract SHA-256 mismatch")
    if (
        _sha256(authorization_path)
        != request["authorization_record_sha256"]
    ):
        raise RuntimeError("authorization record SHA-256 mismatch")

    project_root = Path(__file__).resolve().parents[1]
    qualification_lock = project_root / "requirements-r3-qualification.lock"
    if not qualification_lock.is_file():
        raise RuntimeError("R3 qualification dependency lock is missing")
    qualification_lock_sha256 = _sha256(qualification_lock)
    if (
        qualification_lock_sha256
        != request["qualification_lock_sha256"]
    ):
        raise RuntimeError("R3 qualification dependency lock mismatch")
    commit, tree, dirty = _git_identity(project_root)
    if dirty:
        raise RuntimeError("implementation worktree must be clean")
    if (
        commit != request["implementation_commit"]
        or tree != request["implementation_tree"]
    ):
        raise RuntimeError("implementation commit/tree mismatch")

    _validate_corrective_environment(
        request,
        gate_root,
        project_root,
        commit,
        tree,
        qualification_lock_sha256,
    )

    # The scientific package is imported only after the request, code,
    # dependency lock, and corrective interpreter have been validated. This
    # keeps launcher failures before any participant archive access.
    from weight_application.model_qualification import (
        CONTRACT_ID as imported_contract_id,
        QualificationInputError,
        evaluate_model_qualification,
        load_pride_archive,
    )

    if imported_contract_id != CONTRACT_ID:
        raise RuntimeError("installed qualification contract identity mismatch")

    archive_path = _absolute_file(request["archive_path"], "A1 archive")
    if _sha256(archive_path) != request["archive_sha256"]:
        raise RuntimeError("A1 archive SHA-256 mismatch")
    result_path = _absolute_new_file(request["result_path"], "result path")
    consumption_path = _absolute_new_file(
        request["consumption_path"],
        "consumption path",
    )
    if result_path == consumption_path:
        raise RuntimeError("result and consumption paths must differ")
    if result_path.is_relative_to(project_root) or consumption_path.is_relative_to(
        project_root
    ):
        raise RuntimeError("qualification records must be outside code root")

    # Dependency availability is checked before the one-time identity is
    # consumed. No participant row is read by this import.
    import pandas  # noqa: F401

    request_sha256 = _sha256(request_path)
    consumption = {
        "record_id": CONSUMPTION_ID,
        "execution_id": EXECUTION_ID,
        "contract_id": CONTRACT_ID,
        "request_sha256": request_sha256,
        "archive_sha256": request["archive_sha256"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "one_time_consumed": True,
        "retry_allowed": False,
        "result_record_path": str(result_path),
    }
    _validate(consumption, consumption_schema)
    _atomic_write(consumption_path, consumption)

    canonical_input_sha256: str | None = None
    audit = _zero_audit()
    try:
        built = load_pride_archive(archive_path)
        canonical_input_sha256 = built.canonical_input_sha256
        audit = dict(built.audit)
        evaluated = evaluate_model_qualification(built.records)
    except QualificationInputError:
        evaluated = {
            "decision": "QUALIFICATION_INPUT_INVALID",
            "pass": False,
            "case_name": NONPASS_CASE_NAME,
            "reason": "input violated the frozen V11-MQ1 contract",
            "eligible_participants": 0,
            "eligible_postbaseline_records": 0,
            "metrics": None,
            "checks": None,
            "prediction_interval_status": UQ_STATUS,
        }
    except Exception:
        evaluated = {
            "decision": "QUALIFICATION_EXECUTION_FAILED",
            "pass": False,
            "case_name": NONPASS_CASE_NAME,
            "reason": (
                "unexpected controlled-runtime failure; the execution identity "
                "is consumed and audit is required"
            ),
            "eligible_participants": 0,
            "eligible_postbaseline_records": 0,
            "metrics": None,
            "checks": None,
            "prediction_interval_status": UQ_STATUS,
        }

    result = {
        "record_id": RESULT_ID,
        "execution_id": EXECUTION_ID,
        "contract_id": CONTRACT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "implementation": {
            "commit": commit,
            "tree": tree,
            "git_dirty": False,
            "qualification_lock_sha256": qualification_lock_sha256,
        },
        "input": {
            "archive_sha256": request["archive_sha256"],
            "canonical_qualification_input_sha256": canonical_input_sha256,
            "source_table": SOURCE_TABLE,
        },
        "decision": evaluated["decision"],
        "pass": evaluated["pass"],
        "case_name": evaluated["case_name"],
        "eligible_participants": evaluated["eligible_participants"],
        "eligible_postbaseline_records": evaluated[
            "eligible_postbaseline_records"
        ],
        "metrics": evaluated["metrics"],
        "checks": evaluated["checks"],
        "reason": evaluated.get("reason"),
        "prediction_interval_status": evaluated[
            "prediction_interval_status"
        ],
        "audit": audit,
        "result_knowledge": {
            "effect_estimation_performed": False,
            "algorithm_or_comparator_selected": False,
            "participant_values_reported": False,
        },
    }
    _validate(result, result_schema)
    _atomic_write(result_path, result)
    return {
        "execution_id": EXECUTION_ID,
        "one_time_consumed": True,
        "decision": result["decision"],
        "case_name": result["case_name"],
        "aggregate_result_written": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        summary = run(Path(args.request))
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 2
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
