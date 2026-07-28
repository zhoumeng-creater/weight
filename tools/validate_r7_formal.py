"""Result-blind readiness validation for the R7-frozen R8 command."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from formal_execution.runtime import file_sha256
from formal_execution.schedule import (
    build_formal_schedule,
    e2_full_reuse_commitment,
    schedule_commitment,
)


DEFAULT_CONTRACT = (
    PROJECT_ROOT / "config" / "r7" / "r7_formal_execution_contract.json"
)
DEFAULT_SCHEMA = (
    PROJECT_ROOT
    / "config"
    / "r7"
    / "r7_formal_execution_contract.schema.json"
)


class ReadinessError(RuntimeError):
    """R8 must remain closed because a readiness invariant failed."""


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ReadinessError(f"{path.name} must contain a JSON object")
    return value


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *args],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()


def _validate_request_pending(
    request: Mapping[str, Any],
    contract: Mapping[str, Any],
    contract_path: Path,
) -> None:
    if request.get("request_id") != (
        "WGT-V11-R8-EXECUTION-REQUEST-20260725-01"
    ):
        raise ReadinessError("pending R8 request identity differs")
    if {
        request.get("scope"),
        request.get("companion_scope"),
    } != {"benchmark_effect", "weight_effect"}:
        raise ReadinessError("pending R8 request scopes differ")
    if request.get("frozen_exact_command") != (
        contract["launch"]["exact_command"]
    ):
        raise ReadinessError("pending request exact command differs")
    if (
        request.get("author_exact_command_confirmed") is not False
        or request.get("author_confirmation_text") != ""
    ):
        raise ReadinessError(
            "R7 readiness requires the R8 command to remain unconfirmed"
        )
    if request.get("formal_effect_execution_requested") is not True:
        raise ReadinessError("pending request lacks formal execution intent")
    for field in (
        "participant_data_requested",
        "hidden_generation_requested",
        "results_analysis_requested",
        "results_writing_requested",
        "remote_git_mutation_requested",
        "release_or_distribution_requested",
    ):
        if request.get(field) is not False:
            raise ReadinessError(f"pending request escalates {field}")
    bindings = request.get("contracts")
    if not isinstance(bindings, Mapping):
        raise ReadinessError("pending request contracts are missing")
    expected = {
        "protocol_id": "WGT-JOURNAL-2026-01",
        "r5_contract_id": (
            "WGT-V11-R5-ENDPOINT-STATISTICS-SAMPLE-SEED-RESOURCE-01"
        ),
        "r5_contract_sha256": (
            "4e2dd0a0f4a97b57d71dd13eb60aa8a3c3eb34f0708aae609d50a31d155f6554"
        ),
        "r5a_contract_id": "WGT-V11-R5A-E3-INPUT-CONTRACT-01",
        "r5a_contract_sha256": (
            "a7275dc1624fc2167c0ed5a599f9b5cb3297151037c47c5b85fb27d38e857424"
        ),
        "r7_contract_id": "WGT-V11-R7-FORMAL-EXECUTION-CONTRACT-01",
        "r7_contract_sha256": file_sha256(contract_path),
        "formal_schedule_id": "WGT-V11-R7-FORMAL-SCHEDULE-01",
        "formal_schedule_sha256": (
            "40ea633532a3ba2c461ae47925a91ccae305bafac397e6246ced1951fa6e8969"
        ),
        "source_git_commit": _git("rev-parse", "HEAD"),
        "source_git_tree": _git("rev-parse", "HEAD^{tree}"),
    }
    if dict(bindings) != expected:
        raise ReadinessError("pending request contract/source bindings differ")


def _active_formal_processes() -> list[str]:
    if sys.platform != "win32":
        return []
    script = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -like '*run_v11_r8_formal.py*' } | "
        "Select-Object -ExpandProperty CommandLine"
    )
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        check=False,
        text=True,
    )
    return [
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip() and "Get-CimInstance Win32_Process" not in line
    ]


def validate(
    contract_path: Path,
    request_path: Path,
    request_schema_path: Path,
) -> dict[str, Any]:
    contract = _json(contract_path)
    contract_schema = _json(DEFAULT_SCHEMA)
    Draft202012Validator(contract_schema).validate(contract)
    request = _json(request_path)
    request_schema = _json(request_schema_path)
    Draft202012Validator(request_schema).validate(request)

    if _git("status", "--porcelain", "--untracked-files=all"):
        raise ReadinessError("R7 readiness requires a clean code worktree")
    _validate_request_pending(request, contract, contract_path)

    r5_path = PROJECT_ROOT / contract["upstream"]["r5"]["path"]
    r5a_path = PROJECT_ROOT / contract["upstream"]["r5a"]["path"]
    if file_sha256(r5_path) != contract["upstream"]["r5"]["sha256"]:
        raise ReadinessError("R5 contract hash drifted")
    if file_sha256(r5a_path) != contract["upstream"]["r5a"]["sha256"]:
        raise ReadinessError("R5a contract hash drifted")
    schedule = build_formal_schedule(_json(r5_path))
    if schedule_commitment(schedule) != contract["schedule"]["sha256"]:
        raise ReadinessError("formal schedule commitment drifted")
    if e2_full_reuse_commitment(schedule) != (
        contract["schedule"]["e2_full_reuse_sha256"]
    ):
        raise ReadinessError("E2 FULL reuse commitment drifted")
    observed_totals = {
        "method_sequences": len(schedule),
        "CFE": sum(row.total_cfe for row in schedule),
        "atomic_model_steps": sum(
            row.total_atomic_steps for row in schedule
        ),
    }
    if observed_totals != contract["schedule"]["totals"]:
        raise ReadinessError("formal schedule totals drifted")

    dependency_versions = {
        name: metadata.version(name)
        for name in ("numpy", "scipy", "jmetalpy", "jsonschema")
    }
    if dependency_versions["jmetalpy"] != "1.7.0":
        raise ReadinessError("jmetalpy differs from frozen 1.7.0")

    output_root = Path(contract["launch"]["output_root"]).resolve()
    scratch_root = Path(
        contract["resources"]["scratch"]["required_root"]
    ).resolve()
    if output_root.exists():
        raise ReadinessError("frozen R8 output root already exists")
    free_bytes = shutil.disk_usage(scratch_root).free
    if free_bytes < contract["resources"]["scratch"][
        "minimum_free_bytes_at_start"
    ]:
        raise ReadinessError("scratch free bytes are below the R7 start gate")
    active = _active_formal_processes()
    if active:
        raise ReadinessError("an R8 formal runner is already active")

    runner = PROJECT_ROOT / "tools" / "run_v11_r8_formal.py"
    blocked = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--contract",
            str(contract_path),
            "--request",
            str(request_path),
            "--output-root",
            str(output_root),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    if blocked.returncode == 0 or output_root.exists():
        raise ReadinessError(
            "pending R8 request did not fail closed before output creation"
        )
    if "verbatim author confirmation" not in blocked.stderr:
        raise ReadinessError("pending request failed for an unexpected reason")

    return {
        "record_id": "WGT-V11-R7-READINESS-VALIDATION-20260725-01",
        "status": "PASS_R7_READY_R8_PENDING_VERBATIM_CONFIRMATION",
        "contract_sha256": file_sha256(contract_path),
        "source_git_commit": _git("rev-parse", "HEAD"),
        "source_git_tree": _git("rev-parse", "HEAD^{tree}"),
        "schedule": {
            **observed_totals,
            "sha256": schedule_commitment(schedule),
            "e2_full_reuse_rows": 310,
            "e2_full_reuse_sha256": e2_full_reuse_commitment(schedule),
        },
        "resources": {
            "scratch_root": str(scratch_root),
            "free_bytes": free_bytes,
            "minimum_free_bytes_at_start": contract["resources"]["scratch"][
                "minimum_free_bytes_at_start"
            ],
            "max_workers": contract["resources"]["parallelism"][
                "max_workers"
            ],
            "max_total_cpu_seconds": contract["resources"][
                "max_total_cpu_seconds"
            ],
            "max_output_bytes": contract["resources"]["output"][
                "max_total_bytes"
            ],
        },
        "dependencies": dependency_versions,
        "active_formal_processes": 0,
        "pending_request_fail_closed": True,
        "formal_effect_execution_started": False,
        "results_analysis_performed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate R7/R8 readiness")
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--request", required=True)
    parser.add_argument("--request-schema", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = validate(
            Path(args.contract).resolve(),
            Path(args.request).resolve(),
            Path(args.request_schema).resolve(),
        )
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
