"""Validate the R6 contract and optional result-blind engineering artifacts."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from weight_application.illustrative_adapter import R6_E3_SCENARIOS


DEFAULT_CONTRACT = ROOT / "config" / "r6" / "r6_pilot_contract.json"
SCHEMA_PATH = ROOT / "config" / "r6" / "r6_pilot_contract.schema.json"
REQUIRED_OUTPUT_FILES = (
    "engineering_records.jsonl.gz",
    "runtime_report.json",
    "deviation_record.json",
    "run_manifest.json",
)
RECORD_KEYS = {
    "worker_id",
    "fixture_id",
    "scenario_id",
    "repetition",
    "status",
    "semantic_sha256",
    "event_count",
    "total_cfe",
    "total_atomic_steps",
    "execution_transition_count",
    "terminal_code_counts",
    "effect_estimation_performed",
    "participant_data_accessed",
    "hidden_instance_accessed_or_generated",
    "missingness_branch_reached",
}
PROHIBITED_PERSISTED_KEY_PARTS = (
    "objective",
    "action",
    "state_after",
    "participant_id",
    "effect_size",
    "p_value",
    "confidence_interval",
    "rank",
)


class R6ValidationError(RuntimeError):
    """The R6 contract or pilot output is incomplete or result-aware."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_false_mapping(mapping: Mapping[str, Any], *, allowed_true: set[str]) -> None:
    for key, value in mapping.items():
        expected = key in allowed_true
        if value is not expected:
            raise R6ValidationError(f"permission differs: {key}")


def validate_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    contract = load_json(path)
    schema = load_json(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(contract)

    upstream = contract["upstream"]
    r5_path = ROOT / upstream["r5_contract_path"]
    if not r5_path.is_file() or file_sha256(r5_path) != upstream["r5_contract_sha256"]:
        raise R6ValidationError("R5 contract hash binding differs")
    r5 = load_json(r5_path)
    if r5["contract_id"] != upstream["r5_contract_id"]:
        raise R6ValidationError("R5 contract identity differs")
    scenarios = contract["pilot_schedule"]["illustrative_e3"]["scenarios"]
    if tuple(scenarios) != R6_E3_SCENARIOS:
        raise R6ValidationError("R6 scenario order differs from the adapter")
    if scenarios != r5["experiment_design"]["E3"]["scenarios"]:
        raise R6ValidationError("R6 scenario set differs from R5")

    schedule = contract["pilot_schedule"]
    repetitions = schedule["repetitions_per_fixture"]
    static = schedule["static_public_correctness"]
    e3 = schedule["illustrative_e3"]
    expected_workers = repetitions * (1 + len(scenarios))
    expected_cfe = repetitions * (
        static["events"] * static["cfe_per_event"]
        + len(scenarios) * e3["events"] * e3["cfe_per_event"]
    )
    expected_steps = repetitions * (
        static["events"]
        * static["cfe_per_event"]
        * static["atomic_steps_per_cfe"]
        + len(scenarios)
        * e3["events"]
        * e3["cfe_per_event"]
        * e3["atomic_steps_per_cfe"]
    )
    budget = contract["budget"]
    if (
        budget["scheduled_worker_processes"] != expected_workers
        or budget["total_cfe"] != expected_cfe
        or budget["total_atomic_model_steps"] != expected_steps
    ):
        raise R6ValidationError("R6 pilot budget arithmetic differs")
    controls = contract["resource_controls"]
    if (
        controls["max_parallel_workers"] != 1
        or controls["auto_retry_allowed"] is not False
        or controls["attempts_per_scheduled_worker"] != 1
        or controls["gpu_allowed"] is not False
    ):
        raise R6ValidationError("R6 resource/no-retry controls differ")
    _assert_false_mapping(
        contract["permissions"],
        allowed_true={"r6_engineering_pilot_allowed"},
    )
    gap = contract["formal_input_gap"]
    if (
        gap["r5_seed_to_subject_parameter_generator_frozen"] is not False
        or gap["r5_formal_target_rule_frozen"] is not False
        or gap["r6_may_invent_formal_rules"] is not False
        or contract["next_gate"]["authorized"] is not False
    ):
        raise R6ValidationError("R6 improperly opened formal R7 inputs")
    expected_command = [
        "python",
        "tools/run_v11_r6_pilot.py",
        "--contract",
        "config/r6/r6_pilot_contract.json",
        "--output-root",
        contract["output_contract"]["required_root"],
    ]
    if contract["official_command"] != expected_command:
        raise R6ValidationError("official R6 command differs")
    return contract


def _find_prohibited_key(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(part in lowered for part in PROHIBITED_PERSISTED_KEY_PARTS):
                return str(key)
            nested = _find_prohibited_key(item)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for item in value:
            nested = _find_prohibited_key(item)
            if nested is not None:
                return nested
    return None


def _canonical_json_file(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if raw != canonical_bytes(value) + b"\n":
        raise R6ValidationError(f"{path.name} is not canonical JSON")
    return value


def _read_records(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if len(raw) < 10 or raw[4:8] != b"\x00\x00\x00\x00":
        raise R6ValidationError("engineering gzip mtime is not zero")
    with gzip.open(path, "rt", encoding="utf-8", newline="\n") as handle:
        lines = handle.readlines()
    records = [json.loads(line) for line in lines]
    expected = b"".join(canonical_bytes(record) + b"\n" for record in records)
    with gzip.open(path, "rb") as handle:
        if handle.read() != expected:
            raise R6ValidationError("engineering records are not canonical JSONL")
    return records


def _git_value(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()


def _validate_historical_code_identity(code: Mapping[str, Any]) -> None:
    if code["git_dirty"] is not False:
        raise R6ValidationError(
            "official R6 output was not produced from clean code"
        )
    commit = str(code["git_commit"])
    tree = str(code["git_tree"])
    try:
        recorded_tree = _git_value("show", "-s", "--format=%T", commit)
    except subprocess.CalledProcessError as error:
        raise R6ValidationError("R6 manifest commit is unavailable") from error
    if tree != recorded_tree:
        raise R6ValidationError("R6 manifest commit/tree binding differs")
    ancestor = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    if ancestor.returncode != 0:
        raise R6ValidationError(
            "R6 manifest commit is not an ancestor of current HEAD"
        )


def validate_output(
    output_root: Path,
    *,
    contract_path: Path = DEFAULT_CONTRACT,
    allow_test_mode: bool = False,
) -> dict[str, Any]:
    contract = validate_contract(contract_path)
    output_root = output_root.resolve()
    if not output_root.is_dir():
        raise R6ValidationError("R6 output root is missing")
    if sorted(path.name for path in output_root.iterdir()) != sorted(
        REQUIRED_OUTPUT_FILES
    ):
        raise R6ValidationError("R6 output file set differs")
    if output_root == ROOT or output_root.is_relative_to(ROOT):
        raise R6ValidationError("R6 output root is inside the repository")

    records_path = output_root / "engineering_records.jsonl.gz"
    runtime_path = output_root / "runtime_report.json"
    deviation_path = output_root / "deviation_record.json"
    manifest_path = output_root / "run_manifest.json"
    records = _read_records(records_path)
    runtime = _canonical_json_file(runtime_path)
    deviation = _canonical_json_file(deviation_path)
    manifest = _canonical_json_file(manifest_path)

    if len(records) != contract["budget"]["scheduled_worker_processes"]:
        raise R6ValidationError("scheduled worker record count differs")
    if len(runtime["workers"]) != len(records):
        raise R6ValidationError("runtime worker count differs")
    if deviation != {
        "deviation_count": 0,
        "deviations": [],
        "effect_estimation_performed": False,
    }:
        raise R6ValidationError("R6 deviation record is not empty/result-blind")

    paired: dict[tuple[str, str | None], list[str]] = {}
    for record in records:
        if set(record) != RECORD_KEYS:
            raise R6ValidationError("engineering record schema differs")
        prohibited = _find_prohibited_key(record)
        if prohibited is not None:
            raise R6ValidationError(
                f"engineering record persisted prohibited key: {prohibited}"
            )
        if (
            record["status"] != "PASS"
            or record["effect_estimation_performed"] is not False
            or record["participant_data_accessed"] is not False
            or record["hidden_instance_accessed_or_generated"] is not False
        ):
            raise R6ValidationError("worker record crossed the result-blind boundary")
        digest = record["semantic_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise R6ValidationError("worker semantic hash is invalid")
        key = (record["fixture_id"], record["scenario_id"])
        paired.setdefault(key, []).append(digest)
    if any(len(values) != 2 or len(set(values)) != 1 for values in paired.values()):
        raise R6ValidationError("paired deterministic replay hash differs")
    if len(paired) != 10:
        raise R6ValidationError("fixture/scenario branch count differs")
    missing = next(
        record
        for record in records
        if record["scenario_id"]
        == "MISSINGNESS_EVERY_FOURTH_POSTBASELINE_WEEK"
    )
    if missing["missingness_branch_reached"] is not True:
        raise R6ValidationError("week-4 missingness branch was not reached")
    if sum(record["total_cfe"] for record in records) != contract["budget"]["total_cfe"]:
        raise R6ValidationError("observed CFE total differs")
    if sum(record["total_atomic_steps"] for record in records) != (
        contract["budget"]["total_atomic_model_steps"]
    ):
        raise R6ValidationError("observed atomic-step total differs")

    artifact_paths = {
        "engineering_records": records_path,
        "runtime_report": runtime_path,
        "deviation_record": deviation_path,
    }
    for name, path in artifact_paths.items():
        artifact = manifest["artifacts"][name]
        if artifact != {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }:
            raise R6ValidationError(f"manifest artifact binding differs: {name}")
    if manifest["contract"]["sha256"] != file_sha256(contract_path):
        raise R6ValidationError("manifest contract hash differs")
    if manifest["permissions"] != contract["permissions"]:
        raise R6ValidationError("manifest permissions differ")
    if manifest["status"] != "PASS" or manifest["effect_analysis_performed"] is not False:
        raise R6ValidationError("manifest status/effect boundary differs")
    if manifest["test_mode"]:
        if not allow_test_mode:
            raise R6ValidationError("test-mode output is not an official R6 artifact")
    else:
        _validate_historical_code_identity(manifest["code"])
    total_bytes = sum(path.stat().st_size for path in output_root.iterdir())
    if total_bytes > contract["resource_controls"]["max_output_bytes"]:
        raise R6ValidationError("R6 output exceeded its byte ceiling")
    if runtime["peak_worker_rss_bytes"] > contract["resource_controls"][
        "max_worker_peak_rss_bytes"
    ]:
        raise R6ValidationError("R6 worker RSS exceeded its ceiling")
    return {
        "validator": "WGT-V11-R6-ENGINEERING-PILOT-VALIDATOR-01",
        "status": "PASS",
        "run_id": manifest["run_id"],
        "worker_processes": len(records),
        "scenario_branches": 9,
        "total_cfe": sum(record["total_cfe"] for record in records),
        "total_atomic_model_steps": sum(
            record["total_atomic_steps"] for record in records
        ),
        "paired_replay_hashes_match": True,
        "effect_estimation_performed": False,
        "r7_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--output-root")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(argv)
    try:
        contract_path = Path(args.contract).resolve()
        if args.output_root:
            summary = validate_output(
                Path(args.output_root),
                contract_path=contract_path,
            )
        else:
            contract = validate_contract(contract_path)
            summary = {
                "validator": "WGT-V11-R6-CONTRACT-VALIDATOR-01",
                "status": "PASS",
                "scheduled_worker_processes": contract["budget"][
                    "scheduled_worker_processes"
                ],
                "total_cfe": contract["budget"]["total_cfe"],
                "total_atomic_model_steps": contract["budget"][
                    "total_atomic_model_steps"
                ],
                "effect_estimation_performed": False,
                "r7_authorized": False,
            }
    except Exception as error:
        print(str(error))
        return 2
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            sort_keys=True,
            indent=None if args.compact else 2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
