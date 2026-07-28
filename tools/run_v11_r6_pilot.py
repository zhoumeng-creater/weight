"""Run the author-authorized isolated R6 result-blind engineering pilot."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from collections import Counter
import ctypes
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark_adapters.public_cmop import StaticCMOPPublicAdapter
from dt_ramde_v11.contracts import (
    AlgorithmConfig,
    ConfigurationError,
    ExecutionScope,
    R6ExecutionRequest,
)
from dt_ramde_v11.engine import DTRAMDE, SequenceRunResult
from evaluation.contracts import EvaluationResult
from weight_application.illustrative_adapter import (
    IllustrativeHallEngineeringAdapter,
    R6_E3_SCENARIOS,
)

from validate_r6_pilot import (
    DEFAULT_CONTRACT,
    ROOT,
    canonical_bytes,
    file_sha256,
    validate_contract,
)


ARTIFACT_ROLE = "R6_RESULT_BLIND_ENGINEERING_ARTIFACT_NOT_EFFECT_RESULT"


class _R6Selector:
    selector_id = "R6-ENGINEERING-MINIMUM-FIRST-OBJECTIVE"
    selector_version = "1.0.0"

    def identity(self) -> Mapping[str, Any]:
        return {
            "selector_id": self.selector_id,
            "selector_version": self.selector_version,
            "role": "R6_result_blind_engineering_only",
        }

    def select(self, archive: Sequence[EvaluationResult]) -> str | None:
        if not archive:
            return None
        return min(
            archive,
            key=lambda result: (result.objectives[0], result.candidate_id),
        ).candidate_id


STATIC_SPEC = {
    "fixture_id": "static_bridge_e0",
    "target_suite_id": "DAS-CMOP-PLATEMO-4.15",
    "target_problem_id": "DASCMOP1",
    "decision_dimension": 30,
    "formal_public_instance": False,
    "effect_evidence": False,
}


def _static_evaluator(
    vector: Sequence[float],
    event_id: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if event_id != 0:
        raise FloatingPointError("R6 static correctness fixture is TS1")
    values = np.asarray(vector, dtype=float)
    return (
        (
            float(np.sum(values**2)),
            float(np.sum((values - 0.25) ** 2)),
        ),
        (float(np.mean(values) - 0.75),),
    )


def _peak_rss_bytes() -> int:
    if os.name == "nt":
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = ctypes.c_void_p
        process = get_current_process()
        get_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory_info.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            ctypes.c_ulong,
        )
        get_memory_info.restype = ctypes.c_int
        ok = get_memory_info(
            process,
            ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.PeakWorkingSetSize)
    import resource

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _request() -> R6ExecutionRequest:
    request = R6ExecutionRequest(scope=ExecutionScope.ENGINEERING_PILOT)
    request.validate()
    return request


def _static_problem(
    contract: Mapping[str, Any],
) -> tuple[StaticCMOPPublicAdapter, AlgorithmConfig]:
    request = _request()
    selector = _R6Selector()
    fixture_hash = sha256(canonical_bytes(STATIC_SPEC)).hexdigest()
    problem = StaticCMOPPublicAdapter(
        suite_id="DAS-CMOP-PLATEMO-4.15",
        problem_id="DASCMOP1",
        evaluator_version="STATIC-CMOP-EVAL-1.0.0",
        fixture_evaluator_sha256=fixture_hash,
        lower=(0.0,) * 30,
        upper=(1.0,) * 30,
        objective_names=("f1", "f2"),
        constraint_names=("g1",),
        evaluator=_static_evaluator,
    )
    workload = contract["pilot_schedule"]["static_public_correctness"]
    config = AlgorithmConfig(
        variant="NO_CROSS_EVENT_MEMORY",
        population_size=4,
        cfe_per_event=workload["cfe_per_event"],
        algorithm_seed=contract["pilot_schedule"]["algorithm_seed"],
        max_events=workload["events"],
        timing_mode="TS1_single_event",
        method_label="F22_MG_STATIC",
        adapter_id=problem.adapter_id,
        adapter_version=problem.adapter_version,
        selector_id=selector.selector_id,
        selector_version=selector.selector_version,
        atomic_steps_per_evaluation=workload["atomic_steps_per_cfe"],
        event_time_limit_seconds=60.0,
        configuration_evidence_id="R6_STATIC_ENGINEERING_PILOT",
        execution_request=request,
    )
    return problem, config


def _illustrative_problem(
    contract: Mapping[str, Any],
    scenario: str,
) -> tuple[IllustrativeHallEngineeringAdapter, AlgorithmConfig]:
    if scenario not in R6_E3_SCENARIOS:
        raise ConfigurationError("worker requested an unknown R6 scenario")
    request = _request()
    selector = _R6Selector()
    schedule = contract["pilot_schedule"]
    workload = schedule["illustrative_e3"]
    problem = IllustrativeHallEngineeringAdapter(
        scenario=scenario,
        development_seed=schedule["development_fixture_seed"],
    )
    config = AlgorithmConfig(
        variant="FULL",
        population_size=4,
        cfe_per_event=workload["cfe_per_event"],
        algorithm_seed=schedule["algorithm_seed"],
        max_events=workload["events"],
        timing_mode="TS2_fixed_periodic_replanning",
        method_label="DT-RAMDE_TS2_FULL",
        adapter_id=problem.adapter_id,
        adapter_version=problem.adapter_version,
        selector_id=selector.selector_id,
        selector_version=selector.selector_version,
        atomic_steps_per_evaluation=workload["atomic_steps_per_cfe"],
        event_time_limit_seconds=60.0,
        configuration_evidence_id="R6_ILLUSTRATIVE_ENGINEERING_PILOT",
        execution_request=request,
    )
    return problem, config


def _redacted_record(
    result: SequenceRunResult,
    *,
    fixture_id: str,
    scenario: str | None,
    repetition: int,
    worker_id: str,
) -> dict[str, Any]:
    semantic_payload = {
        "artifact_role": ARTIFACT_ROLE,
        "fixture_id": fixture_id,
        "scenario_id": scenario,
        "run_result": result.to_dict(),
    }
    ledgers = [event.ledger for event in result.events]
    return {
        "worker_id": worker_id,
        "fixture_id": fixture_id,
        "scenario_id": scenario,
        "repetition": repetition,
        "status": "PASS",
        "semantic_sha256": sha256(
            canonical_bytes(semantic_payload)
        ).hexdigest(),
        "event_count": len(result.events),
        "total_cfe": sum(item["cfe"] for item in ledgers),
        "total_atomic_steps": sum(
            item["atomic_model_steps"] for item in ledgers
        ),
        "execution_transition_count": sum(
            item["execution_transition_count"] for item in ledgers
        ),
        "terminal_code_counts": dict(
            sorted(
                Counter(
                    event.terminal.code.value for event in result.events
                ).items()
            )
        ),
        "effect_estimation_performed": False,
        "participant_data_accessed": False,
        "hidden_instance_accessed_or_generated": False,
        "missingness_branch_reached": (
            scenario == "MISSINGNESS_EVERY_FOURTH_POSTBASELINE_WEEK"
            and len(result.events) >= 5
        ),
    }


def _run_worker(args: argparse.Namespace) -> int:
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    contract = validate_contract(Path(args.contract).resolve())
    if args.worker_kind == "static":
        fixture_id = "static_bridge_e0"
        scenario = None
        problem, config = _static_problem(contract)
    else:
        fixture_id = "illustrative_nonformal_development_fixture"
        scenario = args.scenario
        problem, config = _illustrative_problem(contract, scenario)
    selector = _R6Selector()
    result = DTRAMDE(config).run_sequence(problem, selector=selector)
    record = _redacted_record(
        result,
        fixture_id=fixture_id,
        scenario=scenario,
        repetition=args.repetition,
        worker_id=args.worker_id,
    )
    payload = {
        "record": record,
        "runtime": {
            "worker_id": args.worker_id,
            "attempt": 1,
            "return_code": 0,
            "timed_out": False,
            "wall_seconds": time.perf_counter() - started_wall,
            "cpu_seconds": time.process_time() - started_cpu,
            "peak_rss_bytes": _peak_rss_bytes(),
        },
    }
    print(canonical_bytes(payload).decode("utf-8"))
    return 0


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()


def _source_identity(*, test_mode: bool) -> dict[str, Any]:
    status = _git("status", "--porcelain", "--untracked-files=all")
    dirty = bool(status)
    if dirty and not test_mode:
        raise ConfigurationError(
            "official R6 pilot requires a clean implementation worktree"
        )
    paths = (
        sorted((ROOT / "src").rglob("*.py"))
        + [
            ROOT / "tools" / "run_v11_r6_pilot.py",
            ROOT / "tools" / "validate_r6_pilot.py",
            ROOT / "config" / "r6" / "r6_pilot_contract.json",
            ROOT / "config" / "r6" / "r6_pilot_contract.schema.json",
            ROOT / "requirements-r2.lock",
        ]
    )
    files = {
        path.relative_to(ROOT).as_posix(): file_sha256(path)
        for path in paths
    }
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_tree": _git("rev-parse", "HEAD^{tree}"),
        "git_dirty": dirty,
        "source_files": files,
        "source_bundle_sha256": sha256(canonical_bytes(files)).hexdigest(),
    }


def _validate_output_root(
    raw: str,
    contract: Mapping[str, Any],
    *,
    test_mode: bool,
) -> Path:
    requested = Path(raw)
    if not requested.is_absolute():
        raise ConfigurationError("R6 output root must be absolute")
    resolved = requested.resolve()
    if resolved == ROOT or resolved.is_relative_to(ROOT):
        raise ConfigurationError("R6 output root must be outside the repository")
    if resolved.exists():
        raise ConfigurationError("R6 output root must not already exist")
    if not test_mode:
        required = Path(
            contract["output_contract"]["required_root"]
        ).resolve()
        if resolved != required:
            raise ConfigurationError(
                "official R6 output root differs from the frozen contract"
            )
    return resolved


def _schedule(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    repetitions = contract["pilot_schedule"]["repetitions_per_fixture"]
    schedule: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        schedule.append(
            {
                "worker_kind": "static",
                "scenario": None,
                "repetition": repetition,
                "worker_id": f"static_bridge_e0:r{repetition}",
            }
        )
    for scenario in contract["pilot_schedule"]["illustrative_e3"]["scenarios"]:
        for repetition in range(repetitions):
            schedule.append(
                {
                    "worker_kind": "illustrative",
                    "scenario": scenario,
                    "repetition": repetition,
                    "worker_id": f"{scenario}:r{repetition}",
                }
            )
    return schedule


def _worker_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    src = str(ROOT / "src")
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        src if not current else os.pathsep.join((src, current))
    )
    return environment


def _write_canonical(path: Path, value: Any) -> None:
    path.write_bytes(canonical_bytes(value) + b"\n")


def _write_records(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            mtime=0,
        ) as compressed:
            for record in records:
                compressed.write(canonical_bytes(record) + b"\n")


def _run_supervisor(args: argparse.Namespace) -> dict[str, Any]:
    contract_path = Path(args.contract).resolve()
    contract = validate_contract(contract_path)
    output_root = _validate_output_root(
        args.output_root,
        contract,
        test_mode=args.test_mode,
    )
    source = _source_identity(test_mode=args.test_mode)
    output_root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    controls = contract["resource_controls"]
    records: list[dict[str, Any]] = []
    runtimes: list[dict[str, Any]] = []
    try:
        for item in _schedule(contract):
            remaining = controls["global_wall_timeout_seconds"] - (
                time.perf_counter() - started
            )
            if remaining <= 0.0:
                raise TimeoutError("R6 global hard timeout reached")
            timeout = min(
                float(controls["worker_hard_timeout_seconds"]),
                remaining,
            )
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--contract",
                str(contract_path),
                "--worker",
                "--worker-kind",
                item["worker_kind"],
                "--worker-id",
                item["worker_id"],
                "--repetition",
                str(item["repetition"]),
            ]
            if item["scenario"] is not None:
                command.extend(["--scenario", item["scenario"]])
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=_worker_environment(),
                capture_output=True,
                check=False,
                text=True,
                timeout=timeout,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"worker failed without retry: {item['worker_id']}: "
                    f"{completed.stderr.strip()}"
                )
            payload = json.loads(completed.stdout)
            if payload["record"]["worker_id"] != item["worker_id"]:
                raise RuntimeError("worker identity differs from the schedule")
            if payload["runtime"]["attempt"] != 1:
                raise RuntimeError("worker attempt count differs from no-retry")
            if payload["runtime"]["peak_rss_bytes"] > controls[
                "max_worker_peak_rss_bytes"
            ]:
                raise RuntimeError("worker exceeded the R6 RSS ceiling")
            records.append(payload["record"])
            runtimes.append(payload["runtime"])

        paired: dict[tuple[str, str | None], list[str]] = {}
        for record in records:
            key = (record["fixture_id"], record["scenario_id"])
            paired.setdefault(key, []).append(record["semantic_sha256"])
        if any(
            len(values) != 2 or len(set(values)) != 1
            for values in paired.values()
        ):
            raise RuntimeError("paired deterministic replay semantic hash differs")
        if sum(record["total_cfe"] for record in records) != contract["budget"][
            "total_cfe"
        ]:
            raise RuntimeError("observed R6 CFE differs from contract")
        if sum(
            record["total_atomic_steps"] for record in records
        ) != contract["budget"]["total_atomic_model_steps"]:
            raise RuntimeError("observed R6 atomic steps differ from contract")

        records_path = output_root / "engineering_records.jsonl.gz"
        runtime_path = output_root / "runtime_report.json"
        deviation_path = output_root / "deviation_record.json"
        _write_records(records_path, records)
        runtime_report = {
            "worker_count": len(runtimes),
            "max_parallel_workers": 1,
            "attempts_per_scheduled_worker": 1,
            "automatic_retries": 0,
            "total_wall_seconds": time.perf_counter() - started,
            "total_cpu_seconds": sum(item["cpu_seconds"] for item in runtimes),
            "peak_worker_rss_bytes": max(
                item["peak_rss_bytes"] for item in runtimes
            ),
            "workers": runtimes,
        }
        _write_canonical(runtime_path, runtime_report)
        deviation_record = {
            "deviation_count": 0,
            "deviations": [],
            "effect_estimation_performed": False,
        }
        _write_canonical(deviation_path, deviation_record)
        artifacts = {
            name: {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for name, path in {
                "engineering_records": records_path,
                "runtime_report": runtime_path,
                "deviation_record": deviation_path,
            }.items()
        }
        manifest_binding = {
            "artifact_role": ARTIFACT_ROLE,
            "status": "PASS",
            "protocol": {
                "stage": "R6",
                "execution_authority": contract["execution_authority"],
            },
            "contract": {
                "id": contract["contract_id"],
                "path": contract_path.relative_to(ROOT).as_posix(),
                "sha256": file_sha256(contract_path),
            },
            "upstream": contract["upstream"],
            "code": source,
            "schedule": {
                "worker_processes": len(records),
                "fixture_scenario_pairs": len(paired),
                "repetitions_per_pair": 2,
                "total_cfe": sum(record["total_cfe"] for record in records),
                "total_atomic_model_steps": sum(
                    record["total_atomic_steps"] for record in records
                ),
            },
            "resources": {
                "worker_hard_timeout_seconds": controls[
                    "worker_hard_timeout_seconds"
                ],
                "max_worker_peak_rss_bytes": controls[
                    "max_worker_peak_rss_bytes"
                ],
                "max_output_bytes": controls["max_output_bytes"],
                "automatic_retries": 0,
            },
            "artifacts": artifacts,
            "permissions": contract["permissions"],
            "effect_analysis_performed": False,
            "result_claims_permitted": False,
            "formal_input_gap": contract["formal_input_gap"],
            "next_gate": contract["next_gate"],
            "test_mode": bool(args.test_mode),
        }
        run_hash = sha256(canonical_bytes(manifest_binding)).hexdigest()
        manifest = {
            **manifest_binding,
            "run_id": f"r6-{run_hash[:24]}",
            "run_binding_sha256": run_hash,
        }
        manifest_path = output_root / "run_manifest.json"
        _write_canonical(manifest_path, manifest)
        total_bytes = sum(path.stat().st_size for path in output_root.iterdir())
        if total_bytes > controls["max_output_bytes"]:
            raise RuntimeError("R6 output exceeded its byte ceiling")
        return {
            "artifact_role": ARTIFACT_ROLE,
            "status": "PASS",
            "run_id": manifest["run_id"],
            "worker_processes": len(records),
            "total_cfe": manifest["schedule"]["total_cfe"],
            "total_atomic_model_steps": manifest["schedule"][
                "total_atomic_model_steps"
            ],
            "effect_estimation_performed": False,
            "r7_authorized": False,
        }
    except Exception as error:
        failure = {
            "artifact_role": ARTIFACT_ROLE,
            "status": "FAIL_CLOSED",
            "error_type": type(error).__name__,
            "error": str(error),
            "completed_workers": len(records),
            "automatic_retries": 0,
            "effect_estimation_performed": False,
        }
        _write_canonical(output_root / "failure_record.json", failure)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the isolated result-blind R6 engineering pilot"
    )
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--output-root")
    parser.add_argument("--test-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-kind",
        choices=("static", "illustrative"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-id", help=argparse.SUPPRESS)
    parser.add_argument("--scenario", choices=R6_E3_SCENARIOS, help=argparse.SUPPRESS)
    parser.add_argument("--repetition", type=int, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.worker:
            if (
                args.worker_kind is None
                or args.worker_id is None
                or args.repetition not in {0, 1}
                or (
                    args.worker_kind == "illustrative"
                    and args.scenario is None
                )
                or (
                    args.worker_kind == "static"
                    and args.scenario is not None
                )
            ):
                raise ConfigurationError("incomplete R6 worker invocation")
            return _run_worker(args)
        if args.output_root is None:
            raise ConfigurationError("R6 supervisor requires --output-root")
        summary = _run_supervisor(args)
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 2
    print(canonical_bytes(summary).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
