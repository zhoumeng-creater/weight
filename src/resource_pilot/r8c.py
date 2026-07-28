"""Result-blind R8C resource qualification on public engineering fixtures.

The pilot deliberately uses :class:`R6ExecutionRequest` and never accepts an
R8/R8C formal request.  Optimization results exist only inside each worker
process long enough to exercise the real 100/100 algorithm path.  Workers
return control-plane resource counters only; objectives, constraints,
archives, candidate identities, actions, feedback, and information hashes are
never serialized.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
from tempfile import gettempdir
import time
from typing import Any, Mapping, Sequence

from benchmark_adapters.r4_public import (
    make_r4_cdf_adapter,
    make_r4_lircmop_adapter,
)
from benchmark_adapters.r4_wgt_rr import WGTRRPublicAdapter
from dt_ramde_v11.contracts import (
    AlgorithmConfig,
    ExecutionScope,
    R6ExecutionRequest,
)
from dt_ramde_v11.engine import DTRAMDE
from evaluation.contracts import EvaluationResult
from weight_application.illustrative_adapter import (
    IllustrativeHallEngineeringAdapter,
)


ARTIFACT_ROLE = (
    "R8C_RESULT_BLIND_RESOURCE_QUALIFICATION_CONTROL_PLANE_ONLY"
)
PILOT_ID = "WGT-V11-R8C-RESOURCE-PILOT-20260726-LOCAL-01"
_SYSTEM_TEMP_ROOT = Path(gettempdir()).resolve() / "dt-ramde-v11"
DEFAULT_OUTPUT_ROOT = _SYSTEM_TEMP_ROOT / "r8c-resource-pilot-default"
FORBIDDEN_FORMAL_ROOT = _SYSTEM_TEMP_ROOT / "r8c-formal-forbidden"
POPULATION_SIZE = 100
ARCHIVE_CAPACITY = 100
DEFAULT_WORKERS = (1, 8, 16, 24, 32)
DEFAULT_REPETITIONS_PER_PROFILE = 6
DEFAULT_CFE_PER_EVENT = 500
DEFAULT_DYNAMIC_EVENTS = 2
PROJECT_ROOT = Path(__file__).resolve().parents[2]
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

# The frozen R5 accounting already incorporates E2 FULL reuse from E1.  These
# are workload-control budgets, not observed effects.
E1_E2_FORMAL_CFE = {
    "E1_STATIC": 42_000_000,
    "E1_DYNAMIC": 270_000_000,
    "E1_ROLLING": 96_000_000,
    "E2_DYNAMIC": 315_000_000,
    "E2_ROLLING": 128_000_000,
}

PROHIBITED_PERSISTED_KEYS = frozenset(
    {
        "action",
        "archive",
        "candidate",
        "candidate_id",
        "constraint",
        "constraints",
        "execution_feedback",
        "feedback",
        "feasible",
        "information_hash",
        "objective",
        "objectives",
        "selected",
        "selected_vector",
        "terminal",
        "terminal_candidate_id",
        "total_violation",
        "vector",
    }
)


class ResourcePilotError(RuntimeError):
    """The fail-closed resource pilot contract was violated."""


@dataclass(frozen=True)
class WorkloadProfile:
    profile_id: str
    adapter_kind: str
    variant: str
    events: int
    atomic_steps_per_cfe: int
    timing_mode: str
    method_label: str


@dataclass(frozen=True)
class PilotTask:
    task_id: str
    worker_count: int
    ordinal: int
    repetition: int
    profile: WorkloadProfile
    cfe_per_event: int
    algorithm_seed: int


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _profiles(dynamic_events: int) -> tuple[WorkloadProfile, ...]:
    if dynamic_events < 1:
        raise ResourcePilotError("dynamic event count must be positive")
    return (
        WorkloadProfile(
            profile_id="E1_STATIC",
            adapter_kind="lircmop",
            variant="NO_CROSS_EVENT_MEMORY",
            events=1,
            atomic_steps_per_cfe=1,
            timing_mode="TS1_single_event",
            method_label="F22_MG_STATIC",
        ),
        WorkloadProfile(
            profile_id="E1_DYNAMIC",
            adapter_kind="cdf",
            variant="FULL",
            events=dynamic_events,
            atomic_steps_per_cfe=1,
            timing_mode="TS2_fixed_periodic_replanning",
            method_label="DT-RAMDE_TS2_FULL",
        ),
        WorkloadProfile(
            profile_id="E1_ROLLING",
            adapter_kind="rolling",
            variant="FULL",
            events=dynamic_events,
            atomic_steps_per_cfe=6,
            timing_mode="TS2_fixed_periodic_replanning",
            method_label="DT-RAMDE_TS2_FULL",
        ),
        WorkloadProfile(
            profile_id="E2_DYNAMIC",
            adapter_kind="cdf",
            variant="NO_CROSS_EVENT_MEMORY",
            events=dynamic_events,
            atomic_steps_per_cfe=1,
            timing_mode="TS2_fixed_periodic_replanning",
            method_label="NO_CROSS_EVENT_MEMORY",
        ),
        WorkloadProfile(
            profile_id="E2_ROLLING",
            adapter_kind="rolling",
            variant="NO_EXECUTION_FEEDBACK",
            events=dynamic_events,
            atomic_steps_per_cfe=6,
            timing_mode="TS2_fixed_periodic_replanning",
            method_label="NO_EXECUTION_FEEDBACK",
        ),
        WorkloadProfile(
            profile_id="E3_SUPPORTIVE",
            adapter_kind="illustrative_e3",
            variant="FULL",
            events=dynamic_events,
            atomic_steps_per_cfe=6,
            timing_mode="TS2_fixed_periodic_replanning",
            method_label="DT-RAMDE_TS2_FULL",
        ),
    )


def build_tasks(
    *,
    worker_count: int,
    repetitions_per_profile: int,
    cfe_per_event: int,
    dynamic_events: int,
) -> tuple[PilotTask, ...]:
    """Build deterministic, worker-sweep-specific pilot identities."""

    if worker_count < 1:
        raise ResourcePilotError("worker count must be positive")
    if repetitions_per_profile < 1:
        raise ResourcePilotError(
            "repetitions per profile must be positive"
        )
    if cfe_per_event < POPULATION_SIZE:
        raise ResourcePilotError(
            "pilot CFE per event must cover population 100"
        )
    tasks: list[PilotTask] = []
    for repetition in range(repetitions_per_profile):
        for profile in _profiles(dynamic_events):
            ordinal = len(tasks)
            seed = 8_202_607_260 + repetition * 101 + ordinal
            binding = {
                "pilot_id": PILOT_ID,
                "worker_count": worker_count,
                "ordinal": ordinal,
                "repetition": repetition,
                "profile_id": profile.profile_id,
                "cfe_per_event": cfe_per_event,
                "events": profile.events,
                "algorithm_seed": seed,
                "population_size": POPULATION_SIZE,
                "archive_capacity": ARCHIVE_CAPACITY,
            }
            digest = sha256(canonical_json_bytes(binding)).hexdigest()[:16]
            task_id = (
                f"r8c-resource-pilot-w{worker_count:02d}-"
                f"{ordinal:04d}-{digest}"
            )
            tasks.append(
                PilotTask(
                    task_id=task_id,
                    worker_count=worker_count,
                    ordinal=ordinal,
                    repetition=repetition,
                    profile=profile,
                    cfe_per_event=cfe_per_event,
                    algorithm_seed=seed,
                )
            )
    task_ids = tuple(task.task_id for task in tasks)
    if len(task_ids) != len(set(task_ids)):
        raise ResourcePilotError("pilot task IDs are not unique")
    return tuple(tasks)


class _PilotSelector:
    selector_id = "R8C-RESOURCE-PILOT-MINIMUM-FIRST-OBJECTIVE"
    selector_version = "1.0.0"

    def identity(self) -> Mapping[str, Any]:
        return {
            "selector_id": self.selector_id,
            "selector_version": self.selector_version,
            "role": "result_blind_resource_pilot_internal_only",
        }

    def select(
        self,
        archive: Sequence[EvaluationResult],
    ) -> str | None:
        if not archive:
            return None
        return min(
            archive,
            key=lambda result: (
                result.objectives[0],
                result.candidate_id,
            ),
        ).candidate_id


def _build_problem(task: PilotTask) -> Any:
    kind = task.profile.adapter_kind
    if kind == "lircmop":
        return make_r4_lircmop_adapter(1)
    if kind == "cdf":
        return make_r4_cdf_adapter(
            1,
            profile="CDF-HARSH",
            environment_seed=task.algorithm_seed,
        )
    if kind == "rolling":
        return WGTRRPublicAdapter.from_known_answer()
    if kind == "illustrative_e3":
        return IllustrativeHallEngineeringAdapter(
            scenario="NOMINAL",
            development_seed=task.algorithm_seed,
        )
    raise ResourcePilotError(f"unknown pilot adapter kind {kind!r}")


def _peak_rss_bytes() -> int:
    if os.name == "nt":
        import ctypes

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
        if not get_memory_info(
            process,
            ctypes.byref(counters),
            counters.cb,
        ):
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.PeakWorkingSetSize)
    import resource

    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if platform.system() == "Darwin" else peak * 1024


def _run_task(task: PilotTask) -> dict[str, Any]:
    """Execute one task and return resource counters only."""

    request = R6ExecutionRequest(scope=ExecutionScope.ENGINEERING_PILOT)
    request.validate()
    selector = _PilotSelector()
    problem = _build_problem(task)
    config = AlgorithmConfig(
        variant=task.profile.variant,
        population_size=POPULATION_SIZE,
        cfe_per_event=task.cfe_per_event,
        algorithm_seed=task.algorithm_seed,
        max_events=task.profile.events,
        timing_mode=task.profile.timing_mode,
        method_label=task.profile.method_label,
        adapter_id=problem.adapter_id,
        adapter_version=problem.adapter_version,
        selector_id=selector.selector_id,
        selector_version=selector.selector_version,
        atomic_steps_per_evaluation=task.profile.atomic_steps_per_cfe,
        event_time_limit_seconds=3_600.0,
        configuration_evidence_id=(
            f"R8C_RESOURCE_{task.profile.profile_id}_PILOT"
        ),
        execution_request=request,
    )
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    run = DTRAMDE(config).run_sequence(problem, selector=selector)
    wall_seconds = time.perf_counter() - started_wall
    cpu_seconds = time.process_time() - started_cpu
    ledgers = tuple(event.ledger for event in run.events)
    total_cfe = sum(int(ledger["cfe"]) for ledger in ledgers)
    total_atomic = sum(
        int(ledger["atomic_model_steps"]) for ledger in ledgers
    )
    expected_cfe = task.profile.events * task.cfe_per_event
    expected_atomic = expected_cfe * task.profile.atomic_steps_per_cfe
    if len(run.events) != task.profile.events:
        raise ResourcePilotError("pilot event count differs from request")
    if total_cfe != expected_cfe or total_atomic != expected_atomic:
        raise ResourcePilotError("pilot accounting differs from request")

    # `run` is intentionally neither serialized nor hashed.
    return {
        "artifact_role": ARTIFACT_ROLE,
        "task_id": task.task_id,
        "worker_count": task.worker_count,
        "ordinal": task.ordinal,
        "repetition": task.repetition,
        "profile_id": task.profile.profile_id,
        "status": "PASS",
        "attempt": 1,
        "automatic_retries": 0,
        "event_count": task.profile.events,
        "cfe_per_event": task.cfe_per_event,
        "total_cfe": total_cfe,
        "atomic_steps_per_cfe": task.profile.atomic_steps_per_cfe,
        "total_atomic_model_steps": total_atomic,
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "peak_rss_bytes": _peak_rss_bytes(),
        "process_id": os.getpid(),
        "formal_request_loaded": False,
        "formal_request_consumed": False,
        "effect_fields_persisted": False,
        "effect_analysis_performed": False,
        "participant_data_accessed": False,
    }


def assert_control_plane_only(value: Any, path: str = "$") -> None:
    """Reject any effect-bearing key before a pilot artifact is persisted."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in PROHIBITED_PERSISTED_KEYS:
                raise ResourcePilotError(
                    f"effect-bearing key {key!r} at {path} is prohibited"
                )
            assert_control_plane_only(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_control_plane_only(item, f"{path}[{index}]")


def _failed_record(task: PilotTask, error: BaseException) -> dict[str, Any]:
    return {
        "artifact_role": ARTIFACT_ROLE,
        "task_id": task.task_id,
        "worker_count": task.worker_count,
        "ordinal": task.ordinal,
        "repetition": task.repetition,
        "profile_id": task.profile.profile_id,
        "status": "FAILED",
        "attempt": 1,
        "automatic_retries": 0,
        "error_type": type(error).__name__,
        "formal_request_loaded": False,
        "formal_request_consumed": False,
        "effect_fields_persisted": False,
        "effect_analysis_performed": False,
        "participant_data_accessed": False,
    }


def _run_worker_sweep(
    *,
    worker_count: int,
    tasks: Sequence[PilotTask],
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    started = time.perf_counter()
    by_id = {task.task_id: task for task in tasks}
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        futures = {pool.submit(_run_task, task): task.task_id for task in tasks}
        for future in as_completed(futures):
            task_id = futures[future]
            task = by_id[task_id]
            try:
                record = future.result()
            except BaseException as error:
                record = _failed_record(task, error)
            assert_control_plane_only(record)
            records.append(record)
    wall_seconds = time.perf_counter() - started
    records.sort(key=lambda item: int(item["ordinal"]))
    passed = tuple(record for record in records if record["status"] == "PASS")
    failed = tuple(record for record in records if record["status"] != "PASS")
    total_cfe = sum(int(record["total_cfe"]) for record in passed)
    total_atomic = sum(
        int(record["total_atomic_model_steps"]) for record in passed
    )
    cpu_seconds = sum(float(record["cpu_seconds"]) for record in passed)
    peaks_by_process: dict[int, int] = {}
    for record in passed:
        process_id = int(record["process_id"])
        peaks_by_process[process_id] = max(
            peaks_by_process.get(process_id, 0),
            int(record["peak_rss_bytes"]),
        )
    logical_processors = os.cpu_count()
    aggregate = {
        "artifact_role": ARTIFACT_ROLE,
        "workers_requested": worker_count,
        "task_count": len(records),
        "passed_task_count": len(passed),
        "failed_task_count": len(failed),
        "wall_seconds": wall_seconds,
        "sum_task_cpu_seconds": cpu_seconds,
        "total_cfe": total_cfe,
        "total_atomic_model_steps": total_atomic,
        "cfe_per_wall_second": (
            total_cfe / wall_seconds if wall_seconds > 0.0 else None
        ),
        "cfe_per_cpu_second": (
            total_cfe / cpu_seconds if cpu_seconds > 0.0 else None
        ),
        "cpu_concurrency_equivalent": (
            cpu_seconds / wall_seconds if wall_seconds > 0.0 else None
        ),
        "worker_process_count_observed": len(peaks_by_process),
        "max_process_peak_rss_bytes": (
            max(peaks_by_process.values()) if peaks_by_process else None
        ),
        "sum_process_peak_rss_upper_bound_bytes": sum(
            peaks_by_process.values()
        ),
        "logical_processors_observed": logical_processors,
        "logical_processor_oversubscription": (
            logical_processors is not None
            and worker_count > logical_processors
        ),
        "automatic_retries": 0,
        "formal_request_loaded": False,
        "formal_request_consumed": False,
        "effect_fields_persisted": False,
        "effect_analysis_performed": False,
    }
    assert_control_plane_only(aggregate)
    return aggregate, tuple(records)


def _profile_rates(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    rates: dict[str, float] = {}
    for profile_id in E1_E2_FORMAL_CFE:
        profile_records = tuple(
            record
            for record in records
            if record["status"] == "PASS"
            and record["profile_id"] == profile_id
        )
        if not profile_records:
            continue
        seconds_per_cfe = tuple(
            float(record["wall_seconds"]) / int(record["total_cfe"])
            for record in profile_records
        )
        median_seconds_per_cfe = statistics.median(seconds_per_cfe)
        if median_seconds_per_cfe > 0.0:
            rates[profile_id] = 1.0 / median_seconds_per_cfe
    return rates


def _projection(
    *,
    worker_count: int,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rates = _profile_rates(records)
    missing = sorted(set(E1_E2_FORMAL_CFE) - set(rates))
    if missing:
        return {
            "workers": worker_count,
            "status": "UNAVAILABLE_FAILED_OR_MISSING_PROFILE",
            "missing_profile_ids": missing,
            "formal_go": False,
        }
    serial_seconds_under_observed_contention = sum(
        E1_E2_FORMAL_CFE[profile_id] / rates[profile_id]
        for profile_id in E1_E2_FORMAL_CFE
    )
    projected_seconds = (
        serial_seconds_under_observed_contention / worker_count
    )
    hours = projected_seconds / 3_600.0
    if hours <= 36.0:
        time_gate = "LOCAL_PROJECTION_LE_36_HOURS"
    elif hours <= 48.0:
        time_gate = "LOCAL_PROJECTION_36_TO_48_HOURS"
    else:
        time_gate = "LOCAL_PROJECTION_GT_48_HOURS"
    return {
        "workers": worker_count,
        "status": "LOCAL_ESTIMATE_ONLY",
        "profile_cfe_per_worker_second": rates,
        "formal_e1_e2_total_cfe": sum(E1_E2_FORMAL_CFE.values()),
        "projected_wall_seconds": projected_seconds,
        "projected_wall_hours": hours,
        "time_gate_classification": time_gate,
        "formal_go": False,
        "formal_go_blocker": (
            "TARGET_64_VCPU_EPYC_HOST_HAS_NOT_BEEN_MEASURED"
        ),
    }


def _normalized(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def validate_output_root(output_root: Path) -> None:
    resolved = output_root.resolve()
    if _normalized(resolved) == _normalized(FORBIDDEN_FORMAL_ROOT):
        raise ResourcePilotError("resource pilot cannot use the formal root")
    if not resolved.name.startswith("r8c-resource-pilot-"):
        raise ResourcePilotError(
            "resource pilot root must start with 'r8c-resource-pilot-'"
        )
    if resolved.exists():
        raise ResourcePilotError(
            "resource pilot root already exists; automatic resume/retry is forbidden"
        )


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def code_identity(*, require_clean: bool) -> dict[str, Any]:
    try:
        commit = _git("rev-parse", "HEAD")
        tree = _git("rev-parse", "HEAD^{tree}")
        status = _git("status", "--porcelain", "--untracked-files=all")
    except (OSError, subprocess.CalledProcessError) as error:
        raise ResourcePilotError(
            "resource pilot requires an identifiable Git worktree"
        ) from error
    clean = status == ""
    if require_clean and not clean:
        raise ResourcePilotError(
            "resource pilot requires a clean committed Git identity"
        )
    return {
        "git_commit": commit,
        "git_tree": tree,
        "worktree_clean": clean,
        "runner_source_sha256": sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }


def run_resource_pilot(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    worker_counts: Sequence[int] = DEFAULT_WORKERS,
    repetitions_per_profile: int = DEFAULT_REPETITIONS_PER_PROFILE,
    cfe_per_event: int = DEFAULT_CFE_PER_EVENT,
    dynamic_events: int = DEFAULT_DYNAMIC_EVENTS,
    require_clean_git: bool = True,
) -> dict[str, Any]:
    """Run the isolated sweep once and persist control-plane artifacts."""

    counts = tuple(int(value) for value in worker_counts)
    if not counts or any(value < 1 for value in counts):
        raise ResourcePilotError("worker counts must be positive")
    if len(counts) != len(set(counts)):
        raise ResourcePilotError("worker counts must be unique")
    identity = code_identity(require_clean=require_clean_git)
    validate_output_root(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    root_marker = {
        "artifact_role": ARTIFACT_ROLE,
        "pilot_id": PILOT_ID,
        "status": "RUNNING",
        "output_root": str(output_root.resolve()),
        "formal_output_root": str(FORBIDDEN_FORMAL_ROOT),
        "formal_output_root_touched": False,
        "formal_request_loaded": False,
        "formal_request_consumed": False,
        "effect_fields_persisted": False,
        "effect_analysis_performed": False,
        "population_size": POPULATION_SIZE,
        "archive_capacity": ARCHIVE_CAPACITY,
        "thread_environment": dict(THREAD_ENVIRONMENT),
        "code_identity": identity,
        "automatic_retries": 0,
    }
    assert_control_plane_only(root_marker)
    (output_root / "pilot_status.json").write_bytes(
        canonical_json_bytes(root_marker) + b"\n"
    )

    sweeps: list[dict[str, Any]] = []
    projections: list[dict[str, Any]] = []
    task_ids: list[str] = []
    for worker_count in counts:
        tasks = build_tasks(
            worker_count=worker_count,
            repetitions_per_profile=repetitions_per_profile,
            cfe_per_event=cfe_per_event,
            dynamic_events=dynamic_events,
        )
        task_ids.extend(task.task_id for task in tasks)
        sweep_dir = output_root / f"workers-{worker_count:02d}"
        sweep_dir.mkdir()
        aggregate, records = _run_worker_sweep(
            worker_count=worker_count,
            tasks=tasks,
        )
        projection = _projection(
            worker_count=worker_count,
            records=records,
        )
        assert_control_plane_only(records)
        assert_control_plane_only(projection)
        (sweep_dir / "task_control_metrics.jsonl").write_bytes(
            b"".join(
                canonical_json_bytes(record) + b"\n"
                for record in records
            )
        )
        (sweep_dir / "sweep_summary.json").write_bytes(
            canonical_json_bytes(aggregate) + b"\n"
        )
        sweeps.append(aggregate)
        projections.append(projection)

    if len(task_ids) != len(set(task_ids)):
        raise ResourcePilotError("pilot task IDs overlap between sweeps")
    failed_count = sum(int(sweep["failed_task_count"]) for sweep in sweeps)
    report = {
        "artifact_role": ARTIFACT_ROLE,
        "pilot_id": PILOT_ID,
        "status": "PASS" if failed_count == 0 else "FAILED_NO_RETRY",
        "output_root": str(output_root.resolve()),
        "scope": "NONFORMAL_RESULT_BLIND_RESOURCE_QUALIFICATION",
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "logical_processors": os.cpu_count(),
        },
        "code_identity": identity,
        "runtime_contract": {
            "population_size": POPULATION_SIZE,
            "archive_capacity": ARCHIVE_CAPACITY,
            "single_thread_per_worker": True,
            "thread_environment": dict(THREAD_ENVIRONMENT),
            "algorithm_coverage": (
                "DT-RAMDE FULL and registered ablation paths on one public "
                "representative adapter per workload stratum"
            ),
            "comparator_family_throughput_measured": False,
            "raw_effect_data_plane_persisted": False,
            "raw_gzip_and_filesystem_contention_measured": False,
            "formal_request_loaded": False,
            "formal_request_consumed": False,
            "r6_engineering_request_only": True,
            "participant_data_accessed": False,
            "effect_fields_persisted": False,
            "effect_analysis_performed": False,
            "automatic_retries": 0,
        },
        "pilot_design": {
            "worker_counts": list(counts),
            "repetitions_per_profile": repetitions_per_profile,
            "profiles": [
                {
                    **asdict(profile),
                    "cfe_per_event": cfe_per_event,
                }
                for profile in _profiles(dynamic_events)
            ],
            "task_count": len(task_ids),
            "all_task_ids_unique": True,
        },
        "sweeps": sweeps,
        "e1_e2_projection": {
            "formal_cfe_by_profile": dict(E1_E2_FORMAL_CFE),
            "formal_total_cfe": sum(E1_E2_FORMAL_CFE.values()),
            "projections": projections,
            "interpretation": (
                "local-host compute-path engineering estimate only; it "
                "excludes comparator-family and raw-gzip/filesystem "
                "contention and cannot qualify an unmeasured 64-vCPU target "
                "instance"
            ),
        },
        "target_resource_gate": {
            "target_description": "AMD EPYC 9754 family, 64 vCPU allocation",
            "target_memory_gib": 80,
            "target_measured": False,
            "formal_go": False,
            "status": "NO_GO_TARGET_HOST_UNMEASURED",
        },
        "failed_task_count": failed_count,
    }
    assert_control_plane_only(report)
    report_path = output_root / "resource_qualification_report.json"
    report_path.write_bytes(canonical_json_bytes(report) + b"\n")
    final_status = {
        **root_marker,
        "status": report["status"],
        "report_sha256": sha256(report_path.read_bytes()).hexdigest(),
        "task_count": len(task_ids),
        "failed_task_count": failed_count,
    }
    assert_control_plane_only(final_status)
    (output_root / "pilot_status.json").write_bytes(
        canonical_json_bytes(final_status) + b"\n"
    )
    return report
