"""Result-blind, full-path target qualification for staged R8C E1+E2.

The qualification exercises every method/workload path in the frozen E1+E2
schedule at population/archive 100/100 on public development instances.  The
real evaluator returns are used by the optimizers in memory, but they are never
serialized.  A transparent proxy instead submits a deterministic,
high-entropy synthetic success/failure stream to the production
:class:`TaskCheckpointWriter`.  This exercises the endpoint-sufficient
21-checkpoint fixed-shape binary path without producing effect evidence.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np

from benchmark_adapters.cdf_operational import (
    CDF_OPERATIONAL_AUTHORITY_ID,
    CDF_OPERATIONAL_SUITE_ID,
)
from benchmark_adapters.r4_wgt_rr import WGTRRPublicAdapter
from comparators.jmetal_bridge import JMetalComparator
from comparators.matched_de import MatchedParetoDE
from dt_ramde_v11.contracts import (
    AlgorithmConfig,
    COMPACT_CHECKPOINT_AUDIT_MATERIALIZATION,
    ExecutionScope,
    R6ExecutionRequest,
)
from dt_ramde_v11.engine import DTRAMDE
from evaluation.contracts import EvaluationResult, TerminalCode
from evaluation.evaluator import BatchEvaluationUnavailableBeforeEntry
from evaluation.ledger import EvaluationLedger
from formal_execution.checkpoint_data import (
    CHECKPOINTS_PER_EVENT,
    FORMAT_ID as CHECKPOINT_FORMAT_ID,
    CheckpointMetadata,
    TaskCheckpointWriter,
    estimate_e1e2_checkpoint_storage,
)
from formal_execution.adapters import (
    make_corrective_cdf_adapter,
    make_corrective_lircmop_adapter,
)
from formal_execution.host import (
    host_fingerprint,
    host_fingerprint_sha256,
)
from formal_execution.schedule import (
    build_corrective_e1e2_formal_schedule,
    canonical_json_bytes,
)


ARTIFACT_ROLE = (
    "R8C_E1E2_RESULT_BLIND_FULL_PATH_TARGET_QUALIFICATION"
)
QUALIFICATION_ID = "WGT-V11-R8C-E1E2-TARGET-QUALIFICATION-20260726-02"
CDF_OPERATIONAL_AUTHORITY_AMENDMENT_ID = (
    "WGT-V11-R8C-E1E2-CDF-OPERATIONAL-AUTHORITY-AMENDMENT-01"
)
DEFAULT_WORKERS = (1, 8, 16, 24, 32, 48, 64)
DEFAULT_REPETITIONS = 2
STATIC_CFE_PER_EVENT = 50_000
DYNAMIC_CFE_PER_EVENT = 5_000
ROLLING_CFE_PER_EVENT = 5_000
# Backward-compatible uniform diagnostic scale.  It is never target-qualifying.
DEFAULT_CFE_PER_EVENT = DYNAMIC_CFE_PER_EVENT
DEFAULT_DYNAMIC_EVENTS = 6
POPULATION_SIZE = 100
ARCHIVE_CAPACITY = 100
CHECKPOINT_FILENAME = "synthetic_endpoint_checkpoints.cfe"
ENDPOINT_SUFFICIENT_FORMAT = CHECKPOINT_FORMAT_ID
SYNTHETIC_FAILURE_INTERVAL = 37
PROJECT_ROOT = Path(__file__).resolve().parents[2]
R5_CONTRACT_PATH = PROJECT_ROOT / "config" / "r5" / "r5_freeze_contract.json"
LINUX_RUNTIME_LOCK_PATH = (
    PROJECT_ROOT / "requirements-r8c-linux-x86_64.lock"
)
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
_UINT64_MASK = (1 << 64) - 1
RSS_SAFETY_FACTOR = 1.25
RSS_ROUNDING_BYTES = 1 << 20
SUPERVISOR_MEMORY_RESERVE_BYTES = 4 << 30
_LOCK_REQUIREMENT_PATTERN = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\s]+) "
    r"--hash=sha256:([0-9a-f]{64})$"
)

FORMAL_METHODS_BY_WORKLOAD: Mapping[str, tuple[str, ...]] = {
    "E1_STATIC": (
        "F22_MG_STATIC",
        "MATCHED_FIXED_DE_PARETO",
        "MATCHED_JDE_STYLE_PARETO",
        "MATCHED_SHADE_STYLE_PARETO",
        "JMETALPY_1_7_GDE3_STANDARD_PARETO_DE",
        "JMETALPY_1_7_NSGAII_STATIC_CMOEA",
    ),
    "E1_DYNAMIC": (
        "DT-RAMDE_TS2_FULL",
        "MATCHED_FIXED_DE_PARETO",
        "MATCHED_JDE_STYLE_PARETO",
        "MATCHED_SHADE_STYLE_PARETO",
        "JMETALPY_1_7_GDE3_STANDARD_PARETO_DE",
        "JMETALPY_1_7_NSGAII_DYNAMIC_RESTART_BRIDGE",
    ),
    "E1_ROLLING": (
        "DT-RAMDE_TS2_FULL",
        "MATCHED_FIXED_DE_PARETO",
        "MATCHED_JDE_STYLE_PARETO",
        "MATCHED_SHADE_STYLE_PARETO",
        "JMETALPY_1_7_GDE3_STANDARD_PARETO_DE",
        "JMETALPY_1_7_NSGAII_DYNAMIC_RESTART_BRIDGE",
    ),
    "E2_DYNAMIC_INCREMENTAL_AFTER_FULL_REUSE": (
        "NO_CROSS_EVENT_MEMORY",
        "NO_REJECTION_CREDIT",
        "NO_MEMORY_RESET_GATE",
        "NO_LINEAGE_CREDIT",
        "CROSS_EVENT_WARM_START_ONLY",
        "CROSS_EVENT_MEMORY_ONLY",
        "SHADE_ONLY",
    ),
    "E2_ROLLING_INCREMENTAL_AFTER_FULL_REUSE": (
        "NO_CROSS_EVENT_MEMORY",
        "NO_EXECUTION_FEEDBACK",
        "NO_REJECTION_CREDIT",
        "NO_MEMORY_RESET_GATE",
        "NO_LINEAGE_CREDIT",
        "CROSS_EVENT_WARM_START_ONLY",
        "CROSS_EVENT_MEMORY_ONLY",
        "SHADE_ONLY",
    ),
}
REPRESENTATIVE_CASES_BY_ADAPTER: Mapping[
    str,
    tuple[tuple[str, str, int | None, str | None], ...],
] = {
    "lircmop": (
        ("LIRCMOP1_2D_HEAD", "STATIC_2D", 1, None),
        ("LIRCMOP12_2D_TAIL", "STATIC_2D", 12, None),
        ("LIRCMOP14_3D", "STATIC_3D", 14, None),
    ),
    "cdf": (
        ("CDF1_HARSH_BASE", "DYNAMIC_2D", 1, "CDF-HARSH"),
        (
            "CDF9_HARSH_DOMAIN_FALLBACK",
            "DYNAMIC_2D",
            9,
            "CDF-HARSH",
        ),
        ("CDF13_HARSH_SEED_DEPENDENT", "DYNAMIC_2D", 13, "CDF-HARSH"),
        ("CDF15_MILD_TAIL", "DYNAMIC_2D", 15, "CDF-MILD"),
    ),
    "rolling": (
        ("WGT_RR_KNOWN_ANSWER", "ROLLING_3D", None, None),
    ),
}

PROHIBITED_CONTROL_KEYS = frozenset(
    {
        "action",
        "archive",
        "candidate",
        "candidate_id",
        "constraint",
        "constraints",
        "execution_feedback",
        "feasible",
        "feedback",
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


class E1E2QualificationError(RuntimeError):
    """The result-blind target qualification contract was violated."""


@dataclass(frozen=True)
class QualificationProfile:
    """One representative full-path workload/method binding."""

    workload_id: str
    method_id: str
    execution_kind: str
    adapter_kind: str
    events: int
    atomic_steps_per_cfe: int
    representative_case_id: str
    projection_rate_class: str
    problem_index: int | None
    dynamic_profile: str | None

    @property
    def key(self) -> tuple[str, str]:
        return (self.workload_id, self.method_id)

    @property
    def case_key(self) -> tuple[str, str, str]:
        return (
            self.workload_id,
            self.method_id,
            self.representative_case_id,
        )

    @property
    def rate_key(self) -> tuple[str, str, str]:
        return (
            self.workload_id,
            self.method_id,
            self.projection_rate_class,
        )


@dataclass(frozen=True)
class QualificationTask:
    """A deterministic task identity for one worker-count sweep."""

    task_id: str
    worker_count: int
    ordinal: int
    repetition: int
    profile: QualificationProfile
    cfe_per_event: int
    seed: int
    task_directory: Path


def _method_kind(method_id: str) -> str:
    if method_id.startswith("MATCHED_") or method_id.startswith("JMETALPY_"):
        return "comparator"
    return "dt_ramde"


def qualification_profiles(
    dynamic_events: int = DEFAULT_DYNAMIC_EVENTS,
) -> tuple[QualificationProfile, ...]:
    """Return 84 workload/method/case bindings over eight benchmark cases."""

    if dynamic_events < 1:
        raise E1E2QualificationError("dynamic events must be positive")
    profiles: list[QualificationProfile] = []
    for workload_id, methods in FORMAL_METHODS_BY_WORKLOAD.items():
        if workload_id == "E1_STATIC":
            adapter_kind = "lircmop"
            events = 1
            atomic_steps = 1
        elif "DYNAMIC" in workload_id:
            adapter_kind = "cdf"
            events = dynamic_events
            atomic_steps = 1
        else:
            adapter_kind = "rolling"
            events = dynamic_events
            atomic_steps = 6
        for method_id in methods:
            for (
                case_id,
                rate_class,
                problem_index,
                dynamic_profile,
            ) in REPRESENTATIVE_CASES_BY_ADAPTER[adapter_kind]:
                profiles.append(
                    QualificationProfile(
                        workload_id=workload_id,
                        method_id=method_id,
                        execution_kind=_method_kind(method_id),
                        adapter_kind=adapter_kind,
                        events=events,
                        atomic_steps_per_cfe=atomic_steps,
                        representative_case_id=case_id,
                        projection_rate_class=rate_class,
                        problem_index=problem_index,
                        dynamic_profile=dynamic_profile,
                    )
                )
    if (
        len(profiles) != 84
        or len({profile.case_key for profile in profiles}) != 84
        or len({profile.key for profile in profiles}) != 33
    ):
        raise E1E2QualificationError(
            "qualification matrix is not the frozen 33-path/84-binding matrix"
        )
    return tuple(profiles)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def runtime_environment_lock_evidence() -> dict[str, Any]:
    """Bind all Linux target lock pins to the live interpreter environment."""

    try:
        lock_bytes = LINUX_RUNTIME_LOCK_PATH.read_bytes()
    except OSError as error:
        raise E1E2QualificationError(
            "Linux target runtime lock is missing"
        ) from error
    locked_versions: dict[str, str] = {}
    locked_wheel_sha256: dict[str, str] = {}
    for raw_line in lock_bytes.decode("utf-8", errors="strict").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        match = _LOCK_REQUIREMENT_PATTERN.fullmatch(line)
        if match is None:
            raise E1E2QualificationError(
                "Linux target runtime lock contains an unsupported line"
            )
        distribution, version, wheel_sha256 = match.groups()
        normalized = distribution.lower().replace("_", "-")
        if normalized in locked_versions:
            raise E1E2QualificationError(
                "Linux target runtime lock repeats a distribution"
            )
        locked_versions[normalized] = version
        locked_wheel_sha256[normalized] = wheel_sha256
    if len(locked_versions) != 35:
        raise E1E2QualificationError(
            "Linux target runtime lock is not the frozen 35-package closure"
        )

    installed_versions: dict[str, str | None] = {}
    for distribution in sorted(locked_versions):
        try:
            installed_versions[distribution] = importlib_metadata.version(
                distribution
            )
        except importlib_metadata.PackageNotFoundError:
            installed_versions[distribution] = None
    missing = [
        distribution
        for distribution, version in installed_versions.items()
        if version is None
    ]
    mismatches = {
        distribution: {
            "locked": locked_versions[distribution],
            "installed": installed_versions[distribution],
        }
        for distribution in sorted(locked_versions)
        if installed_versions[distribution] is not None
        and installed_versions[distribution] != locked_versions[distribution]
    }
    interpreter_matches = (
        platform.python_implementation() == "CPython"
        and sys.version_info[:2] == (3, 12)
    )
    machine = platform.machine().lower()
    platform_matches = (
        platform.system() == "Linux"
        and machine in {"x86_64", "amd64"}
    )
    return {
        "artifact_role": "R8C_E1E2_LINUX_RUNTIME_LOCK_EVIDENCE",
        "lock_path": str(
            LINUX_RUNTIME_LOCK_PATH.relative_to(PROJECT_ROOT).as_posix()
        ),
        "lock_sha256": sha256(lock_bytes).hexdigest(),
        "lock_bytes": len(lock_bytes),
        "target_interpreter": "CPython 3.12",
        "target_platform": "Linux x86_64 manylinux2014-compatible",
        "actual_python_implementation": platform.python_implementation(),
        "actual_python_version": platform.python_version(),
        "actual_system": platform.system(),
        "actual_machine": platform.machine(),
        "locked_package_count": len(locked_versions),
        "locked_package_versions": {
            key: locked_versions[key] for key in sorted(locked_versions)
        },
        "locked_wheel_sha256": {
            key: locked_wheel_sha256[key]
            for key in sorted(locked_wheel_sha256)
        },
        "installed_package_versions": installed_versions,
        "missing_locked_packages": missing,
        "version_mismatches": mismatches,
        "interpreter_matches": interpreter_matches,
        "platform_matches": platform_matches,
        "all_locked_packages_match": not missing and not mismatches,
        "target_environment_match": (
            interpreter_matches
            and platform_matches
            and not missing
            and not mismatches
        ),
    }


def build_tasks(
    *,
    output_root: Path,
    worker_count: int,
    repetitions: int,
    cfe_per_event: int | None = None,
    static_cfe_per_event: int = STATIC_CFE_PER_EVENT,
    dynamic_cfe_per_event: int = DYNAMIC_CFE_PER_EVENT,
    rolling_cfe_per_event: int = ROLLING_CFE_PER_EVENT,
    dynamic_events: int,
) -> tuple[QualificationTask, ...]:
    """Build unique deterministic task identities for one sweep."""

    if worker_count < 1 or repetitions < 1:
        raise E1E2QualificationError(
            "worker count and repetitions must be positive"
        )
    budgets = (
        {
            "lircmop": int(cfe_per_event),
            "cdf": int(cfe_per_event),
            "rolling": int(cfe_per_event),
        }
        if cfe_per_event is not None
        else {
            "lircmop": int(static_cfe_per_event),
            "cdf": int(dynamic_cfe_per_event),
            "rolling": int(rolling_cfe_per_event),
        }
    )
    if any(
        budget < POPULATION_SIZE or budget % POPULATION_SIZE != 0
        for budget in budgets.values()
    ):
        raise E1E2QualificationError(
            "each qualification CFE/event budget must be a positive "
            "multiple of population 100"
        )
    tasks: list[QualificationTask] = []
    for repetition in range(repetitions):
        for profile in qualification_profiles(dynamic_events):
            ordinal = len(tasks)
            seed = 8_202_607_260 + worker_count * 10_000 + ordinal
            task_cfe_per_event = budgets[profile.adapter_kind]
            binding = {
                "qualification_id": QUALIFICATION_ID,
                "worker_count": worker_count,
                "ordinal": ordinal,
                "repetition": repetition,
                "workload_id": profile.workload_id,
                "method_id": profile.method_id,
                "representative_case_id": profile.representative_case_id,
                "projection_rate_class": profile.projection_rate_class,
                "problem_index": profile.problem_index,
                "dynamic_profile": profile.dynamic_profile,
                "cfe_per_event": task_cfe_per_event,
                "events": profile.events,
                "seed": seed,
                "population_size": POPULATION_SIZE,
                "archive_capacity": ARCHIVE_CAPACITY,
            }
            digest = sha256(_canonical_bytes(binding)).hexdigest()[:16]
            task_id = (
                f"r8c-e1e2-qual-w{worker_count:02d}-"
                f"{ordinal:04d}-{digest}"
            )
            tasks.append(
                QualificationTask(
                    task_id=task_id,
                    worker_count=worker_count,
                    ordinal=ordinal,
                    repetition=repetition,
                    profile=profile,
                    cfe_per_event=task_cfe_per_event,
                    seed=seed,
                    task_directory=(
                        output_root
                        / f"workers-{worker_count:02d}"
                        / "tasks"
                        / task_id
                    ),
                )
            )
    if len({task.task_id for task in tasks}) != len(tasks):
        raise E1E2QualificationError("qualification task IDs are not unique")
    return tuple(tasks)


class _QualificationSelector:
    selector_id = "WGT-V11-R8C-E1E2-QUALIFICATION-SELECTOR-01"
    selector_version = "1.0.0"

    def __init__(self, *, rolling: bool) -> None:
        self.rolling = bool(rolling)

    def identity(self) -> Mapping[str, Any]:
        return {
            "selector_id": self.selector_id,
            "selector_version": self.selector_version,
            "role": "result_blind_resource_qualification_internal_only",
            "rolling": self.rolling,
        }

    def select(
        self,
        archive: Sequence[EvaluationResult],
    ) -> str | None:
        values = tuple(archive)
        if not values:
            return None
        if not self.rolling:
            return min(
                values,
                key=lambda item: (item.objectives, item.candidate_id),
            ).candidate_id
        matrix = np.asarray(
            [item.objectives for item in values],
            dtype=float,
        )
        transformed = matrix / (1.0 + matrix)
        ideal = np.min(transformed, axis=0)
        weighted = (transformed - ideal) / matrix.shape[1]
        scores = np.max(weighted, axis=1) + 1e-6 * np.sum(
            weighted,
            axis=1,
        )
        return min(
            zip(scores.tolist(), values, strict=True),
            key=lambda pair: (pair[0], pair[1].candidate_id),
        )[1].candidate_id


def _mix_uint64(value: int) -> int:
    """SplitMix64 finalizer for deterministic synthetic entropy."""

    mixed = (value + 0x9E3779B97F4A7C15) & _UINT64_MASK
    mixed = (
        (mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9
    ) & _UINT64_MASK
    mixed = (
        (mixed ^ (mixed >> 27)) * 0x94D049BB133111EB
    ) & _UINT64_MASK
    return (mixed ^ (mixed >> 31)) & _UINT64_MASK


def _synthetic_word(sequence: int, field_group: int, index: int) -> int:
    """Derive one word solely from synthetic sequence and field position."""

    binding = (
        int(sequence) * 0xD1342543DE82EF95
        + int(field_group) * 0xA24BAED4963EE407
        + int(index) * 0x9FB21C651E98DF25
    ) & _UINT64_MASK
    return _mix_uint64(binding)


def _synthetic_float(sequence: int, field_group: int, index: int) -> float:
    """Return a finite high-entropy float unrelated to any evaluator value."""

    mantissa = _synthetic_word(sequence, field_group, index) >> 11
    unit = (mantissa + 0.5) / float(1 << 53)
    return (unit * 2.0 - 1.0) * 1_000_000.0


def _synthetic_candidate_id(sequence: int) -> str:
    """Return a unique, visibly synthetic high-entropy task-local ID."""

    entropy = "".join(
        f"{_synthetic_word(sequence, 0, index):016x}"
        for index in range(3)
    )
    return f"SYNTHETIC-{sequence:016x}-{entropy}"


class _SyntheticRecordingAdapter:
    """Delegate real evaluation while checkpointing only synthetic values."""

    def __init__(self, problem: Any, writer: TaskCheckpointWriter) -> None:
        self._problem = problem
        self._writer = writer
        self.synthetic_write_seconds = 0.0
        self.synthetic_record_count = 0
        self.synthetic_success_count = 0
        self.synthetic_failure_count = 0
        self._synthetic_sequence = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._problem, name)

    def identity(self) -> Mapping[str, Any]:
        return self._problem.identity()

    def _next_sequence(self) -> int:
        self._synthetic_sequence += 1
        return self._synthetic_sequence

    def begin_event(self, *, event_id: int, cfe_budget: int) -> None:
        """Open the production writer's fixed 21-point event schedule."""

        self._writer.begin_event(event_id=event_id, cfe_budget=cfe_budget)

    def finish_event(self, *, terminal_snapshot: bool = False) -> None:
        """Close an event, retaining a typed short-CFE terminal snapshot."""

        self._writer.finish_event(terminal_snapshot=terminal_snapshot)

    @staticmethod
    def _dummy_result(
        *,
        sequence: int,
        objective_count: int,
        constraint_count: int,
    ) -> EvaluationResult:
        objectives = [
            float(sequence),
            -float(sequence),
        ]
        objectives.extend(
            _synthetic_float(sequence, 2, index)
            for index in range(2, objective_count)
        )
        return EvaluationResult(
            candidate_id=_synthetic_candidate_id(sequence),
            objectives=tuple(objectives[:objective_count]),
            objective_names=tuple(
                f"synthetic_metric_{index:03d}"
                for index in range(objective_count)
            ),
            constraints=tuple(
                -abs(_synthetic_float(sequence, 3, index)) - 1e-12
                for index in range(constraint_count)
            ),
            constraint_names=tuple(
                f"synthetic_limit_{index:03d}"
                for index in range(constraint_count)
            ),
        )

    @staticmethod
    def _synthetic_failure_due(sequence: int) -> bool:
        return sequence % SYNTHETIC_FAILURE_INTERVAL == 0

    def _record_dummy(
        self,
        *,
        event_id: int,
        vector_dimension: int,
        objective_count: int,
        constraint_count: int,
    ) -> None:
        sequence = self._next_sequence()
        vector = tuple(
            _synthetic_float(sequence, 1, index)
            for index in range(vector_dimension)
        )
        started = time.perf_counter()
        if self._synthetic_failure_due(sequence):
            self._writer.record_failure(
                event_id=event_id,
                candidate_id=_synthetic_candidate_id(sequence),
                vector=vector,
                error_type="SyntheticQualificationFailure",
                reason=(
                    "SYNTHETIC_FAILURE_"
                    f"{_synthetic_word(sequence, 4, 0):016x}"
                ),
            )
            self.synthetic_failure_count += 1
        else:
            self._writer.record_success(
                event_id=event_id,
                vector=vector,
                result=self._dummy_result(
                    sequence=sequence,
                    objective_count=objective_count,
                    constraint_count=constraint_count,
                ),
            )
            self.synthetic_success_count += 1
        self.synthetic_record_count += 1
        self.synthetic_write_seconds += time.perf_counter() - started

    def _write_dummy_failure(
        self,
        *,
        event_id: int,
        vector_dimension: int,
    ) -> None:
        sequence = self._next_sequence()
        started = time.perf_counter()
        self._writer.record_failure(
            event_id=event_id,
            candidate_id=_synthetic_candidate_id(sequence),
            vector=tuple(
                _synthetic_float(sequence, 1, index)
                for index in range(vector_dimension)
            ),
            error_type="SyntheticQualificationFailure",
            reason=(
                "SYNTHETIC_REAL_PATH_FAILURE_"
                f"{_synthetic_word(sequence, 5, 0):016x}"
            ),
        )
        self.synthetic_record_count += 1
        self.synthetic_failure_count += 1
        self.synthetic_write_seconds += time.perf_counter() - started

    def evaluate(
        self,
        vector: Sequence[float],
        event_id: int,
        ledger: EvaluationLedger,
        candidate_id: str,
    ) -> EvaluationResult:
        try:
            result = self._problem.evaluate(
                vector,
                event_id,
                ledger,
                candidate_id,
            )
        except Exception:
            self._write_dummy_failure(
                event_id=event_id,
                vector_dimension=len(vector),
            )
            raise
        self._record_dummy(
            event_id=event_id,
            vector_dimension=len(vector),
            objective_count=len(result.objectives),
            constraint_count=len(result.constraints),
        )
        return result

    def evaluate_batch(
        self,
        vectors: Sequence[Sequence[float]],
        event_id: int,
        ledger: EvaluationLedger,
        candidate_ids: Sequence[str],
    ) -> tuple[EvaluationResult, ...]:
        vector_values = tuple(vectors)
        id_values = tuple(str(value) for value in candidate_ids)
        if len(vector_values) != len(id_values):
            raise ValueError("batch vectors and candidate IDs must align")
        evaluator = getattr(self._problem, "evaluate_batch", None)
        if not callable(evaluator):
            raise BatchEvaluationUnavailableBeforeEntry(
                "qualification adapter has no ordered batch method"
            )
        try:
            results = tuple(
                evaluator(
                    vector_values,
                    event_id,
                    ledger,
                    id_values,
                )
            )
        except BatchEvaluationUnavailableBeforeEntry:
            # The engine will retry this generation through the scalar path.
            # This exception guarantees that the real evaluator and its ledger
            # were not entered, so the synthetic checkpoint stream must also
            # remain untouched.
            raise
        except Exception:
            for vector in vector_values:
                self._write_dummy_failure(
                    event_id=event_id,
                    vector_dimension=len(vector),
                )
            raise
        if len(results) != len(vector_values):
            raise E1E2QualificationError(
                "batch evaluator returned the wrong result count"
            )
        for vector, candidate_id, result in zip(
            vector_values,
            id_values,
            results,
            strict=True,
        ):
            if result.candidate_id != candidate_id:
                raise E1E2QualificationError(
                    "batch evaluator changed candidate identity or order"
                )
            self._record_dummy(
                event_id=event_id,
                vector_dimension=len(vector),
                objective_count=len(result.objectives),
                constraint_count=len(result.constraints),
            )
        return results


def _build_problem(task: QualificationTask) -> Any:
    if task.profile.adapter_kind == "lircmop":
        if task.profile.problem_index is None:
            raise E1E2QualificationError(
                "LIR-CMOP qualification case lacks a problem index"
            )
        return make_corrective_lircmop_adapter(task.profile.problem_index)
    if task.profile.adapter_kind == "cdf":
        if (
            task.profile.problem_index is None
            or task.profile.dynamic_profile is None
        ):
            raise E1E2QualificationError(
                "CDF qualification case lacks problem/profile identity"
            )
        return make_corrective_cdf_adapter(
            task.profile.problem_index,
            profile=task.profile.dynamic_profile,
            environment_seed=task.seed,
        )
    if task.profile.adapter_kind == "rolling":
        return WGTRRPublicAdapter.from_known_answer()
    raise E1E2QualificationError(
        f"unknown adapter kind {task.profile.adapter_kind!r}"
    )


def _dt_ramde_variant(profile: QualificationProfile) -> str:
    if profile.method_id == "F22_MG_STATIC":
        return "NO_CROSS_EVENT_MEMORY"
    if profile.method_id == "DT-RAMDE_TS2_FULL":
        return "FULL"
    return profile.method_id


def _run_dt_ramde(
    task: QualificationTask,
    problem: _SyntheticRecordingAdapter,
) -> int:
    request = R6ExecutionRequest(scope=ExecutionScope.ENGINEERING_PILOT)
    request.validate()
    timing = (
        "TS1_single_event"
        if task.profile.workload_id == "E1_STATIC"
        else "TS2_fixed_periodic_replanning"
    )
    method_label = (
        "F22_MG_STATIC"
        if timing == "TS1_single_event"
        else task.profile.method_id
    )
    selector = _QualificationSelector(
        rolling=task.profile.adapter_kind == "rolling"
    )
    config = AlgorithmConfig(
        variant=_dt_ramde_variant(task.profile),
        population_size=POPULATION_SIZE,
        cfe_per_event=task.cfe_per_event,
        algorithm_seed=task.seed,
        max_events=task.profile.events,
        timing_mode=timing,
        method_label=method_label,
        adapter_id=problem.adapter_id,
        adapter_version=problem.adapter_version,
        selector_id=selector.selector_id,
        selector_version=selector.selector_version,
        atomic_steps_per_evaluation=task.profile.atomic_steps_per_cfe,
        event_time_limit_seconds=3_600.0,
        configuration_evidence_id=(
            "WGT_V11_R8C_E1E2_FULL_PATH_TARGET_QUALIFICATION_PILOT"
        ),
        execution_request=request,
        audit_materialization=COMPACT_CHECKPOINT_AUDIT_MATERIALIZATION,
    )
    engine = DTRAMDE(config)
    prior_feedback: Mapping[str, Any] | None = None
    total_cfe = 0
    for event_id in range(task.profile.events):
        problem.begin_event(
            event_id=event_id,
            cfe_budget=task.cfe_per_event,
        )
        event = engine.run_event(
            problem,
            selector=selector,
            event_id=event_id,
            prior_feedback=prior_feedback,
        )
        event_cfe = int(event.ledger["cfe"])
        if (
            not 0 <= event_cfe <= task.cfe_per_event
            or (
                event_cfe < task.cfe_per_event
                and event.terminal.code
                not in {
                    TerminalCode.REJECT_NUMERICAL,
                    TerminalCode.REJECT_TIMEOUT,
                }
            )
        ):
            raise E1E2QualificationError(
                "DT-RAMDE CFE differs without a typed numerical/timeout "
                "terminal"
            )
        problem.finish_event(
            terminal_snapshot=event_cfe < task.cfe_per_event
        )
        prior_feedback = event.execution_feedback
        total_cfe += event_cfe
    return total_cfe


def _build_comparator(method_id: str) -> Any:
    matched_modes = {
        "MATCHED_FIXED_DE_PARETO": "fixed",
        "MATCHED_JDE_STYLE_PARETO": "jde",
        "MATCHED_SHADE_STYLE_PARETO": "shade",
    }
    if method_id in matched_modes:
        return MatchedParetoDE(
            mode=matched_modes[method_id],
            population_size=POPULATION_SIZE,
            archive_capacity=ARCHIVE_CAPACITY,
            method_id_override=method_id,
        )
    jmetal_modes = {
        "JMETALPY_1_7_GDE3_STANDARD_PARETO_DE": "gde3",
        "JMETALPY_1_7_NSGAII_STATIC_CMOEA": "nsgaii_static",
        "JMETALPY_1_7_NSGAII_DYNAMIC_RESTART_BRIDGE": (
            "nsgaii_dynamic_restart"
        ),
    }
    if method_id in jmetal_modes:
        return JMetalComparator(
            mode=jmetal_modes[method_id],
            population_size=POPULATION_SIZE,
            archive_capacity=ARCHIVE_CAPACITY,
        )
    raise E1E2QualificationError(
        f"unknown comparator method {method_id!r}"
    )


def _shared_initialization(
    problem: Any,
    *,
    task: QualificationTask,
    event_id: int,
) -> list[list[float]]:
    rng = np.random.Generator(
        np.random.PCG64(task.seed + event_id * 1_000_003)
    )
    lower = np.asarray(problem.lower_bounds, dtype=float)
    upper = np.asarray(problem.upper_bounds, dtype=float)
    return [
        rng.uniform(lower, upper).tolist()
        for _ in range(POPULATION_SIZE)
    ]


def _run_comparator(
    task: QualificationTask,
    problem: _SyntheticRecordingAdapter,
) -> int:
    comparator = _build_comparator(task.profile.method_id)
    feedback: Mapping[str, Any] | None = None
    total_cfe = 0
    for event_id in range(task.profile.events):
        problem.begin_event(
            event_id=event_id,
            cfe_budget=task.cfe_per_event,
        )
        problem.freeze_information(event_id, feedback)
        ledger = EvaluationLedger(max_cfe=task.cfe_per_event)
        result = comparator.optimize(
            problem,
            event_id=event_id,
            budget=task.cfe_per_event,
            seed=task.seed + event_id * 104_729,
            ledger=ledger,
            initialization_vectors=_shared_initialization(
                problem,
                task=task,
                event_id=event_id,
            ),
        )
        accepted = result.terminal.code is TerminalCode.ACCEPTED
        action = (
            problem.first_action(result.selected_vector)
            if accepted and result.selected_vector is not None
            else problem.fallback_action(event_id)
        )
        feedback = dict(problem.execute(action, event_id, accepted, ledger))
        ledger.assert_joint_contract(
            atomic_steps_per_evaluation=(
                task.profile.atomic_steps_per_cfe
            )
        )
        snapshot = ledger.snapshot()
        event_cfe = int(snapshot["cfe"])
        if (
            not 0 <= event_cfe <= task.cfe_per_event
            or (
                event_cfe < task.cfe_per_event
                and result.terminal.code
                not in {
                    TerminalCode.REJECT_NUMERICAL,
                    TerminalCode.REJECT_TIMEOUT,
                }
            )
        ):
            raise E1E2QualificationError(
                "comparator CFE differs without a typed numerical/timeout "
                "terminal"
            )
        problem.finish_event(
            terminal_snapshot=event_cfe < task.cfe_per_event
        )
        total_cfe += event_cfe
    return total_cfe


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
        ok = get_memory_info(
            process,
            ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.PeakWorkingSetSize)
    import resource

    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if platform.system() == "Darwin" else peak * 1024


def _run_task(task: QualificationTask) -> dict[str, Any]:
    """Run one path once, with no retry, and return control metrics only."""

    task.task_directory.mkdir(parents=True, exist_ok=False)
    checkpoint_path = task.task_directory / CHECKPOINT_FILENAME
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    writer: TaskCheckpointWriter | None = None
    wrapped: _SyntheticRecordingAdapter | None = None
    problem_identity: dict[str, Any] | None = None
    total_cfe = 0
    status = "PASS"
    error_type: str | None = None
    close_seconds = 0.0
    try:
        base_problem = _build_problem(task)
        problem_identity = dict(base_problem.identity())
        synthetic_names = tuple(
            f"synthetic_metric_{index:03d}"
            for index in range(len(base_problem.objective_names))
        )
        writer = TaskCheckpointWriter(
            checkpoint_path,
            CheckpointMetadata(
                task_id=task.task_id,
                objective_names=synthetic_names,
                archive_capacity=ARCHIVE_CAPACITY,
                checkpoints_per_event=CHECKPOINTS_PER_EVENT,
            ),
        )
        wrapped = _SyntheticRecordingAdapter(base_problem, writer)
        if task.profile.execution_kind == "dt_ramde":
            total_cfe = _run_dt_ramde(task, wrapped)
        else:
            total_cfe = _run_comparator(task, wrapped)
        scheduled_cfe = task.profile.events * task.cfe_per_event
        if (
            not 0 <= total_cfe <= scheduled_cfe
            or wrapped.synthetic_record_count != total_cfe
        ):
            raise E1E2QualificationError(
                "task accounting or checkpoint evaluation count differs"
            )
    except BaseException as error:
        status = "FAILED"
        error_type = type(error).__name__
    finally:
        if writer is not None:
            close_started = time.perf_counter()
            writer.close()
            close_seconds = time.perf_counter() - close_started
    wall_seconds = time.perf_counter() - started_wall
    cpu_seconds = time.process_time() - started_cpu
    record = {
        "artifact_role": ARTIFACT_ROLE,
        "task_id": task.task_id,
        "workload_id": task.profile.workload_id,
        "method_id": task.profile.method_id,
        "representative_case_id": task.profile.representative_case_id,
        "projection_rate_class": task.profile.projection_rate_class,
        "representative_problem_index": task.profile.problem_index,
        "representative_dynamic_profile": task.profile.dynamic_profile,
        "representative_problem_identity": problem_identity,
        "execution_kind": task.profile.execution_kind,
        "worker_count": task.worker_count,
        "ordinal": task.ordinal,
        "repetition": task.repetition,
        "status": status,
        "error_type": error_type,
        "attempt": 1,
        "automatic_retries": 0,
        "population_size": POPULATION_SIZE,
        "archive_capacity": ARCHIVE_CAPACITY,
        "event_count": task.profile.events,
        "cfe_per_event": task.cfe_per_event,
        "scheduled_cfe": task.profile.events * task.cfe_per_event,
        "cfe_consumed": total_cfe,
        "unconsumed_cfe_due_to_typed_terminal": (
            task.profile.events * task.cfe_per_event - total_cfe
        ),
        "atomic_steps_per_cfe": task.profile.atomic_steps_per_cfe,
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "peak_rss_bytes": _peak_rss_bytes(),
        "checkpoint_evaluation_count": (
            0 if wrapped is None else wrapped.synthetic_record_count
        ),
        "checkpoint_synthetic_success_count": (
            0 if wrapped is None else wrapped.synthetic_success_count
        ),
        "checkpoint_synthetic_failure_count": (
            0 if wrapped is None else wrapped.synthetic_failure_count
        ),
        "checkpoint_record_count": (
            task.profile.events * CHECKPOINTS_PER_EVENT
            if status == "PASS"
            else 0
        ),
        "checkpoint_file_bytes": (
            checkpoint_path.stat().st_size
            if checkpoint_path.exists()
            else 0
        ),
        "checkpoint_record_write_seconds": (
            0.0 if wrapped is None else wrapped.synthetic_write_seconds
        ),
        "checkpoint_close_seconds": close_seconds,
        "production_checkpoint_writer_used": writer is not None,
        "endpoint_sufficient_format": ENDPOINT_SUFFICIENT_FORMAT,
        "checkpoint_points_per_event": CHECKPOINTS_PER_EVENT,
        "r6_engineering_request_used_for_dt_ramde": (
            task.profile.execution_kind == "dt_ramde"
        ),
        "real_comparator_optimize_called": (
            task.profile.execution_kind == "comparator"
        ),
        "real_effect_values_persisted": False,
        "formal_execution_started": False,
    }
    assert_control_plane_only(record)
    return record


def assert_control_plane_only(value: Any, path: str = "$") -> None:
    """Reject effect-bearing keys from every JSON control artifact."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in PROHIBITED_CONTROL_KEYS:
                raise E1E2QualificationError(
                    f"effect-bearing key {key!r} at {path} is prohibited"
                )
            assert_control_plane_only(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_control_plane_only(item, f"{path}[{index}]")


def _infrastructure_failure(
    task: QualificationTask,
    error: BaseException,
) -> dict[str, Any]:
    checkpoint_path = task.task_directory / CHECKPOINT_FILENAME
    record = {
        "artifact_role": ARTIFACT_ROLE,
        "task_id": task.task_id,
        "workload_id": task.profile.workload_id,
        "method_id": task.profile.method_id,
        "representative_case_id": task.profile.representative_case_id,
        "projection_rate_class": task.profile.projection_rate_class,
        "representative_problem_index": task.profile.problem_index,
        "representative_dynamic_profile": task.profile.dynamic_profile,
        "execution_kind": task.profile.execution_kind,
        "worker_count": task.worker_count,
        "ordinal": task.ordinal,
        "repetition": task.repetition,
        "status": "FAILED",
        "error_type": type(error).__name__,
        "attempt": 1,
        "automatic_retries": 0,
        "population_size": POPULATION_SIZE,
        "archive_capacity": ARCHIVE_CAPACITY,
        "event_count": task.profile.events,
        "cfe_per_event": task.cfe_per_event,
        "cfe_consumed": 0,
        "atomic_steps_per_cfe": task.profile.atomic_steps_per_cfe,
        "wall_seconds": 0.0,
        "cpu_seconds": 0.0,
        "peak_rss_bytes": 0,
        "checkpoint_evaluation_count": 0,
        "checkpoint_synthetic_success_count": 0,
        "checkpoint_synthetic_failure_count": 0,
        "checkpoint_record_count": 0,
        "checkpoint_file_bytes": (
            checkpoint_path.stat().st_size
            if checkpoint_path.exists()
            else 0
        ),
        "checkpoint_record_write_seconds": 0.0,
        "checkpoint_close_seconds": 0.0,
        "production_checkpoint_writer_used": checkpoint_path.exists(),
        "endpoint_sufficient_format": ENDPOINT_SUFFICIENT_FORMAT,
        "checkpoint_points_per_event": CHECKPOINTS_PER_EVENT,
        "r6_engineering_request_used_for_dt_ramde": (
            task.profile.execution_kind == "dt_ramde"
        ),
        "real_comparator_optimize_called": False,
        "real_effect_values_persisted": False,
        "formal_execution_started": False,
    }
    assert_control_plane_only(record)
    return record


def _run_sweep(
    *,
    worker_count: int,
    tasks: Sequence[QualificationTask],
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    started = time.perf_counter()
    by_id = {task.task_id: task for task in tasks}
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(_run_task, task): task.task_id for task in tasks
        }
        for future in as_completed(futures):
            task = by_id[futures[future]]
            try:
                record = future.result()
            except BaseException as error:
                record = _infrastructure_failure(task, error)
            records.append(record)
    sweep_seconds = time.perf_counter() - started
    records.sort(key=lambda item: int(item["ordinal"]))
    passed = tuple(row for row in records if row["status"] == "PASS")
    failed = tuple(row for row in records if row["status"] != "PASS")
    total_cfe = sum(int(row["cfe_consumed"]) for row in passed)
    total_bytes = sum(int(row["checkpoint_file_bytes"]) for row in records)
    aggregate = {
        "artifact_role": ARTIFACT_ROLE,
        "workers_requested": worker_count,
        "task_count": len(records),
        "passed_task_count": len(passed),
        "failed_task_count": len(failed),
        "sweep_wall_seconds": sweep_seconds,
        "sum_task_cpu_seconds": sum(
            float(row["cpu_seconds"]) for row in passed
        ),
        "cfe_consumed": total_cfe,
        "cfe_per_sweep_wall_second": (
            total_cfe / sweep_seconds if sweep_seconds > 0.0 else None
        ),
        "max_task_peak_rss_bytes": (
            max(int(row["peak_rss_bytes"]) for row in records)
            if records
            else 0
        ),
        "sum_task_peak_rss_upper_bound_bytes": sum(
            int(row["peak_rss_bytes"]) for row in records
        ),
        "checkpoint_file_bytes": total_bytes,
        "checkpoint_evaluation_count": sum(
            int(row["checkpoint_evaluation_count"]) for row in records
        ),
        "checkpoint_record_count": sum(
            int(row["checkpoint_record_count"]) for row in passed
        ),
        "production_checkpoint_writer_used": all(
            bool(row["production_checkpoint_writer_used"])
            for row in records
        ),
        "endpoint_sufficient_format": ENDPOINT_SUFFICIENT_FORMAT,
        "automatic_retries": 0,
        "real_effect_values_persisted": False,
        "formal_execution_started": False,
    }
    assert_control_plane_only(aggregate)
    return aggregate, tuple(records)


def formal_schedule_weights() -> tuple[dict[str, Any], ...]:
    """Return exact task/CFE weights from the staged frozen schedule."""

    payload = json.loads(R5_CONTRACT_PATH.read_text(encoding="utf-8"))
    rows = build_corrective_e1e2_formal_schedule(payload)
    weights: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if row.workload_id == "E1_STATIC":
            rate_class = (
                "STATIC_3D"
                if row.problem_index in {13, 14}
                else "STATIC_2D"
            )
        elif "DYNAMIC" in row.workload_id:
            rate_class = "DYNAMIC_2D"
        else:
            rate_class = "ROLLING_3D"
        key = (row.workload_id, row.method_id, rate_class)
        item = weights.setdefault(
            key,
            {
                "workload_id": row.workload_id,
                "method_id": row.method_id,
                "projection_rate_class": rate_class,
                "formal_task_count": 0,
                "formal_cfe": 0,
            },
        )
        item["formal_task_count"] += 1
        item["formal_cfe"] += row.total_cfe
    expected_keys = {profile.rate_key for profile in qualification_profiles()}
    if set(weights) != expected_keys:
        raise E1E2QualificationError(
            "qualification rate classes differ from the staged formal schedule"
        )
    values = tuple(weights[key] for key in sorted(weights))
    if (
        sum(int(item["formal_task_count"]) for item in values) != 5_030
        or sum(int(item["formal_cfe"]) for item in values) != 851_000_000
    ):
        raise E1E2QualificationError(
            "formal schedule weights differ from the frozen totals"
        )
    return values


def r5_memory_limits() -> tuple[int, int]:
    """Return the frozen R5 per-worker and aggregate RSS ceilings."""

    payload = json.loads(R5_CONTRACT_PATH.read_text(encoding="utf-8"))
    try:
        parallelism = payload["resource_budget"]["parallelism"]
        max_worker_gib = float(parallelism["max_worker_peak_rss_gib"])
        max_pool_gib = float(parallelism["max_pool_peak_rss_gib"])
    except (KeyError, TypeError, ValueError) as error:
        raise E1E2QualificationError(
            "R5 memory ceilings are missing or invalid"
        ) from error
    if (
        not math.isfinite(max_worker_gib)
        or not math.isfinite(max_pool_gib)
        or max_worker_gib <= 0.0
        or max_pool_gib <= 0.0
    ):
        raise E1E2QualificationError(
            "R5 memory ceilings must be finite and positive"
        )
    return (
        int(max_worker_gib * (1024**3)),
        int(max_pool_gib * (1024**3)),
    )


def _projection(
    *,
    worker_count: int,
    records: Sequence[Mapping[str, Any]],
    weights: Sequence[Mapping[str, Any]],
    host_memory_bytes: int,
    r5_max_worker_rss_bytes: int,
    r5_max_pool_rss_bytes: int,
) -> dict[str, Any]:
    by_case: dict[
        tuple[str, str, str, str],
        list[Mapping[str, Any]],
    ] = defaultdict(list)
    for row in records:
        if row["status"] == "PASS":
            by_case[
                (
                    str(row["workload_id"]),
                    str(row["method_id"]),
                    str(row["projection_rate_class"]),
                    str(row["representative_case_id"]),
                )
            ].append(row)
    cases_by_rate: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for profile in qualification_profiles():
        cases_by_rate[profile.rate_key].add(
            profile.representative_case_id
        )
    missing: list[str] = []
    rate_rows: list[dict[str, Any]] = []
    projected_serial_seconds = 0.0
    for weight in weights:
        rate_key = (
            str(weight["workload_id"]),
            str(weight["method_id"]),
            str(weight["projection_rate_class"]),
        )
        case_rates: list[dict[str, Any]] = []
        for case_id in sorted(cases_by_rate[rate_key]):
            samples = by_case.get((*rate_key, case_id), [])
            if not samples:
                missing.append(
                    "::".join((*rate_key, case_id))
                )
                continue
            case_rates.append(
                {
                    "representative_case_id": case_id,
                    "sample_count": len(samples),
                    "median_seconds_per_cfe_under_sweep_contention": (
                        statistics.median(
                            float(row["wall_seconds"])
                            / int(row["cfe_consumed"])
                            for row in samples
                        )
                    ),
                }
            )
        if len(case_rates) != len(cases_by_rate[rate_key]):
            continue
        seconds_per_cfe = max(
            float(row["median_seconds_per_cfe_under_sweep_contention"])
            for row in case_rates
        )
        formal_cfe = int(weight["formal_cfe"])
        projected_serial_seconds += seconds_per_cfe * formal_cfe
        rate_rows.append(
            {
                **dict(weight),
                "representative_case_rates": case_rates,
                "conservative_max_case_seconds_per_cfe": seconds_per_cfe,
            }
        )
    if missing:
        projection = {
            "workers": worker_count,
            "status": "UNAVAILABLE_FAILED_OR_MISSING_PATH",
            "missing_paths": missing,
            "formal_launch_authorized": False,
        }
        assert_control_plane_only(projection)
        return projection
    projected_seconds = projected_serial_seconds / worker_count
    projected_hours = projected_seconds / 3_600.0
    measured_max_worker_rss = max(
        int(row["peak_rss_bytes"]) for row in records
    )
    conservative_worker_rss = (
        math.ceil(
            measured_max_worker_rss
            * RSS_SAFETY_FACTOR
            / RSS_ROUNDING_BYTES
        )
        * RSS_ROUNDING_BYTES
    )
    conservative_pool_rss = conservative_worker_rss * worker_count
    host_pool_ceiling = max(
        0,
        int(host_memory_bytes) - SUPERVISOR_MEMORY_RESERVE_BYTES,
    )
    memory_eligible = (
        measured_max_worker_rss > 0
        and conservative_worker_rss <= r5_max_worker_rss_bytes
        and conservative_pool_rss <= r5_max_pool_rss_bytes
        and conservative_pool_rss <= host_pool_ceiling
    )
    if not memory_eligible:
        gate = "NO_GO_R5_OR_HOST_RSS_CEILING"
    elif projected_hours <= 36.0:
        gate = "GO_ELIGIBLE_AFTER_CONTRACT_AND_REQUEST_FREEZE"
    elif projected_hours <= 48.0:
        gate = "HOLD_OPTIMIZE_CONTENTION_AND_RETEST"
    else:
        gate = "NO_GO_PROJECTED_OVER_48_HOURS"
    storage = estimate_e1e2_checkpoint_storage()
    projection = {
        "workers": worker_count,
        "status": "TARGET_HOST_FULL_PATH_ESTIMATE",
        "formal_task_count": 5_030,
        "formal_cfe": 851_000_000,
        "method_cfe_weighted_rates": rate_rows,
        "projected_wall_seconds": projected_seconds,
        "projected_wall_hours": projected_hours,
        "projected_wall_hours_with_25_percent_headroom": (
            projected_hours * 1.25
        ),
        "memory_qualification": {
            "measured_max_worker_peak_rss_bytes": measured_max_worker_rss,
            "rss_safety_factor": RSS_SAFETY_FACTOR,
            "rss_rounding_bytes": RSS_ROUNDING_BYTES,
            "conservative_worker_peak_rss_bytes": (
                conservative_worker_rss
            ),
            "worker_count": worker_count,
            "conservative_pool_peak_rss_bytes": conservative_pool_rss,
            "r5_max_worker_peak_rss_bytes": r5_max_worker_rss_bytes,
            "r5_max_pool_peak_rss_bytes": r5_max_pool_rss_bytes,
            "host_memory_bytes": int(host_memory_bytes),
            "supervisor_memory_reserve_bytes": (
                SUPERVISOR_MEMORY_RESERVE_BYTES
            ),
            "host_pool_ceiling_bytes": host_pool_ceiling,
            "eligible": memory_eligible,
        },
        "production_checkpoint_writer_used": True,
        "endpoint_sufficient_format": ENDPOINT_SUFFICIENT_FORMAT,
        "formal_checkpoint_storage_conservative_upper_bound_bytes": (
            storage.conservative_total_upper_bound_bytes
        ),
        "formal_checkpoint_storage_conservative_upper_bound_gib": (
            storage.conservative_total_upper_bound_gib
        ),
        "storage_estimate_is_frozen_schedule_upper_bound": True,
        "decision_classification": gate,
        "formal_launch_authorized": False,
        "authorization_blocker": (
            "QUALIFICATION_MUST_BE_REVIEWED_AND_BOUND_IN_A_NEW_ONE_TIME_REQUEST"
        ),
    }
    assert_control_plane_only(projection)
    return projection


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def code_identity(*, allow_dirty: bool) -> dict[str, Any]:
    """Bind the qualification to Git; clean is mandatory by default."""

    try:
        commit = _git("rev-parse", "HEAD")
        tree = _git("rev-parse", "HEAD^{tree}")
        status = _git("status", "--porcelain", "--untracked-files=all")
    except (OSError, subprocess.CalledProcessError) as error:
        raise E1E2QualificationError(
            "qualification requires an identifiable Git worktree"
        ) from error
    clean = status == ""
    if not clean and not allow_dirty:
        raise E1E2QualificationError(
            "qualification requires a clean committed Git identity; "
            "use --allow-dirty only for a nonqualifying smoke run"
        )
    return {
        "git_commit": commit,
        "git_tree": tree,
        "worktree_clean": clean,
        "qualification_source_sha256": sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }


def validate_output_root(output_root: Path) -> None:
    requested = Path(output_root)
    if not requested.is_absolute():
        raise E1E2QualificationError(
            "qualification output root must be an absolute path"
        )
    resolved = requested.resolve()
    if not resolved.name.startswith("r8c-e1e2-qualification-"):
        raise E1E2QualificationError(
            "output root name must start with 'r8c-e1e2-qualification-'"
        )
    if resolved.exists():
        raise E1E2QualificationError(
            "qualification output root already exists; resume and retry "
            "are forbidden"
        )
    project_root = PROJECT_ROOT.resolve()
    if resolved.is_relative_to(project_root):
        raise E1E2QualificationError(
            "qualification output root must be outside the source worktree"
        )
    if not resolved.parent.is_dir():
        raise E1E2QualificationError(
            "qualification output parent must already exist"
        )


def _is_target_qualifying_design(
    *,
    worker_counts: tuple[int, ...],
    repetitions: int,
    static_cfe_per_event: int,
    dynamic_cfe_per_event: int,
    rolling_cfe_per_event: int,
    dynamic_events: int,
    smoke: bool,
    allow_dirty: bool,
    worktree_clean: bool,
    target_environment_match: bool,
    failed_count: int,
) -> bool:
    """Return whether a sweep used the complete frozen qualification design."""

    return (
        not smoke
        and not allow_dirty
        and worktree_clean
        and target_environment_match
        and worker_counts == DEFAULT_WORKERS
        and repetitions == DEFAULT_REPETITIONS
        and static_cfe_per_event == STATIC_CFE_PER_EVENT
        and dynamic_cfe_per_event == DYNAMIC_CFE_PER_EVENT
        and rolling_cfe_per_event == ROLLING_CFE_PER_EVENT
        and dynamic_events == DEFAULT_DYNAMIC_EVENTS
        and failed_count == 0
    )


def run_e1e2_qualification(
    *,
    output_root: Path,
    worker_counts: Sequence[int] = DEFAULT_WORKERS,
    repetitions: int = DEFAULT_REPETITIONS,
    cfe_per_event: int | None = None,
    static_cfe_per_event: int = STATIC_CFE_PER_EVENT,
    dynamic_cfe_per_event: int = DYNAMIC_CFE_PER_EVENT,
    rolling_cfe_per_event: int = ROLLING_CFE_PER_EVENT,
    dynamic_events: int = DEFAULT_DYNAMIC_EVENTS,
    allow_dirty: bool = False,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run one isolated target-host sweep and persist control artifacts."""

    validate_output_root(output_root)
    counts = tuple(int(value) for value in worker_counts)
    if (
        not counts
        or any(value < 1 for value in counts)
        or len(counts) != len(set(counts))
    ):
        raise E1E2QualificationError(
            "worker counts must be positive and unique"
        )
    if cfe_per_event is not None and not smoke:
        raise E1E2QualificationError(
            "uniform cfe_per_event is allowed only for a nonqualifying smoke "
            "run; target qualification uses workload-specific full budgets"
        )
    if cfe_per_event is not None:
        static_budget = int(cfe_per_event)
        dynamic_budget = int(cfe_per_event)
        rolling_budget = int(cfe_per_event)
    else:
        static_budget = int(static_cfe_per_event)
        dynamic_budget = int(dynamic_cfe_per_event)
        rolling_budget = int(rolling_cfe_per_event)
    for name, value in THREAD_ENVIRONMENT.items():
        os.environ[name] = value
    runtime_environment = runtime_environment_lock_evidence()
    identity = code_identity(allow_dirty=allow_dirty)
    profiles = qualification_profiles(dynamic_events)
    workload_method_path_count = len({profile.key for profile in profiles})
    workload_method_case_binding_count = len(
        {profile.case_key for profile in profiles}
    )
    unique_representative_case_count = len(
        {profile.representative_case_id for profile in profiles}
    )
    formal_projection_rate_class_count = len(
        {profile.rate_key for profile in profiles}
    )
    weights = formal_schedule_weights()
    r5_max_worker_rss_bytes, r5_max_pool_rss_bytes = r5_memory_limits()
    checkpoint_storage = estimate_e1e2_checkpoint_storage()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    host = host_fingerprint()
    host_sha = host_fingerprint_sha256(host)
    marker = {
        "artifact_role": ARTIFACT_ROLE,
        "qualification_id": QUALIFICATION_ID,
        "status": "RUNNING",
        "mode": "SMOKE_NONQUALIFYING" if smoke else "TARGET_QUALIFICATION",
        "output_root": str(output_root),
        "population_size": POPULATION_SIZE,
        "archive_capacity": ARCHIVE_CAPACITY,
        "thread_environment": dict(THREAD_ENVIRONMENT),
        "code_identity": identity,
        "host_fingerprint": host,
        "host_fingerprint_sha256": host_sha,
        "automatic_retries": 0,
        "real_effect_values_persisted": False,
        "formal_execution_started": False,
    }
    assert_control_plane_only(marker)
    (output_root / "qualification_status.json").write_bytes(
        canonical_json_bytes(marker) + b"\n"
    )

    sweeps: list[dict[str, Any]] = []
    projections: list[dict[str, Any]] = []
    all_task_ids: list[str] = []
    for worker_count in counts:
        sweep_dir = output_root / f"workers-{worker_count:02d}"
        (sweep_dir / "tasks").mkdir(parents=True, exist_ok=False)
        tasks = build_tasks(
            output_root=output_root,
            worker_count=worker_count,
            repetitions=repetitions,
            static_cfe_per_event=static_budget,
            dynamic_cfe_per_event=dynamic_budget,
            rolling_cfe_per_event=rolling_budget,
            dynamic_events=dynamic_events,
        )
        all_task_ids.extend(task.task_id for task in tasks)
        aggregate, records = _run_sweep(
            worker_count=worker_count,
            tasks=tasks,
        )
        projection = _projection(
            worker_count=worker_count,
            records=records,
            weights=weights,
            host_memory_bytes=int(host["memory_bytes"]),
            r5_max_worker_rss_bytes=r5_max_worker_rss_bytes,
            r5_max_pool_rss_bytes=r5_max_pool_rss_bytes,
        )
        (sweep_dir / "task_control_metrics.jsonl").write_bytes(
            b"".join(
                canonical_json_bytes(row) + b"\n" for row in records
            )
        )
        (sweep_dir / "sweep_control_summary.json").write_bytes(
            canonical_json_bytes(aggregate) + b"\n"
        )
        sweeps.append(aggregate)
        projections.append(projection)
    if len(all_task_ids) != len(set(all_task_ids)):
        raise E1E2QualificationError(
            "task IDs overlap between worker sweeps"
        )
    memory_eligible_projections = [
        row
        for row in projections
        if row.get("status") == "TARGET_HOST_FULL_PATH_ESTIMATE"
        and isinstance(row.get("memory_qualification"), Mapping)
        and row["memory_qualification"].get("eligible") is True
    ]
    recommended_projection = (
        min(
            memory_eligible_projections,
            key=lambda row: (
                float(row["projected_wall_hours"]),
                int(row["workers"]),
            ),
        )
        if memory_eligible_projections
        else None
    )
    worker_recommendation = {
        "status": (
            "MEMORY_ELIGIBLE_THROUGHPUT_OPTIMUM_IDENTIFIED"
            if recommended_projection is not None
            else "NO_MEMORY_ELIGIBLE_WORKER_COUNT"
        ),
        "selection_rule": (
            "MIN_PROJECTED_WALL_HOURS_AMONG_R5_AND_HOST_RSS_ELIGIBLE_"
            "MEASURED_WORKER_COUNTS;_LOWER_WORKER_COUNT_BREAKS_EXACT_TIES"
        ),
        "measured_worker_counts": list(counts),
        "memory_eligible_worker_counts": [
            int(row["workers"]) for row in memory_eligible_projections
        ],
        "recommended_worker_count": (
            None
            if recommended_projection is None
            else int(recommended_projection["workers"])
        ),
        "recommended_projected_wall_hours": (
            None
            if recommended_projection is None
            else float(recommended_projection["projected_wall_hours"])
        ),
        "recommended_projected_wall_hours_with_25_percent_headroom": (
            None
            if recommended_projection is None
            else float(
                recommended_projection[
                    "projected_wall_hours_with_25_percent_headroom"
                ]
            )
        ),
        "recommended_decision_classification": (
            None
            if recommended_projection is None
            else recommended_projection["decision_classification"]
        ),
        "formal_launch_authorized": False,
    }
    assert_control_plane_only(worker_recommendation)
    failed_count = sum(int(row["failed_task_count"]) for row in sweeps)
    target_qualifying = _is_target_qualifying_design(
        worker_counts=counts,
        repetitions=repetitions,
        static_cfe_per_event=static_budget,
        dynamic_cfe_per_event=dynamic_budget,
        rolling_cfe_per_event=rolling_budget,
        dynamic_events=dynamic_events,
        smoke=smoke,
        allow_dirty=allow_dirty,
        worktree_clean=bool(identity["worktree_clean"]),
        target_environment_match=bool(
            runtime_environment["target_environment_match"]
        ),
        failed_count=failed_count,
    )
    if failed_count:
        status = "FAILED_NO_RETRY"
    elif target_qualifying:
        status = "PASS_PENDING_REVIEW_AND_ONE_TIME_REQUEST_FREEZE"
    else:
        status = "PASS_NONQUALIFYING_DIAGNOSTIC"
    report = {
        "artifact_role": ARTIFACT_ROLE,
        "qualification_id": QUALIFICATION_ID,
        "status": status,
        "mode": "SMOKE_NONQUALIFYING" if smoke else "TARGET_QUALIFICATION",
        "output_root": str(output_root),
        "code_identity": identity,
        "host_fingerprint": host,
        "host_fingerprint_sha256": host_sha,
        "runtime_environment_lock": runtime_environment,
        "runtime_contract": {
            "population_size": POPULATION_SIZE,
            "archive_capacity": ARCHIVE_CAPACITY,
            "single_thread_per_worker": True,
            "thread_environment": dict(THREAD_ENVIRONMENT),
            "workload_blocks_covered": list(
                FORMAL_METHODS_BY_WORKLOAD
            ),
            "workload_method_paths_covered": workload_method_path_count,
            "workload_method_case_bindings_covered": (
                workload_method_case_binding_count
            ),
            "unique_representative_benchmark_cases_covered": (
                unique_representative_case_count
            ),
            "formal_projection_rate_classes_covered": (
                formal_projection_rate_class_count
            ),
            "cdf_operational_suite_id": CDF_OPERATIONAL_SUITE_ID,
            "cdf_operational_authority_id": CDF_OPERATIONAL_AUTHORITY_ID,
            "cdf_operational_authority_amendment_id": (
                CDF_OPERATIONAL_AUTHORITY_AMENDMENT_ID
            ),
            "cdf9_undefined_domain_policy": (
                "CHARGED_TYPED_CDFDomainUndefinedError_TO_"
                "REJECT_NUMERICAL_NO_EXTENSION"
            ),
            "dynamic_event_ids_covered": list(range(dynamic_events)),
            "cdf9_max_domain_stress_event_id": 5,
            "cdf9_max_domain_stress_event_exercised": dynamic_events >= 6,
            "representative_problem_policy": (
                "STATIC_2D_LIRCMOP1_AND_12__STATIC_3D_LIRCMOP14__"
                "DYNAMIC_CDF1_9_13_15__ROLLING_KNOWN_ANSWER"
            ),
            "qualification_cfe_policy": (
                "FULL_PATH_STATIC_50000_CFE;_DYNAMIC_AND_ROLLING_5000_"
                "CFE_PER_EVENT_ACROSS_EVENTS_0_TO_5_INCLUDING_CDF9_"
                "MAX_DOMAIN_STRESS"
            ),
            "default_dynamic_events_exercised": DEFAULT_DYNAMIC_EVENTS,
            "dt_ramde_authority": "R6_ENGINEERING_REQUEST_ONLY",
            "comparators_use_real_optimize": True,
            "production_checkpoint_writer_used": True,
            "endpoint_sufficient_format": ENDPOINT_SUFFICIENT_FORMAT,
            "checkpoint_points_per_event": CHECKPOINTS_PER_EVENT,
            "checkpoint_payload": (
                "FIXED_SHAPE_SYNTHETIC_PARETO_FRONT_SNAPSHOTS_AND_"
                "SYNTHETIC_EVALUATION_STREAM_COMMITMENTS"
            ),
            "synthetic_payload_derivation": (
                "TASK_LOCAL_MONOTONIC_SEQUENCE_AND_FIELD_INDEX_ONLY"
            ),
            "synthetic_success_failure_stream_exercised": True,
            "formal_checkpoint_storage_conservative_upper_bound_bytes": (
                checkpoint_storage.conservative_total_upper_bound_bytes
            ),
            "formal_checkpoint_storage_conservative_upper_bound_gib": (
                checkpoint_storage.conservative_total_upper_bound_gib
            ),
            "storage_estimate_is_frozen_schedule_upper_bound": True,
            "real_effect_values_persisted": False,
            "formal_execution_started": False,
            "automatic_retries": 0,
        },
        "pilot_design": {
            "worker_counts": list(counts),
            "repetitions": repetitions,
            "static_cfe_per_event": static_budget,
            "dynamic_cfe_per_event": dynamic_budget,
            "rolling_cfe_per_event": rolling_budget,
            "dynamic_events": dynamic_events,
            "profiles": [asdict(profile) for profile in profiles],
            "task_count": len(all_task_ids),
            "all_task_ids_unique": True,
        },
        "formal_schedule_control_weights": list(weights),
        "sweeps": sweeps,
        "e1_e2_wall_projection": {
            "formal_task_count": 5_030,
            "formal_cfe": 851_000_000,
            "weighting_rule": (
                "within every workload/method/rate class take the maximum "
                "of representative-case median wall seconds per CFE under "
                "contention, multiply by exact frozen schedule-class CFE, "
                "sum, then divide by workers"
            ),
            "production_checkpoint_writer_used": True,
            "endpoint_sufficient_format": ENDPOINT_SUFFICIENT_FORMAT,
            "formal_checkpoint_storage_conservative_upper_bound_bytes": (
                checkpoint_storage.conservative_total_upper_bound_bytes
            ),
            "formal_checkpoint_storage_conservative_upper_bound_gib": (
                checkpoint_storage.conservative_total_upper_bound_gib
            ),
            "projections": projections,
        },
        "worker_recommendation": worker_recommendation,
        "target_qualification_complete": target_qualifying,
        "formal_launch_authorized": False,
        "next_gate": (
            "REVIEW_REPORT_THEN_FREEZE_TARGET_CONTRACT_AND_ONE_TIME_REQUEST"
        ),
        "failed_task_count": failed_count,
        "automatic_retries": 0,
        "real_effect_values_persisted": False,
        "formal_execution_started": False,
    }
    assert_control_plane_only(report)
    report_path = output_root / "qualification_report.json"
    report_path.write_bytes(canonical_json_bytes(report) + b"\n")
    final_marker = {
        **marker,
        "status": status,
        "task_count": len(all_task_ids),
        "failed_task_count": failed_count,
        "report_bytes": report_path.stat().st_size,
        "report_sha256": sha256(report_path.read_bytes()).hexdigest(),
        "target_qualification_complete": target_qualifying,
    }
    assert_control_plane_only(final_marker)
    (output_root / "qualification_status.json").write_bytes(
        canonical_json_bytes(final_marker) + b"\n"
    )
    return report
