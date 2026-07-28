"""Deterministic expansion of the frozen R5 E1--E3 run matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from typing import Any, Mapping, Sequence


SCHEDULE_ID = "WGT-V11-R7-FORMAL-SCHEDULE-01"
CORRECTIVE_SCHEDULE_ID = "WGT-V11-R8C-FORMAL-SCHEDULE-01"
CORRECTIVE_E1E2_SCHEDULE_ID = "WGT-V11-R8C-E1E2-FORMAL-SCHEDULE-01"
CORRECTIVE_E3_SCENARIOS = (
    "NOMINAL",
    "PARAMETER_MISMATCH_EVAL_EE_PLUS_10_PERCENT",
    "INFEASIBLE_REQUIRED_DEFICIT_1500_OVER_ACTION_CAP_1000_KCAL_DAY",
)


class FormalScheduleError(ValueError):
    """The supplied upstream contracts cannot produce the frozen matrix."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class FormalSequenceSpec:
    schedule_index: int
    workload_id: str
    unit_id: str
    method_id: str
    replicate_index: int
    master_seed_u64: str | None
    events: int
    cfe_per_event: int
    atomic_steps_per_cfe: int
    timeout_seconds: int
    problem_index: int | None = None
    problem_id: str | None = None
    profile: str | None = None
    rolling_template: str | None = None
    rolling_index: int | None = None
    rolling_seed_u64: str | None = None
    subject_id: str | None = None
    subject_seed_u64: str | None = None
    scenario_id: str | None = None
    reused_full_workload_id: str | None = None
    task_namespace: str = "r8"

    @property
    def task_id(self) -> str:
        payload = asdict(self)
        payload.pop("schedule_index")
        namespace = payload.pop("task_namespace")
        if namespace != "r8":
            payload["task_namespace"] = namespace
        digest = sha256(canonical_json_bytes(payload)).hexdigest()
        return f"{namespace}-{self.schedule_index:04d}-{digest[:16]}"

    @property
    def total_cfe(self) -> int:
        return self.events * self.cfe_per_event

    @property
    def total_atomic_steps(self) -> int:
        return self.total_cfe * self.atomic_steps_per_cfe

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.task_namespace == "r8":
            payload.pop("task_namespace")
        return {
            **payload,
            "task_id": self.task_id,
            "total_cfe": self.total_cfe,
            "total_atomic_steps": self.total_atomic_steps,
        }


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FormalScheduleError(f"{label} must be an object")
    return value


def _append(
    rows: list[FormalSequenceSpec],
    **kwargs: Any,
) -> None:
    rows.append(FormalSequenceSpec(schedule_index=len(rows), **kwargs))


def build_formal_schedule(
    r5_contract: Mapping[str, Any],
) -> list[FormalSequenceSpec]:
    """Expand exactly 7,046 method sequences without running a method."""

    if r5_contract.get("contract_id") != (
        "WGT-V11-R5-ENDPOINT-STATISTICS-SAMPLE-SEED-RESOURCE-01"
    ):
        raise FormalScheduleError("unexpected R5 contract identity")
    design = _require_mapping(r5_contract.get("experiment_design"), "design")
    seeds = _require_mapping(r5_contract.get("seed_contract"), "seeds")
    resources = _require_mapping(
        r5_contract.get("resource_budget"), "resources"
    )
    timeouts = _require_mapping(resources.get("timeouts_seconds"), "timeouts")
    master_seeds = tuple(str(value) for value in seeds["paired_master_seeds_u64"])
    rolling_instances = tuple(seeds["rolling_public_instances"])
    subjects = tuple(seeds["e3_public_subjects"])
    rows: list[FormalSequenceSpec] = []

    static = design["E1_STATIC"]
    for problem_index in range(1, 15):
        unit_id = f"LIRCMOP{problem_index}"
        for replicate in range(static["paired_replicates"]):
            for method in static["methods"]:
                _append(
                    rows,
                    workload_id="E1_STATIC",
                    unit_id=unit_id,
                    method_id=method,
                    replicate_index=replicate,
                    master_seed_u64=master_seeds[replicate],
                    events=static["events"],
                    cfe_per_event=static["cfe_per_event"],
                    atomic_steps_per_cfe=static["atomic_steps_per_cfe"],
                    timeout_seconds=timeouts["per_static_sequence"],
                    problem_index=problem_index,
                    problem_id=unit_id,
                )

    dynamic = design["E1_DYNAMIC"]
    for problem_index in range(1, 16):
        problem_id = f"CDF{problem_index}"
        for profile in dynamic["profiles"]:
            unit_id = f"{problem_id}/{profile}"
            for replicate in range(dynamic["paired_replicates"]):
                for method in dynamic["methods"]:
                    _append(
                        rows,
                        workload_id="E1_DYNAMIC",
                        unit_id=unit_id,
                        method_id=method,
                        replicate_index=replicate,
                        master_seed_u64=master_seeds[replicate],
                        events=dynamic["events"],
                        cfe_per_event=dynamic["cfe_per_event"],
                        atomic_steps_per_cfe=dynamic["atomic_steps_per_cfe"],
                        timeout_seconds=timeouts["per_dynamic_sequence"],
                        problem_index=problem_index,
                        problem_id=problem_id,
                        profile=profile,
                    )

    rolling = design["E1_ROLLING"]
    for instance in rolling_instances:
        template = str(instance["template"])
        rolling_index = int(instance["index"])
        unit_id = f"{template}/{rolling_index}"
        for replicate in range(rolling["paired_replicates"]):
            for method in rolling["methods"]:
                _append(
                    rows,
                    workload_id="E1_ROLLING",
                    unit_id=unit_id,
                    method_id=method,
                    replicate_index=replicate,
                    master_seed_u64=master_seeds[replicate],
                    events=rolling["events"],
                    cfe_per_event=rolling["cfe_per_event"],
                    atomic_steps_per_cfe=rolling["atomic_steps_per_cfe"],
                    timeout_seconds=timeouts["per_rolling_sequence"],
                    rolling_template=template,
                    rolling_index=rolling_index,
                    rolling_seed_u64=str(instance["derived_seed_u64"]),
                )

    e2_dynamic = design["E2_DYNAMIC"]
    for problem_index in range(1, 16):
        problem_id = f"CDF{problem_index}"
        for profile in e2_dynamic["profiles"]:
            unit_id = f"{problem_id}/{profile}"
            for replicate in range(e2_dynamic["paired_replicates"]):
                for method in e2_dynamic["methods"]:
                    if method == "FULL":
                        continue
                    _append(
                        rows,
                        workload_id="E2_DYNAMIC_INCREMENTAL_AFTER_FULL_REUSE",
                        unit_id=unit_id,
                        method_id=method,
                        replicate_index=replicate,
                        master_seed_u64=master_seeds[replicate],
                        events=e2_dynamic["events"],
                        cfe_per_event=e2_dynamic["cfe_per_event"],
                        atomic_steps_per_cfe=e2_dynamic[
                            "atomic_steps_per_cfe"
                        ],
                        timeout_seconds=timeouts["per_dynamic_sequence"],
                        problem_index=problem_index,
                        problem_id=problem_id,
                        profile=profile,
                        reused_full_workload_id="E1_DYNAMIC",
                    )

    e2_rolling = design["E2_ROLLING"]
    for instance in rolling_instances:
        template = str(instance["template"])
        rolling_index = int(instance["index"])
        unit_id = f"{template}/{rolling_index}"
        for replicate in range(e2_rolling["paired_replicates"]):
            for method in e2_rolling["methods"]:
                if method == "FULL":
                    continue
                _append(
                    rows,
                    workload_id="E2_ROLLING_INCREMENTAL_AFTER_FULL_REUSE",
                    unit_id=unit_id,
                    method_id=method,
                    replicate_index=replicate,
                    master_seed_u64=master_seeds[replicate],
                    events=e2_rolling["events"],
                    cfe_per_event=e2_rolling["cfe_per_event"],
                    atomic_steps_per_cfe=e2_rolling[
                        "atomic_steps_per_cfe"
                    ],
                    timeout_seconds=timeouts["per_rolling_sequence"],
                    rolling_template=template,
                    rolling_index=rolling_index,
                    rolling_seed_u64=str(instance["derived_seed_u64"]),
                    reused_full_workload_id="E1_ROLLING",
                )

    e3 = design["E3"]
    stochastic_methods = tuple(e3["methods"][:2])
    deterministic_method = str(e3["methods"][2])
    for subject in subjects:
        subject_id = str(subject["subject_id"])
        subject_seed = str(subject["seed_u64"])
        for scenario in e3["scenarios"]:
            unit_id = f"{subject_id}/{scenario}"
            for replicate in range(e3["paired_replicates"]):
                for method in stochastic_methods:
                    _append(
                        rows,
                        workload_id="E3",
                        unit_id=unit_id,
                        method_id=method,
                        replicate_index=replicate,
                        master_seed_u64=master_seeds[replicate],
                        events=e3["events"],
                        cfe_per_event=e3[
                            "cfe_per_event_stochastic_search"
                        ],
                        atomic_steps_per_cfe=e3["atomic_steps_per_cfe"],
                        timeout_seconds=timeouts["per_e3_sequence"],
                        subject_id=subject_id,
                        subject_seed_u64=subject_seed,
                        scenario_id=str(scenario),
                    )
            _append(
                rows,
                workload_id="E3",
                unit_id=unit_id,
                method_id=deterministic_method,
                replicate_index=0,
                master_seed_u64=master_seeds[0],
                events=e3["events"],
                cfe_per_event=e3["cfe_per_event_deterministic_policy"],
                atomic_steps_per_cfe=e3["atomic_steps_per_cfe"],
                timeout_seconds=timeouts["per_e3_sequence"],
                subject_id=subject_id,
                subject_seed_u64=subject_seed,
                scenario_id=str(scenario),
            )

    expected = {
        item["id"]: item
        for item in r5_contract["workload_budget"]["unique_workloads"]
    }
    observed: dict[str, dict[str, int]] = {}
    for row in rows:
        item = observed.setdefault(
            row.workload_id,
            {"method_sequences": 0, "CFE": 0, "atomic_model_steps": 0},
        )
        item["method_sequences"] += 1
        item["CFE"] += row.total_cfe
        item["atomic_model_steps"] += row.total_atomic_steps
    if observed != {
        key: {
            "method_sequences": value["method_sequences"],
            "CFE": value["CFE"],
            "atomic_model_steps": value["atomic_model_steps"],
        }
        for key, value in expected.items()
    }:
        raise FormalScheduleError("expanded workload differs from R5 budget")
    if len({row.task_id for row in rows}) != len(rows):
        raise FormalScheduleError("formal task IDs are not unique")
    return rows


def build_corrective_formal_schedule(
    r5_contract: Mapping[str, Any],
) -> list[FormalSequenceSpec]:
    """Build E1+E2 plus the prospectively frozen three-scenario E3."""

    allowed = set(CORRECTIVE_E3_SCENARIOS)
    base = build_formal_schedule(r5_contract)
    selected = [
        row
        for row in base
        if row.workload_id != "E3" or row.scenario_id in allowed
    ]
    rows = [
        replace(
            row,
            schedule_index=index,
            task_namespace="r8c",
        )
        for index, row in enumerate(selected)
    ]
    observed = {
        "method_sequences": len(rows),
        "CFE": sum(row.total_cfe for row in rows),
        "atomic_model_steps": sum(
            row.total_atomic_steps for row in rows
        ),
    }
    if observed != {
        "method_sequences": 5702,
        "CFE": 925882496,
        "atomic_model_steps": 2420294976,
    }:
        raise FormalScheduleError(
            "corrective schedule differs from its result-blind freeze"
        )
    e3_scenarios = {
        row.scenario_id for row in rows if row.workload_id == "E3"
    }
    if e3_scenarios != allowed:
        raise FormalScheduleError(
            "corrective E3 scenarios differ from the frozen three"
        )
    if len({row.task_id for row in rows}) != len(rows):
        raise FormalScheduleError(
            "corrective formal task IDs are not unique"
        )
    return rows


def build_corrective_e1e2_formal_schedule(
    r5_contract: Mapping[str, Any],
) -> list[FormalSequenceSpec]:
    """Build the prospectively staged E1+E2 prefix without dispatching E3."""

    full = build_corrective_formal_schedule(r5_contract)
    rows = full[:5030]
    allowed_workloads = {
        "E1_STATIC",
        "E1_DYNAMIC",
        "E1_ROLLING",
        "E2_DYNAMIC_INCREMENTAL_AFTER_FULL_REUSE",
        "E2_ROLLING_INCREMENTAL_AFTER_FULL_REUSE",
    }
    if any(row.workload_id not in allowed_workloads for row in rows):
        raise FormalScheduleError(
            "corrective E1+E2 prefix contains a non-E1/E2 workload"
        )
    if any(row.workload_id != "E3" for row in full[len(rows) :]):
        raise FormalScheduleError(
            "corrective schedule is not the exact E1+E2 prefix followed by E3"
        )
    observed = {
        "method_sequences": len(rows),
        "CFE": sum(row.total_cfe for row in rows),
        "atomic_model_steps": sum(
            row.total_atomic_steps for row in rows
        ),
    }
    if observed != {
        "method_sequences": 5030,
        "CFE": 851000000,
        "atomic_model_steps": 1971000000,
    }:
        raise FormalScheduleError(
            "corrective E1+E2 schedule differs from its staged freeze"
        )
    if any(row.schedule_index != index for index, row in enumerate(rows)):
        raise FormalScheduleError(
            "corrective E1+E2 schedule is not the exact R8C prefix"
        )
    if len({row.task_id for row in rows}) != len(rows):
        raise FormalScheduleError(
            "corrective E1+E2 formal task IDs are not unique"
        )
    return rows


def schedule_commitment(rows: Sequence[FormalSequenceSpec]) -> str:
    payload = b"".join(
        canonical_json_bytes(row.to_dict()) + b"\n" for row in rows
    )
    return sha256(payload).hexdigest()


def build_e2_full_reuse_map(
    rows: Sequence[FormalSequenceSpec],
) -> list[dict[str, Any]]:
    """Bind every E2 unit-replicate to the exact E1 FULL task it reuses."""

    full: dict[tuple[str, str, int], FormalSequenceSpec] = {}
    for row in rows:
        if row.workload_id == "E1_DYNAMIC" and row.method_id == (
            "DT-RAMDE_TS2_FULL"
        ):
            full[("E1_DYNAMIC", row.unit_id, row.replicate_index)] = row
        elif row.workload_id == "E1_ROLLING" and row.method_id == (
            "DT-RAMDE_TS2_FULL"
        ):
            full[("E1_ROLLING", row.unit_id, row.replicate_index)] = row
    reuse: list[dict[str, Any]] = []
    for source_workload in ("E1_DYNAMIC", "E1_ROLLING"):
        target_workload = source_workload.replace("E1_", "E2_")
        keys = sorted(
            (
                key
                for key in full
                if key[0] == source_workload
            ),
            key=lambda key: (key[1], key[2]),
        )
        for key in keys:
            row = full[key]
            reuse.append(
                {
                    "e2_workload_id": target_workload,
                    "unit_id": row.unit_id,
                    "replicate_index": row.replicate_index,
                    "reused_e1_workload_id": source_workload,
                    "reused_method_id": "DT-RAMDE_TS2_FULL",
                    "reused_task_id": row.task_id,
                    "reused_schedule_index": row.schedule_index,
                }
            )
    if len(reuse) != 310:
        raise FormalScheduleError("E2 FULL reuse map must contain 310 rows")
    if len({row["reused_task_id"] for row in reuse}) != len(reuse):
        raise FormalScheduleError("E2 FULL reuse task IDs must be unique")
    return reuse


def e2_full_reuse_commitment(rows: Sequence[FormalSequenceSpec]) -> str:
    payload = b"".join(
        canonical_json_bytes(item) + b"\n"
        for item in build_e2_full_reuse_map(rows)
    )
    return sha256(payload).hexdigest()
