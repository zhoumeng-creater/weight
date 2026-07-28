"""PRE-R10 supporting descriptive audit of frozen R9 E1/E2 exports.

This module intentionally does not create new confirmatory hypotheses,
confidence intervals, sign-flip tests, Holm families, or C4 decisions.  It
reports deterministic counts, paired macro effects, missing-observation bounds,
and multiple explicitly labelled cost available sets.
"""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
import io
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence

from .r9_inference import (
    RegisteredHypothesis,
    canonical_json_bytes,
    file_sha256,
    registered_hypotheses,
)


IMPLEMENTATION_ID = (
    "WGT-V11-R9-SUPPORTING-DESCRIPTIVE-v1.0.1-"
    "result-aware-authorized-pre-R10"
)
ANALYSIS_STATUS = (
    "PRE_R10_SUPPORTING_DESCRIPTIVE_RESULT_AWARE__"
    "NO_NEW_CONFIRMATORY_INFERENCE"
)
AUTHORIZATION_TOKEN = (
    "PRE_R10_FAST_PUBLICATION_ROUTE_AUTHORIZED_NO_R10"
)
RAW_MANIFEST_SHA256 = (
    "33ab590adf809ca2b1f87c1ef225a18d43f50dbc40d7f3c2e2da7a379b1768d3"
)
R5_CONTRACT_SHA256 = (
    "4e2dd0a0f4a97b57d71dd13eb60aa8a3c3eb34f0708aae609d50a31d155f6554"
)
R9_EXPORT_MANIFEST_SHA256 = (
    "9b9761360294b6194aea05d09504223c49fafee26f9e343e5fd7e5667d0b9e94"
)
FAILURE_MARGIN = 0.02
HARD_VIOLATION_MARGIN = 0.01
WALL_RATIO_MARGIN = 1.5
RSS_RATIO_MARGIN = 2.0
FLOAT_TOLERANCE = 1e-12

FAILURE_COST_FIELDS = (
    "task_id",
    "schedule_index",
    "workload_id",
    "unit_id",
    "method_id",
    "replicate_index",
    "task_status",
    "outcome_class",
    "terminal_failure_event_count",
    "terminal_code_counts_json",
    "evaluation_failure_count",
    "scheduled_cfe",
    "charged_cfe",
    "unconsumed_cfe",
    "scheduled_atomic_model_steps",
    "charged_atomic_model_steps",
    "charged_work_exact",
    "event_summary_trailing_fragment_present",
    "event_summary_trailing_fragment_bytes",
    "event_summary_trailing_fragment_sha256",
    "wall_seconds",
    "cpu_seconds",
    "peak_rss_bytes",
    "output_bytes",
    "automatic_retries",
)
HARD_VIOLATION_FIELDS = (
    "task_id",
    "schedule_index",
    "workload_id",
    "unit_id",
    "method_id",
    "replicate_index",
    "task_status",
    "endpoint_status",
    "post_execution_hard_violation_rate",
    "hard_violation_event_count",
    "executed_event_count",
    "execution_observation_available_event_count",
    "execution_observation_missing_event_count",
    "durably_completed_event_count",
)
PRIMARY_TERMINAL_CODES = (
    "ACCEPTED",
    "REJECT_SAFETY_FILTER",
    "REJECT_BUDGET_NO_FEASIBLE",
    "REJECT_NUMERICAL",
)
COST_METRICS = (
    "wall_seconds",
    "cpu_seconds",
    "peak_rss_bytes",
    "output_bytes",
    "charged_cfe",
    "charged_atomic_model_steps",
)
COST_AVAILABLE_SETS = (
    "ALL_COMPLETED_TASK_PAIRS",
    "EQUAL_CHARGED_WORK_TASK_PAIRS",
    "BOTH_ALL_EVENTS_ACCEPTED_TASK_PAIRS",
)


class R9SupportingError(RuntimeError):
    """A PRE-R10 supporting-audit binding or invariant failed."""


@dataclass(frozen=True)
class OutcomeRow:
    """One authenticated task-level failure and cost row."""

    task_id: str
    schedule_index: int
    workload_id: str
    unit_id: str
    method_id: str
    replicate_index: int
    task_status: str
    outcome_class: str
    terminal_counts: tuple[tuple[str, int], ...]
    event_count: int
    failure_count: int
    evaluation_failure_count: int
    scheduled_cfe: int
    charged_cfe: int
    unconsumed_cfe: int
    scheduled_atomic_model_steps: int
    charged_atomic_model_steps: int
    charged_work_exact: bool
    wall_seconds: float
    cpu_seconds: float
    peak_rss_bytes: int
    output_bytes: int
    automatic_retries: int

    @property
    def pair_key(self) -> tuple[str, int]:
        return (self.unit_id, self.replicate_index)

    @property
    def failure_rate(self) -> float:
        return self.failure_count / self.event_count

    @property
    def all_events_accepted(self) -> bool:
        return self.failure_count == 0

    def terminal_rate(self, code: str) -> float:
        counts = dict(self.terminal_counts)
        return counts.get(code, 0) / self.event_count


@dataclass(frozen=True)
class HardViolationRow:
    """One task-level execution-observation availability row."""

    task_id: str
    workload_id: str
    unit_id: str
    method_id: str
    replicate_index: int
    task_status: str
    endpoint_status: str
    hard_count: int
    executed_count: int
    available_count: int
    missing_count: int
    durable_count: int
    reported_rate: float | None

    @property
    def pair_key(self) -> tuple[str, int]:
        return (self.unit_id, self.replicate_index)

    @property
    def fas_lower(self) -> float | None:
        if self.executed_count == 0:
            return None
        return self.hard_count / self.executed_count

    @property
    def fas_upper(self) -> float | None:
        if self.executed_count == 0:
            return None
        return (
            self.hard_count + self.missing_count
        ) / self.executed_count

    @property
    def observed_rate(self) -> float | None:
        if self.available_count == 0:
            return None
        return self.hard_count / self.available_count


@dataclass(frozen=True)
class Comparison:
    """One frozen R5 proposed/comparator pairing."""

    index: int
    comparison_id: str
    analysis_workload_id: str
    proposed_workload_id: str
    proposed_method_id: str
    comparator_workload_id: str
    comparator_method_id: str
    unit_rule: str
    expected_clusters: int
    expected_pairs: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise R9SupportingError(message)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise R9SupportingError(f"cannot read valid JSON: {path}") from error


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left.resolve())) == os.path.normcase(
        str(right.resolve())
    )


def _is_within(candidate: Path, parent: Path) -> bool:
    candidate_text = os.path.normcase(str(candidate.resolve()))
    parent_text = os.path.normcase(str(parent.resolve()))
    try:
        return os.path.commonpath((candidate_text, parent_text)) == parent_text
    except ValueError:
        return False


def _parse_int(
    value: str,
    *,
    label: str,
    allow_empty: bool = False,
) -> int | None:
    if allow_empty and value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise R9SupportingError(f"{label} is not an integer") from error
    _require(parsed >= 0, f"{label} is negative")
    return parsed


def _parse_float(
    value: str,
    *,
    label: str,
    allow_empty: bool = False,
) -> float | None:
    if allow_empty and value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise R9SupportingError(f"{label} is not numeric") from error
    _require(math.isfinite(parsed), f"{label} is not finite")
    _require(parsed >= 0.0, f"{label} is negative")
    return parsed


def _parse_bool(value: str, *, label: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise R9SupportingError(f"{label} is not a canonical boolean")


def _csv_rows(path: Path, expected_fields: Sequence[str]) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            _require(
                tuple(reader.fieldnames or ()) == tuple(expected_fields),
                f"{path.name} schema drifted",
            )
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise R9SupportingError(f"cannot read CSV: {path}") from error
    _require(rows, f"{path.name} is empty")
    return rows


def _load_outcomes(input_root: Path) -> tuple[OutcomeRow, ...]:
    rows = _csv_rows(input_root / "failure_cost.csv", FAILURE_COST_FIELDS)
    outcomes: list[OutcomeRow] = []
    task_ids: set[str] = set()
    for raw in rows:
        task_id = raw["task_id"]
        _require(task_id not in task_ids, "failure/cost task ID duplicated")
        task_ids.add(task_id)
        try:
            decoded = json.loads(raw["terminal_code_counts_json"])
        except json.JSONDecodeError as error:
            raise R9SupportingError(
                f"terminal counts are invalid for {task_id}"
            ) from error
        _require(
            isinstance(decoded, dict) and decoded,
            f"terminal counts are empty for {task_id}",
        )
        terminal_counts: list[tuple[str, int]] = []
        for code, value in sorted(decoded.items()):
            _require(
                isinstance(code, str) and code,
                f"terminal code is invalid for {task_id}",
            )
            _require(
                type(value) is int and value >= 0,
                f"terminal count is invalid for {task_id}",
            )
            terminal_counts.append((code, value))
        event_count = sum(value for _, value in terminal_counts)
        failure_count = sum(
            value
            for code, value in terminal_counts
            if code != "ACCEPTED"
        )
        declared_failure = _parse_int(
            raw["terminal_failure_event_count"],
            label=f"{task_id} terminal failure count",
        )
        _require(
            event_count > 0 and failure_count == declared_failure,
            f"terminal totals drifted for {task_id}",
        )
        scheduled_cfe = _parse_int(
            raw["scheduled_cfe"],
            label=f"{task_id} scheduled CFE",
        )
        charged_cfe = _parse_int(
            raw["charged_cfe"],
            label=f"{task_id} charged CFE",
        )
        unconsumed_cfe = _parse_int(
            raw["unconsumed_cfe"],
            label=f"{task_id} unconsumed CFE",
        )
        scheduled_atomic = _parse_int(
            raw["scheduled_atomic_model_steps"],
            label=f"{task_id} scheduled atomic steps",
        )
        charged_atomic = _parse_int(
            raw["charged_atomic_model_steps"],
            label=f"{task_id} charged atomic steps",
        )
        assert scheduled_cfe is not None
        assert charged_cfe is not None
        assert unconsumed_cfe is not None
        assert scheduled_atomic is not None
        assert charged_atomic is not None
        _require(
            charged_cfe + unconsumed_cfe == scheduled_cfe,
            f"charged CFE accounting drifted for {task_id}",
        )
        _require(
            charged_atomic <= scheduled_atomic,
            f"charged atomic work exceeds schedule for {task_id}",
        )
        wall = _parse_float(
            raw["wall_seconds"],
            label=f"{task_id} wall seconds",
        )
        cpu = _parse_float(
            raw["cpu_seconds"],
            label=f"{task_id} CPU seconds",
        )
        rss = _parse_int(
            raw["peak_rss_bytes"],
            label=f"{task_id} peak RSS",
        )
        output = _parse_int(
            raw["output_bytes"],
            label=f"{task_id} output bytes",
        )
        eval_fail = _parse_int(
            raw["evaluation_failure_count"],
            label=f"{task_id} evaluation failure count",
        )
        retries = _parse_int(
            raw["automatic_retries"],
            label=f"{task_id} automatic retries",
        )
        assert wall is not None
        assert cpu is not None
        assert rss is not None
        assert output is not None
        assert eval_fail is not None
        assert retries is not None
        _require(raw["task_status"] == "COMPLETE", "task is not COMPLETE")
        _require(retries == 0, "automatic retry was observed")
        _require(
            raw["event_summary_trailing_fragment_present"] == "False",
            "event summary trailing fragment is present",
        )
        outcomes.append(
            OutcomeRow(
                task_id=task_id,
                schedule_index=int(raw["schedule_index"]),
                workload_id=raw["workload_id"],
                unit_id=raw["unit_id"],
                method_id=raw["method_id"],
                replicate_index=int(raw["replicate_index"]),
                task_status=raw["task_status"],
                outcome_class=raw["outcome_class"],
                terminal_counts=tuple(terminal_counts),
                event_count=event_count,
                failure_count=failure_count,
                evaluation_failure_count=eval_fail,
                scheduled_cfe=scheduled_cfe,
                charged_cfe=charged_cfe,
                unconsumed_cfe=unconsumed_cfe,
                scheduled_atomic_model_steps=scheduled_atomic,
                charged_atomic_model_steps=charged_atomic,
                charged_work_exact=_parse_bool(
                    raw["charged_work_exact"],
                    label=f"{task_id} charged work exactness",
                ),
                wall_seconds=wall,
                cpu_seconds=cpu,
                peak_rss_bytes=rss,
                output_bytes=output,
                automatic_retries=retries,
            )
        )
    _require(len(outcomes) == 5_030, "failure/cost row count drifted")
    _require(
        sorted(row.schedule_index for row in outcomes)
        == list(range(5_030)),
        "schedule indexes are not complete and consecutive",
    )
    return tuple(outcomes)


def _load_hard_rows(input_root: Path) -> tuple[HardViolationRow, ...]:
    rows = _csv_rows(
        input_root / "post_execution_hard_violation.csv",
        HARD_VIOLATION_FIELDS,
    )
    results: list[HardViolationRow] = []
    task_ids: set[str] = set()
    for raw in rows:
        task_id = raw["task_id"]
        _require(task_id not in task_ids, "hard-violation task ID duplicated")
        task_ids.add(task_id)
        hard = _parse_int(
            raw["hard_violation_event_count"],
            label=f"{task_id} hard violations",
        )
        executed = _parse_int(
            raw["executed_event_count"],
            label=f"{task_id} executed events",
        )
        available = _parse_int(
            raw["execution_observation_available_event_count"],
            label=f"{task_id} available observations",
        )
        missing = _parse_int(
            raw["execution_observation_missing_event_count"],
            label=f"{task_id} missing observations",
        )
        durable = _parse_int(
            raw["durably_completed_event_count"],
            label=f"{task_id} durable events",
        )
        reported = _parse_float(
            raw["post_execution_hard_violation_rate"],
            label=f"{task_id} hard-violation rate",
            allow_empty=True,
        )
        assert hard is not None
        assert executed is not None
        assert available is not None
        assert missing is not None
        assert durable is not None
        _require(
            available + missing == executed,
            f"execution observation accounting drifted for {task_id}",
        )
        _require(hard <= available, f"hard count exceeds observations for {task_id}")
        status = raw["endpoint_status"]
        if executed == 0:
            _require(
                status == "NOT_COMPUTED_NO_EXECUTED_EVENTS"
                and reported is None,
                f"zero-execution status drifted for {task_id}",
            )
        elif missing > 0:
            _require(
                status == "NOT_COMPUTED_MISSING_EXECUTION_OBSERVATION"
                and reported is None,
                f"missing-observation status drifted for {task_id}",
            )
        else:
            _require(
                status == "INCLUDED" and reported is not None,
                f"included hard endpoint drifted for {task_id}",
            )
            _require(
                abs(reported - hard / executed) <= FLOAT_TOLERANCE,
                f"reported hard rate drifted for {task_id}",
            )
        results.append(
            HardViolationRow(
                task_id=task_id,
                workload_id=raw["workload_id"],
                unit_id=raw["unit_id"],
                method_id=raw["method_id"],
                replicate_index=int(raw["replicate_index"]),
                task_status=raw["task_status"],
                endpoint_status=status,
                hard_count=hard,
                executed_count=executed,
                available_count=available,
                missing_count=missing,
                durable_count=durable,
                reported_rate=reported,
            )
        )
    _require(len(results) == 5_030, "hard-violation row count drifted")
    return tuple(results)


def _comparison_id(hypothesis: RegisteredHypothesis) -> str:
    return (
        f"{hypothesis.analysis_workload_id}__"
        f"{hypothesis.proposed_method_id}__VS__"
        f"{hypothesis.comparator_method_id}"
    )


def _comparisons(r5: Mapping[str, Any]) -> tuple[Comparison, ...]:
    comparisons = tuple(
        Comparison(
            index=row.hypothesis_index,
            comparison_id=_comparison_id(row),
            analysis_workload_id=row.analysis_workload_id,
            proposed_workload_id=row.proposed_workload_id,
            proposed_method_id=row.proposed_method_id,
            comparator_workload_id=row.comparator_workload_id,
            comparator_method_id=row.comparator_method_id,
            unit_rule=row.top_level_unit_rule,
            expected_clusters=row.expected_top_level_clusters,
            expected_pairs=row.expected_nested_pairs,
        )
        for row in registered_hypotheses(r5)
    )
    _require(len(comparisons) == 30, "comparison count drifted")
    return comparisons


def _cluster_id(unit_id: str, unit_rule: str) -> str:
    if unit_rule == "problem_before_slash":
        return unit_id.split("/", 1)[0]
    _require(unit_rule == "unit_id", f"unknown unit rule: {unit_rule}")
    return unit_id


def _indexed_outcomes(
    rows: Sequence[OutcomeRow],
) -> dict[tuple[str, str, str, int], OutcomeRow]:
    index: dict[tuple[str, str, str, int], OutcomeRow] = {}
    for row in rows:
        key = (
            row.workload_id,
            row.method_id,
            row.unit_id,
            row.replicate_index,
        )
        _require(key not in index, f"outcome pairing key duplicated: {key}")
        index[key] = row
    return index


def _indexed_hard(
    rows: Sequence[HardViolationRow],
) -> dict[tuple[str, str, str, int], HardViolationRow]:
    index: dict[tuple[str, str, str, int], HardViolationRow] = {}
    for row in rows:
        key = (
            row.workload_id,
            row.method_id,
            row.unit_id,
            row.replicate_index,
        )
        _require(key not in index, f"hard pairing key duplicated: {key}")
        index[key] = row
    return index


def _paired_outcomes(
    comparison: Comparison,
    index: Mapping[tuple[str, str, str, int], OutcomeRow],
) -> tuple[tuple[OutcomeRow, OutcomeRow], ...]:
    proposed = {
        (unit, replicate): row
        for (workload, method, unit, replicate), row in index.items()
        if workload == comparison.proposed_workload_id
        and method == comparison.proposed_method_id
    }
    comparator = {
        (unit, replicate): row
        for (workload, method, unit, replicate), row in index.items()
        if workload == comparison.comparator_workload_id
        and method == comparison.comparator_method_id
    }
    _require(
        set(proposed) == set(comparator),
        f"unpaired failure/cost rows for {comparison.comparison_id}",
    )
    _require(
        len(proposed) == comparison.expected_pairs,
        f"pair count drifted for {comparison.comparison_id}",
    )
    pairs = tuple(
        (proposed[key], comparator[key])
        for key in sorted(proposed)
    )
    clusters = {
        _cluster_id(proposed_row.unit_id, comparison.unit_rule)
        for proposed_row, _ in pairs
    }
    _require(
        len(clusters) == comparison.expected_clusters,
        f"cluster count drifted for {comparison.comparison_id}",
    )
    return pairs


def _paired_hard(
    comparison: Comparison,
    index: Mapping[
        tuple[str, str, str, int],
        HardViolationRow,
    ],
) -> tuple[tuple[HardViolationRow, HardViolationRow], ...]:
    proposed = {
        (unit, replicate): row
        for (workload, method, unit, replicate), row in index.items()
        if workload == comparison.proposed_workload_id
        and method == comparison.proposed_method_id
    }
    comparator = {
        (unit, replicate): row
        for (workload, method, unit, replicate), row in index.items()
        if workload == comparison.comparator_workload_id
        and method == comparison.comparator_method_id
    }
    _require(
        set(proposed) == set(comparator),
        f"unpaired hard rows for {comparison.comparison_id}",
    )
    _require(
        len(proposed) == comparison.expected_pairs,
        f"hard pair count drifted for {comparison.comparison_id}",
    )
    return tuple(
        (proposed[key], comparator[key])
        for key in sorted(proposed)
    )


def _float_text(value: float | None) -> str:
    if value is None:
        return ""
    return format(value, ".17g")


def _csv_bytes(
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> bytes:
    target = io.StringIO(newline="")
    writer = csv.DictWriter(
        target,
        fieldnames=fieldnames,
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(rows)
    return target.getvalue().encode("utf-8")


def _csv_data_row_count(payload: bytes) -> int:
    """Count data rows in one deterministic single-line-record CSV payload."""

    _require(payload.endswith(b"\n"), "CSV payload lacks final newline")
    line_count = payload.count(b"\n")
    _require(line_count >= 1, "CSV payload lacks header")
    return line_count - 1


def _mean(values: Sequence[float]) -> float:
    _require(bool(values), "mean requested for empty values")
    return math.fsum(values) / len(values)


def _median(values: Sequence[float]) -> float:
    _require(bool(values), "median requested for empty values")
    return float(statistics.median(values))


def _geometric_mean(values: Sequence[float]) -> float:
    _require(bool(values), "geometric mean requested for empty values")
    _require(all(value > 0.0 for value in values), "ratio is not positive")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def _practical_class(effect: float, threshold: float) -> str:
    if effect >= threshold:
        return "POSITIVE_FAVORS_PROPOSED"
    if effect <= -threshold:
        return "NEGATIVE_FAVORS_COMPARATOR"
    return "SMALL_OR_NULL"


def _method_summary_payload(
    outcomes: Sequence[OutcomeRow],
    hard_rows: Sequence[HardViolationRow],
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    hard_by_task = {row.task_id: row for row in hard_rows}
    _require(
        set(hard_by_task) == {row.task_id for row in outcomes},
        "hard/failure task coverage differs",
    )
    groups: dict[tuple[str, str], list[OutcomeRow]] = defaultdict(list)
    for row in outcomes:
        groups[(row.workload_id, row.method_id)].append(row)
    fields = (
        "workload_id",
        "method_id",
        "task_count",
        "event_count",
        "accepted_event_count",
        "nonaccepted_event_count",
        "failure_rate",
        "reject_safety_filter_count",
        "reject_safety_filter_rate",
        "reject_budget_no_feasible_count",
        "reject_budget_no_feasible_rate",
        "reject_numerical_count",
        "reject_numerical_rate",
        "other_terminal_count",
        "evaluation_failure_count",
        "scheduled_cfe",
        "charged_cfe",
        "unconsumed_cfe",
        "scheduled_atomic_model_steps",
        "charged_atomic_model_steps",
        "wall_seconds_median",
        "cpu_seconds_median",
        "peak_rss_bytes_median",
        "output_bytes_median",
        "executed_event_count",
        "execution_observation_available_count",
        "execution_observation_missing_count",
        "execution_observation_availability_rate",
        "hard_violation_event_count",
        "hard_violation_observed_rate",
        "hard_violation_fas_lower",
        "hard_violation_fas_upper",
        "analysis_status",
    )
    payload: list[dict[str, Any]] = []
    for (workload, method), rows in sorted(groups.items()):
        events = sum(row.event_count for row in rows)
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            for code, value in row.terminal_counts:
                counts[code] += value
        accepted = counts.get("ACCEPTED", 0)
        nonaccepted = events - accepted
        hard = [hard_by_task[row.task_id] for row in rows]
        executed = sum(row.executed_count for row in hard)
        available = sum(row.available_count for row in hard)
        missing = sum(row.missing_count for row in hard)
        hard_count = sum(row.hard_count for row in hard)
        primary_total = sum(
            counts.get(code, 0) for code in PRIMARY_TERMINAL_CODES
        )
        payload.append(
            {
                "workload_id": workload,
                "method_id": method,
                "task_count": len(rows),
                "event_count": events,
                "accepted_event_count": accepted,
                "nonaccepted_event_count": nonaccepted,
                "failure_rate": _float_text(nonaccepted / events),
                "reject_safety_filter_count": counts.get(
                    "REJECT_SAFETY_FILTER",
                    0,
                ),
                "reject_safety_filter_rate": _float_text(
                    counts.get("REJECT_SAFETY_FILTER", 0) / events
                ),
                "reject_budget_no_feasible_count": counts.get(
                    "REJECT_BUDGET_NO_FEASIBLE",
                    0,
                ),
                "reject_budget_no_feasible_rate": _float_text(
                    counts.get("REJECT_BUDGET_NO_FEASIBLE", 0) / events
                ),
                "reject_numerical_count": counts.get(
                    "REJECT_NUMERICAL",
                    0,
                ),
                "reject_numerical_rate": _float_text(
                    counts.get("REJECT_NUMERICAL", 0) / events
                ),
                "other_terminal_count": events - primary_total,
                "evaluation_failure_count": sum(
                    row.evaluation_failure_count for row in rows
                ),
                "scheduled_cfe": sum(row.scheduled_cfe for row in rows),
                "charged_cfe": sum(row.charged_cfe for row in rows),
                "unconsumed_cfe": sum(row.unconsumed_cfe for row in rows),
                "scheduled_atomic_model_steps": sum(
                    row.scheduled_atomic_model_steps for row in rows
                ),
                "charged_atomic_model_steps": sum(
                    row.charged_atomic_model_steps for row in rows
                ),
                "wall_seconds_median": _float_text(
                    _median([row.wall_seconds for row in rows])
                ),
                "cpu_seconds_median": _float_text(
                    _median([row.cpu_seconds for row in rows])
                ),
                "peak_rss_bytes_median": _float_text(
                    _median([float(row.peak_rss_bytes) for row in rows])
                ),
                "output_bytes_median": _float_text(
                    _median([float(row.output_bytes) for row in rows])
                ),
                "executed_event_count": executed,
                "execution_observation_available_count": available,
                "execution_observation_missing_count": missing,
                "execution_observation_availability_rate": _float_text(
                    None if executed == 0 else available / executed
                ),
                "hard_violation_event_count": hard_count,
                "hard_violation_observed_rate": _float_text(
                    None if available == 0 else hard_count / available
                ),
                "hard_violation_fas_lower": _float_text(
                    None if executed == 0 else hard_count / executed
                ),
                "hard_violation_fas_upper": _float_text(
                    None
                    if executed == 0
                    else (hard_count + missing) / executed
                ),
                "analysis_status": ANALYSIS_STATUS,
            }
        )
    return fields, payload


def _failure_payloads(
    comparisons: Sequence[Comparison],
    outcome_index: Mapping[tuple[str, str, str, int], OutcomeRow],
) -> tuple[
    tuple[tuple[str, ...], list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    fields = (
        "comparison_index",
        "comparison_id",
        "analysis_workload_id",
        "proposed_method_id",
        "comparator_method_id",
        "expected_top_level_clusters",
        "available_top_level_clusters",
        "paired_task_count",
        "proposed_failure_rate_macro",
        "comparator_failure_rate_macro",
        "effect_comparator_minus_proposed",
        "practical_margin",
        "practical_effect_class",
        "reject_safety_filter_effect",
        "reject_budget_no_feasible_effect",
        "reject_numerical_effect",
        "other_nonaccepted_effect",
        "confidence_interval",
        "p_value",
        "multiplicity",
        "analysis_status",
    )
    summaries: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    typed_codes = (
        "FAILURE_RATE",
        "REJECT_SAFETY_FILTER",
        "REJECT_BUDGET_NO_FEASIBLE",
        "REJECT_NUMERICAL",
        "OTHER_NONACCEPTED",
    )
    for comparison in comparisons:
        pairs = _paired_outcomes(comparison, outcome_index)
        grouped: dict[
            str,
            list[tuple[OutcomeRow, OutcomeRow]],
        ] = defaultdict(list)
        for proposed, comparator in pairs:
            grouped[
                _cluster_id(proposed.unit_id, comparison.unit_rule)
            ].append((proposed, comparator))
        _require(
            len(grouped) == comparison.expected_clusters,
            "failure clusters drifted",
        )
        cluster_failure_effects: list[float] = []
        cluster_proposed_rates: list[float] = []
        cluster_comparator_rates: list[float] = []
        typed_cluster_effects: dict[str, list[float]] = {
            code: [] for code in typed_codes
        }
        for cluster_id, cluster_pairs in sorted(grouped.items()):
            proposed_failure = _mean(
                [row.failure_rate for row, _ in cluster_pairs]
            )
            comparator_failure = _mean(
                [row.failure_rate for _, row in cluster_pairs]
            )
            cluster_failure_effects.append(
                comparator_failure - proposed_failure
            )
            cluster_proposed_rates.append(proposed_failure)
            cluster_comparator_rates.append(comparator_failure)
            for code in typed_codes:
                if code == "FAILURE_RATE":
                    effect = comparator_failure - proposed_failure
                elif code == "OTHER_NONACCEPTED":
                    known = PRIMARY_TERMINAL_CODES[1:]
                    proposed_other = _mean(
                        [
                            row.failure_rate
                            - sum(row.terminal_rate(item) for item in known)
                            for row, _ in cluster_pairs
                        ]
                    )
                    comparator_other = _mean(
                        [
                            row.failure_rate
                            - sum(row.terminal_rate(item) for item in known)
                            for _, row in cluster_pairs
                        ]
                    )
                    effect = comparator_other - proposed_other
                else:
                    effect = _mean(
                        [
                            comparator.terminal_rate(code)
                            - proposed.terminal_rate(code)
                            for proposed, comparator in cluster_pairs
                        ]
                    )
                typed_cluster_effects[code].append(effect)
                cluster_rows.append(
                    {
                        "comparison_index": comparison.index,
                        "comparison_id": comparison.comparison_id,
                        "analysis_workload_id": (
                            comparison.analysis_workload_id
                        ),
                        "endpoint_id": code,
                        "cluster_id": cluster_id,
                        "paired_task_count": len(cluster_pairs),
                        "cluster_effect": _float_text(effect),
                        "cluster_lower_bound": _float_text(effect),
                        "cluster_upper_bound": _float_text(effect),
                        "effect_direction": (
                            "comparator_rate_minus_proposed_rate"
                        ),
                        "analysis_status": ANALYSIS_STATUS,
                    }
                )
        point = _mean(cluster_failure_effects)
        summaries.append(
            {
                "comparison_index": comparison.index,
                "comparison_id": comparison.comparison_id,
                "analysis_workload_id": comparison.analysis_workload_id,
                "proposed_method_id": comparison.proposed_method_id,
                "comparator_method_id": comparison.comparator_method_id,
                "expected_top_level_clusters": (
                    comparison.expected_clusters
                ),
                "available_top_level_clusters": len(grouped),
                "paired_task_count": len(pairs),
                "proposed_failure_rate_macro": _float_text(
                    _mean(cluster_proposed_rates)
                ),
                "comparator_failure_rate_macro": _float_text(
                    _mean(cluster_comparator_rates)
                ),
                "effect_comparator_minus_proposed": _float_text(point),
                "practical_margin": _float_text(FAILURE_MARGIN),
                "practical_effect_class": _practical_class(
                    point,
                    FAILURE_MARGIN,
                ),
                "reject_safety_filter_effect": _float_text(
                    _mean(typed_cluster_effects["REJECT_SAFETY_FILTER"])
                ),
                "reject_budget_no_feasible_effect": _float_text(
                    _mean(
                        typed_cluster_effects[
                            "REJECT_BUDGET_NO_FEASIBLE"
                        ]
                    )
                ),
                "reject_numerical_effect": _float_text(
                    _mean(typed_cluster_effects["REJECT_NUMERICAL"])
                ),
                "other_nonaccepted_effect": _float_text(
                    _mean(typed_cluster_effects["OTHER_NONACCEPTED"])
                ),
                "confidence_interval": "NOT_GENERATED_PRE_R10",
                "p_value": "NOT_GENERATED_PRE_R10",
                "multiplicity": "NOT_APPLICABLE_SUPPORTING_DESCRIPTIVE",
                "analysis_status": ANALYSIS_STATUS,
            }
        )
    return (fields, summaries), cluster_rows


def _hard_payloads(
    comparisons: Sequence[Comparison],
    hard_index: Mapping[
        tuple[str, str, str, int],
        HardViolationRow,
    ],
) -> tuple[
    tuple[tuple[str, ...], list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    fields = (
        "comparison_index",
        "comparison_id",
        "analysis_workload_id",
        "proposed_method_id",
        "comparator_method_id",
        "paired_task_count",
        "proposed_executed_event_count",
        "comparator_executed_event_count",
        "proposed_observation_availability_rate",
        "comparator_observation_availability_rate",
        "proposed_observed_hard_rate",
        "comparator_observed_hard_rate",
        "complete_task_pair_count",
        "complete_cluster_count",
        "complete_pair_effect_comparator_minus_proposed",
        "fas_effect_lower_bound",
        "fas_effect_upper_bound",
        "practical_margin",
        "complete_method_comparison_allowed",
        "analysis_interpretation",
        "analysis_status",
    )
    summaries: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        pairs = _paired_hard(comparison, hard_index)
        proposed_executed = sum(row.executed_count for row, _ in pairs)
        comparator_executed = sum(row.executed_count for _, row in pairs)
        if proposed_executed == 0 and comparator_executed == 0:
            summaries.append(
                {
                    "comparison_index": comparison.index,
                    "comparison_id": comparison.comparison_id,
                    "analysis_workload_id": (
                        comparison.analysis_workload_id
                    ),
                    "proposed_method_id": comparison.proposed_method_id,
                    "comparator_method_id": (
                        comparison.comparator_method_id
                    ),
                    "paired_task_count": len(pairs),
                    "proposed_executed_event_count": 0,
                    "comparator_executed_event_count": 0,
                    "proposed_observation_availability_rate": "",
                    "comparator_observation_availability_rate": "",
                    "proposed_observed_hard_rate": "",
                    "comparator_observed_hard_rate": "",
                    "complete_task_pair_count": 0,
                    "complete_cluster_count": 0,
                    "complete_pair_effect_comparator_minus_proposed": "",
                    "fas_effect_lower_bound": "",
                    "fas_effect_upper_bound": "",
                    "practical_margin": _float_text(
                        HARD_VIOLATION_MARGIN
                    ),
                    "complete_method_comparison_allowed": "false",
                    "analysis_interpretation": (
                        "NOT_APPLICABLE_NO_EXECUTED_EVENTS"
                    ),
                    "analysis_status": ANALYSIS_STATUS,
                }
            )
            continue
        _require(
            proposed_executed > 0 and comparator_executed > 0,
            "hard execution support is asymmetric",
        )
        grouped: dict[
            str,
            list[tuple[HardViolationRow, HardViolationRow]],
        ] = defaultdict(list)
        for proposed, comparator in pairs:
            grouped[
                _cluster_id(proposed.unit_id, comparison.unit_rule)
            ].append((proposed, comparator))
        lower_effects: list[float] = []
        upper_effects: list[float] = []
        exact_cluster_effects: list[float] = []
        complete_pair_count = 0
        for cluster_id, cluster_pairs in sorted(grouped.items()):
            pair_lowers: list[float] = []
            pair_uppers: list[float] = []
            pair_exact: list[float] = []
            for proposed, comparator in cluster_pairs:
                assert proposed.fas_lower is not None
                assert proposed.fas_upper is not None
                assert comparator.fas_lower is not None
                assert comparator.fas_upper is not None
                pair_lowers.append(
                    comparator.fas_lower - proposed.fas_upper
                )
                pair_uppers.append(
                    comparator.fas_upper - proposed.fas_lower
                )
                if proposed.missing_count == 0 and comparator.missing_count == 0:
                    pair_exact.append(
                        comparator.fas_lower - proposed.fas_lower
                    )
                    complete_pair_count += 1
            cluster_lower = _mean(pair_lowers)
            cluster_upper = _mean(pair_uppers)
            cluster_exact = (
                None if not pair_exact else _mean(pair_exact)
            )
            lower_effects.append(cluster_lower)
            upper_effects.append(cluster_upper)
            if cluster_exact is not None:
                exact_cluster_effects.append(cluster_exact)
            cluster_rows.append(
                {
                    "comparison_index": comparison.index,
                    "comparison_id": comparison.comparison_id,
                    "analysis_workload_id": (
                        comparison.analysis_workload_id
                    ),
                    "endpoint_id": (
                        "POST_EXECUTION_HARD_VIOLATION_RATE"
                    ),
                    "cluster_id": cluster_id,
                    "paired_task_count": len(cluster_pairs),
                    "cluster_effect": _float_text(cluster_exact),
                    "cluster_lower_bound": _float_text(cluster_lower),
                    "cluster_upper_bound": _float_text(cluster_upper),
                    "effect_direction": (
                        "comparator_rate_minus_proposed_rate"
                    ),
                    "analysis_status": ANALYSIS_STATUS,
                }
            )
        proposed_available = sum(row.available_count for row, _ in pairs)
        comparator_available = sum(row.available_count for _, row in pairs)
        proposed_hard = sum(row.hard_count for row, _ in pairs)
        comparator_hard = sum(row.hard_count for _, row in pairs)
        all_complete = complete_pair_count == len(pairs)
        summaries.append(
            {
                "comparison_index": comparison.index,
                "comparison_id": comparison.comparison_id,
                "analysis_workload_id": comparison.analysis_workload_id,
                "proposed_method_id": comparison.proposed_method_id,
                "comparator_method_id": comparison.comparator_method_id,
                "paired_task_count": len(pairs),
                "proposed_executed_event_count": proposed_executed,
                "comparator_executed_event_count": comparator_executed,
                "proposed_observation_availability_rate": _float_text(
                    proposed_available / proposed_executed
                ),
                "comparator_observation_availability_rate": _float_text(
                    comparator_available / comparator_executed
                ),
                "proposed_observed_hard_rate": _float_text(
                    proposed_hard / proposed_available
                    if proposed_available
                    else None
                ),
                "comparator_observed_hard_rate": _float_text(
                    comparator_hard / comparator_available
                    if comparator_available
                    else None
                ),
                "complete_task_pair_count": complete_pair_count,
                "complete_cluster_count": len(exact_cluster_effects),
                "complete_pair_effect_comparator_minus_proposed": (
                    _float_text(
                        _mean(exact_cluster_effects)
                        if exact_cluster_effects
                        else None
                    )
                ),
                "fas_effect_lower_bound": _float_text(
                    _mean(lower_effects)
                ),
                "fas_effect_upper_bound": _float_text(
                    _mean(upper_effects)
                ),
                "practical_margin": _float_text(
                    HARD_VIOLATION_MARGIN
                ),
                "complete_method_comparison_allowed": str(
                    all_complete
                ).lower(),
                "analysis_interpretation": (
                    "COMPLETE_DESCRIPTIVE"
                    if all_complete
                    else (
                        "METHOD_RELATED_MISSINGNESS__"
                        "COMPLETE_COMPARISON_NOT_ALLOWED"
                    )
                ),
                "analysis_status": ANALYSIS_STATUS,
            }
        )
    return (fields, summaries), cluster_rows


def _metric_value(row: OutcomeRow, metric: str) -> float:
    value = getattr(row, metric)
    _require(
        isinstance(value, int | float) and not isinstance(value, bool),
        f"cost metric {metric} is not numeric",
    )
    return float(value)


def _cost_set_membership(
    proposed: OutcomeRow,
    comparator: OutcomeRow,
) -> dict[str, bool]:
    return {
        "ALL_COMPLETED_TASK_PAIRS": (
            proposed.task_status == comparator.task_status == "COMPLETE"
        ),
        "EQUAL_CHARGED_WORK_TASK_PAIRS": (
            proposed.charged_work_exact
            and comparator.charged_work_exact
            and proposed.charged_cfe == comparator.charged_cfe
            and proposed.charged_atomic_model_steps
            == comparator.charged_atomic_model_steps
        ),
        "BOTH_ALL_EVENTS_ACCEPTED_TASK_PAIRS": (
            proposed.all_events_accepted
            and comparator.all_events_accepted
        ),
    }


def _cost_payloads(
    comparisons: Sequence[Comparison],
    outcome_index: Mapping[tuple[str, str, str, int], OutcomeRow],
) -> tuple[
    tuple[tuple[str, ...], list[dict[str, Any]]],
    tuple[tuple[str, ...], list[dict[str, Any]]],
]:
    summary_fields = (
        "comparison_index",
        "comparison_id",
        "analysis_workload_id",
        "proposed_method_id",
        "comparator_method_id",
        "metric",
        "available_set",
        "expected_task_pairs",
        "available_task_pairs",
        "geometric_mean_ratio_proposed_over_comparator",
        "median_ratio_proposed_over_comparator",
        "minimum_ratio",
        "maximum_ratio",
        "point_margin",
        "point_margin_class",
        "formal_cost_gate_status",
        "analysis_status",
    )
    value_fields = (
        "comparison_index",
        "comparison_id",
        "analysis_workload_id",
        "unit_id",
        "replicate_index",
        "metric",
        "ratio_proposed_over_comparator",
        "all_completed_pair",
        "equal_charged_work_pair",
        "both_all_events_accepted_pair",
        "analysis_status",
    )
    summary_rows: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        pairs = _paired_outcomes(comparison, outcome_index)
        for metric in COST_METRICS:
            metric_values: list[
                tuple[float, dict[str, bool], OutcomeRow]
            ] = []
            for proposed, comparator in pairs:
                denominator = _metric_value(comparator, metric)
                numerator = _metric_value(proposed, metric)
                if denominator <= 0.0 or numerator <= 0.0:
                    continue
                ratio = numerator / denominator
                membership = _cost_set_membership(
                    proposed,
                    comparator,
                )
                metric_values.append((ratio, membership, proposed))
                value_rows.append(
                    {
                        "comparison_index": comparison.index,
                        "comparison_id": comparison.comparison_id,
                        "analysis_workload_id": (
                            comparison.analysis_workload_id
                        ),
                        "unit_id": proposed.unit_id,
                        "replicate_index": proposed.replicate_index,
                        "metric": metric,
                        "ratio_proposed_over_comparator": _float_text(
                            ratio
                        ),
                        "all_completed_pair": str(
                            membership["ALL_COMPLETED_TASK_PAIRS"]
                        ).lower(),
                        "equal_charged_work_pair": str(
                            membership[
                                "EQUAL_CHARGED_WORK_TASK_PAIRS"
                            ]
                        ).lower(),
                        "both_all_events_accepted_pair": str(
                            membership[
                                "BOTH_ALL_EVENTS_ACCEPTED_TASK_PAIRS"
                            ]
                        ).lower(),
                        "analysis_status": ANALYSIS_STATUS,
                    }
                )
            for available_set in COST_AVAILABLE_SETS:
                ratios = [
                    ratio
                    for ratio, membership, _ in metric_values
                    if membership[available_set]
                ]
                margin: float | None
                if metric == "wall_seconds":
                    margin = WALL_RATIO_MARGIN
                elif metric == "peak_rss_bytes":
                    margin = RSS_RATIO_MARGIN
                else:
                    margin = None
                geometric = _geometric_mean(ratios) if ratios else None
                if margin is None or geometric is None:
                    margin_class = "NOT_APPLICABLE"
                elif geometric <= margin:
                    margin_class = "AT_OR_BELOW_POINT_MARGIN"
                else:
                    margin_class = "ABOVE_POINT_MARGIN"
                summary_rows.append(
                    {
                        "comparison_index": comparison.index,
                        "comparison_id": comparison.comparison_id,
                        "analysis_workload_id": (
                            comparison.analysis_workload_id
                        ),
                        "proposed_method_id": (
                            comparison.proposed_method_id
                        ),
                        "comparator_method_id": (
                            comparison.comparator_method_id
                        ),
                        "metric": metric,
                        "available_set": available_set,
                        "expected_task_pairs": len(pairs),
                        "available_task_pairs": len(ratios),
                        "geometric_mean_ratio_proposed_over_comparator": (
                            _float_text(geometric)
                        ),
                        "median_ratio_proposed_over_comparator": (
                            _float_text(_median(ratios) if ratios else None)
                        ),
                        "minimum_ratio": _float_text(
                            min(ratios) if ratios else None
                        ),
                        "maximum_ratio": _float_text(
                            max(ratios) if ratios else None
                        ),
                        "point_margin": _float_text(margin),
                        "point_margin_class": margin_class,
                        "formal_cost_gate_status": (
                            "NOT_CONFIRMATORY__ZERO_DENOMINATOR_AND_"
                            "FAILURE_HANDLING_NOT_RESULT_BLIND_FROZEN"
                        ),
                        "analysis_status": ANALYSIS_STATUS,
                    }
                )
    return (
        (summary_fields, summary_rows),
        (value_fields, value_rows),
    )


def _cluster_fields() -> tuple[str, ...]:
    return (
        "comparison_index",
        "comparison_id",
        "analysis_workload_id",
        "endpoint_id",
        "cluster_id",
        "paired_task_count",
        "cluster_effect",
        "cluster_lower_bound",
        "cluster_upper_bound",
        "effect_direction",
        "analysis_status",
    )


def _verify_file_commitment(
    path: Path,
    commitment: Mapping[str, Any],
    *,
    label: str,
) -> None:
    _require(path.is_file(), f"{label} is missing")
    _require(
        path.stat().st_size == int(commitment["bytes"]),
        f"{label} byte count mismatch",
    )
    _require(
        file_sha256(path) == commitment["sha256"],
        f"{label} SHA-256 mismatch",
    )


def _validate_export(input_root: Path) -> Mapping[str, Any]:
    manifest_path = input_root / "r9_export_manifest.json"
    _require(
        file_sha256(manifest_path) == R9_EXPORT_MANIFEST_SHA256,
        "R9 export manifest binding drifted",
    )
    manifest = _read_json(manifest_path)
    _require(
        manifest["raw_run_manifest_sha256"] == RAW_MANIFEST_SHA256,
        "raw manifest binding drifted",
    )
    _require(
        manifest["raw_source_mutated_or_deleted"] is False,
        "export manifest reports source mutation/deletion",
    )
    _require(
        manifest["failure_cost_rows"] == 5_030
        and manifest["post_execution_hard_violation_rows"] == 5_030,
        "supporting table row count binding drifted",
    )
    for name, commitment in manifest["artifacts"].items():
        _verify_file_commitment(
            input_root / name,
            commitment,
            label=f"R9 export artifact {name}",
        )
    return manifest


def validate_r9_supporting_inputs(
    *,
    project_root: Path,
    input_root: Path,
    r5_contract_path: Path,
    implementation_contract_path: Path,
    implementation_contract_sha256: str,
    output_root: Path,
    authorization: str,
) -> dict[str, Any]:
    """Validate PRE-R10 source and governance bindings without analysis."""

    project_root = project_root.resolve()
    input_root = input_root.resolve()
    r5_contract_path = r5_contract_path.resolve()
    implementation_contract_path = implementation_contract_path.resolve()
    output_root = output_root.resolve()
    _require(
        authorization == AUTHORIZATION_TOKEN,
        "exact PRE-R10 authorization token was not supplied",
    )
    _require(input_root.is_dir(), "R9 event export root is missing")
    _require(not output_root.exists(), "supporting output root already exists")
    _require(
        not _is_within(output_root, input_root),
        "supporting output must be outside the read-only input root",
    )
    _require(
        "r10" not in output_root.name.casefold(),
        "supporting output root must not claim R10",
    )
    contract_hash = file_sha256(implementation_contract_path)
    _require(
        contract_hash == implementation_contract_sha256,
        "implementation contract SHA-256 argument is incorrect",
    )
    contract = _read_json(implementation_contract_path)
    _require(
        contract["implementation_id"] == IMPLEMENTATION_ID,
        "supporting implementation ID drifted",
    )
    _require(
        contract["status"]
        == "RESULT_AWARE_AUTHOR_APPROVED_PRE_R10_FROZEN",
        "supporting implementation contract is not frozen",
    )
    _require(
        contract["analysis_status"] == ANALYSIS_STATUS,
        "supporting analysis status drifted",
    )
    _require(
        contract["authorization"]["token"] == AUTHORIZATION_TOKEN,
        "supporting authorization token drifted",
    )
    _require(
        contract["authorization"]["r10_authorized"] is False,
        "contract unexpectedly authorizes R10",
    )
    _require(
        _same_path(
            input_root,
            Path(contract["input_binding"]["input_root"]),
        ),
        "input root differs from supporting contract",
    )
    _require(
        contract["input_binding"]["export_manifest_sha256"]
        == R9_EXPORT_MANIFEST_SHA256,
        "contract export binding drifted",
    )
    _require(
        file_sha256(r5_contract_path) == R5_CONTRACT_SHA256,
        "R5 contract SHA-256 drifted",
    )
    for relative, commitment in contract[
        "implementation_source_artifacts"
    ].items():
        _verify_file_commitment(
            project_root / relative,
            commitment,
            label=f"supporting source artifact {relative}",
        )
    manifest = _validate_export(input_root)
    return {
        "artifact_role": "R9_PRE_R10_SUPPORTING_DESCRIPTIVE_AUDIT",
        "status": "VALIDATED",
        "analysis_status": ANALYSIS_STATUS,
        "implementation_id": IMPLEMENTATION_ID,
        "implementation_contract_sha256": contract_hash,
        "r5_contract_sha256": R5_CONTRACT_SHA256,
        "r9_export_manifest_sha256": R9_EXPORT_MANIFEST_SHA256,
        "raw_run_manifest_sha256": manifest[
            "raw_run_manifest_sha256"
        ],
        "new_confirmatory_hypotheses": 0,
        "new_confidence_intervals": 0,
        "new_p_values": 0,
        "new_holm_families": 0,
        "r10_authorized": False,
        "source_input_modified_or_deleted": False,
    }


def _output_payloads(
    outcomes: Sequence[OutcomeRow],
    hard_rows: Sequence[HardViolationRow],
    comparisons: Sequence[Comparison],
) -> dict[str, bytes]:
    outcome_index = _indexed_outcomes(outcomes)
    hard_index = _indexed_hard(hard_rows)
    method_fields, method_rows = _method_summary_payload(
        outcomes,
        hard_rows,
    )
    (failure_fields, failure_rows), failure_clusters = _failure_payloads(
        comparisons,
        outcome_index,
    )
    (hard_fields, hard_summary), hard_clusters = _hard_payloads(
        comparisons,
        hard_index,
    )
    (cost_fields, cost_rows), (
        cost_value_fields,
        cost_value_rows,
    ) = _cost_payloads(comparisons, outcome_index)
    readme = f"""# PRE-R10 supporting descriptive audit

- Implementation: `{IMPLEMENTATION_ID}`
- Analysis status: `{ANALYSIS_STATUS}`
- Input: the immutable `-02` R9 event export only.
- Excluded: `-01`, E3, old R9 v1, manuscript Results writing, and R10.
- Confirmatory R9 v2 remains unchanged and authoritative for its 30 hypotheses.
- This audit adds no stochastic confidence interval, p-value, sign-flip test,
  Holm family, or C4 decision.
- Failure direction: comparator rate minus proposed rate; positive favors the
  proposed method because failure is lower-is-better.
- Hard-violation bounds set every missing execution observation to no violation
  for the lower endpoint bound and to violation for the upper endpoint bound.
  Method-related missingness blocks a complete method-comparison statement.
- Cost ratios are proposed/comparator. Three available sets are reported:
  every completed task pair, equal charged-work pairs, and pairs where both
  methods accepted every event. No set is retrospectively promoted to a
  confirmatory cost gate.
- R10 remains blocked. These artifacts are inputs for a future separately
  authorized writing stage, not manuscript prose.
"""
    return {
        "README.md": readme.encode("utf-8"),
        "method_outcome_summary.csv": _csv_bytes(
            method_fields,
            method_rows,
        ),
        "pairwise_failure_summary.csv": _csv_bytes(
            failure_fields,
            failure_rows,
        ),
        "pairwise_hard_violation_summary.csv": _csv_bytes(
            hard_fields,
            hard_summary,
        ),
        "pairwise_cost_summary.csv": _csv_bytes(
            cost_fields,
            cost_rows,
        ),
        "paired_cost_values.csv": _csv_bytes(
            cost_value_fields,
            cost_value_rows,
        ),
        "supporting_cluster_effects.csv": _csv_bytes(
            _cluster_fields(),
            [*failure_clusters, *hard_clusters],
        ),
    }


def _write_new_file(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except OSError as error:
        raise R9SupportingError(
            f"cannot create output artifact: {path}"
        ) from error


def run_r9_supporting_descriptive(
    *,
    project_root: Path,
    input_root: Path,
    r5_contract_path: Path,
    implementation_contract_path: Path,
    implementation_contract_sha256: str,
    output_root: Path,
    authorization: str,
) -> dict[str, Any]:
    """Run the deterministic PRE-R10 supporting audit."""

    validation = validate_r9_supporting_inputs(
        project_root=project_root,
        input_root=input_root,
        r5_contract_path=r5_contract_path,
        implementation_contract_path=implementation_contract_path,
        implementation_contract_sha256=implementation_contract_sha256,
        output_root=output_root,
        authorization=authorization,
    )
    project_root = project_root.resolve()
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    r5 = _read_json(r5_contract_path.resolve())
    contract = _read_json(implementation_contract_path.resolve())
    outcomes = _load_outcomes(input_root)
    hard_rows = _load_hard_rows(input_root)
    comparisons = _comparisons(r5)
    payloads = _output_payloads(outcomes, hard_rows, comparisons)

    for relative, commitment in contract[
        "implementation_source_artifacts"
    ].items():
        _verify_file_commitment(
            project_root / relative,
            commitment,
            label=f"supporting source artifact {relative}",
        )
    _validate_export(input_root)
    _require(
        not output_root.exists(),
        "supporting output root appeared during analysis",
    )
    try:
        output_root.mkdir(parents=False, exist_ok=False)
    except OSError as error:
        raise R9SupportingError(
            f"cannot create supporting output root: {output_root}"
        ) from error
    for name, payload in payloads.items():
        _write_new_file(output_root / name, payload)

    _validate_export(input_root)
    for relative, commitment in contract[
        "implementation_source_artifacts"
    ].items():
        _verify_file_commitment(
            project_root / relative,
            commitment,
            label=f"supporting source artifact {relative}",
        )
    artifacts = {
        name: {
            "bytes": (output_root / name).stat().st_size,
            "sha256": file_sha256(output_root / name),
        }
        for name in sorted(payloads)
    }
    manifest = {
        "artifact_role": "R9_PRE_R10_SUPPORTING_DESCRIPTIVE_AUDIT",
        "implementation_id": IMPLEMENTATION_ID,
        "status": "COMPLETE",
        "analysis_status": ANALYSIS_STATUS,
        "authorization": AUTHORIZATION_TOKEN,
        "implementation_contract_sha256": (
            implementation_contract_sha256
        ),
        "r5_contract_sha256": R5_CONTRACT_SHA256,
        "r9_export_manifest_sha256": R9_EXPORT_MANIFEST_SHA256,
        "raw_run_manifest_sha256": RAW_MANIFEST_SHA256,
        "scope": {
            "minus_01_included": False,
            "e3_included": False,
            "r10_authorized": False,
            "manuscript_results_written": False,
            "source_input_modified_or_deleted": False,
        },
        "statistical_identity": {
            "comparison_order_reuses_r5_registered_pair_order": True,
            "comparison_count": len(comparisons),
            "new_confirmatory_hypotheses": 0,
            "new_confidence_intervals": 0,
            "new_sign_flip_tests": 0,
            "new_p_values": 0,
            "new_holm_families": 0,
            "retroactive_c4_decision": False,
        },
        "reporting": {
            "failure_margin_reference": FAILURE_MARGIN,
            "hard_violation_margin_reference": HARD_VIOLATION_MARGIN,
            "wall_ratio_point_margin_reference": WALL_RATIO_MARGIN,
            "rss_ratio_point_margin_reference": RSS_RATIO_MARGIN,
            "cost_available_sets": list(COST_AVAILABLE_SETS),
            "method_related_missingness_complete_comparison_allowed": False,
        },
        "row_counts": {
            "source_failure_cost": len(outcomes),
            "source_hard_violation": len(hard_rows),
            "method_outcome_summary": _csv_data_row_count(
                payloads["method_outcome_summary.csv"]
            ),
            "pairwise_failure_summary": _csv_data_row_count(
                payloads["pairwise_failure_summary.csv"]
            ),
            "pairwise_hard_violation_summary": _csv_data_row_count(
                payloads["pairwise_hard_violation_summary.csv"]
            ),
            "pairwise_cost_summary": _csv_data_row_count(
                payloads["pairwise_cost_summary.csv"]
            ),
            "paired_cost_values": _csv_data_row_count(
                payloads["paired_cost_values.csv"]
            ),
            "supporting_cluster_effects": _csv_data_row_count(
                payloads["supporting_cluster_effects.csv"]
            ),
        },
        "artifacts": artifacts,
        "runtime": {
            "python": (
                f"{sys.version_info.major}."
                f"{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            )
        },
    }
    manifest_path = output_root / "supporting_analysis_manifest.json"
    _write_new_file(
        manifest_path,
        canonical_json_bytes(manifest) + b"\n",
    )
    return {
        **validation,
        "status": "COMPLETE",
        "output_root": str(output_root),
        "output_artifacts": {
            **artifacts,
            "supporting_analysis_manifest.json": {
                "bytes": manifest_path.stat().st_size,
                "sha256": file_sha256(manifest_path),
            },
        },
        "r10_authorized": False,
        "source_input_modified_or_deleted": False,
    }
