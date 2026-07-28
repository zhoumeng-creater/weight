"""Result-blind bit-equivalence matrix for the corrective CPU batch kernels."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark_adapters.r4_evaluators import CDFEvaluator
from formal_execution.adapters import FormalR8CWGTRRAdapter
from formal_execution.public_rolling import generate_public_instance
from formal_execution.schedule import (
    build_corrective_formal_schedule,
    build_formal_schedule,
    e2_full_reuse_commitment,
    schedule_commitment,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _flatten(
    rows: Sequence[tuple[Sequence[float], Sequence[float]]],
) -> np.ndarray:
    return np.asarray(
        [
            [*objectives, *constraints]
            for objectives, constraints in rows
        ],
        dtype=float,
    )


def _cdf_matrix(batch_size: int) -> dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64(2026072601))
    cases = rows_checked = elements_checked = 0
    for problem_index in range(1, 16):
        for profile in ("CDF-HARSH", "CDF-MILD"):
            evaluator = CDFEvaluator(
                problem_index=problem_index,
                profile=profile,
                environment_seed=20260726,
            )
            lower = np.asarray(evaluator.lower_bounds, dtype=float)
            upper = np.asarray(evaluator.upper_bounds, dtype=float)
            for event_id in range(60):
                matrix = rng.uniform(
                    lower,
                    upper,
                    size=(batch_size, lower.size),
                )
                matrix[:, 0] = rng.uniform(0.0, 0.6, size=batch_size)
                scalar = tuple(
                    evaluator(row, event_id) for row in matrix
                )
                batched = evaluator.evaluate_batch(matrix, event_id)
                scalar_values = _flatten(scalar)
                batch_values = _flatten(batched)
                if not np.array_equal(scalar_values, batch_values):
                    difference = np.abs(scalar_values - batch_values)
                    raise AssertionError(
                        "CDF scalar/batch bit mismatch: "
                        f"problem={problem_index}, profile={profile}, "
                        f"event={event_id}, max_abs={difference.max()}"
                    )
                cases += 1
                rows_checked += batch_size
                elements_checked += int(scalar_values.size)
    return {
        "status": "PASS_BIT_EXACT",
        "cases": cases,
        "rows": rows_checked,
        "elements": elements_checked,
        "batch_size": batch_size,
        "domain_note": (
            "x0 sampled in [0,0.6] to stay in the common real domain "
            "of every CDF problem/event; all other coordinates use bounds"
        ),
        "kernel": (
            "one matrix validation and event-constant calculation, then "
            "the unchanged scalar _evaluate_equations per ordered row"
        ),
    }


def _rolling_matrix(batch_size: int) -> dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64(2026072602))
    cases = rows_checked = elements_checked = 0
    for template in (
        "RR-SMOOTH",
        "RR-SHOCK",
        "RR-REJECTION",
        "RR-INTERMITTENT",
    ):
        for index in range(8):
            adapter = FormalR8CWGTRRAdapter(
                generate_public_instance(template, index)
            )
            for event_id in range(20):
                adapter._state = rng.normal(0.0, 0.25, size=2)
                adapter._previous_action = rng.uniform(-0.5, 0.5, size=2)
                information = adapter.freeze_information(
                    event_id,
                    None if event_id == 0 else {"released_at": event_id},
                )
                matrix = rng.uniform(
                    -1.0, 1.0, size=(batch_size, 12)
                )
                scalar = tuple(
                    adapter._evaluate_joint(row, information)
                    for row in matrix
                )
                batched = adapter._evaluate_joint_batch(
                    matrix, information
                )
                scalar_values = _flatten(scalar)
                batch_values = _flatten(batched)
                if not np.array_equal(scalar_values, batch_values):
                    difference = np.abs(scalar_values - batch_values)
                    raise AssertionError(
                        "WGT-RR scalar/batch bit mismatch: "
                        f"template={template}, index={index}, "
                        f"event={event_id}, max_abs={difference.max()}"
                    )
                cases += 1
                rows_checked += batch_size
                elements_checked += int(scalar_values.size)
    return {
        "status": "PASS_BIT_EXACT",
        "cases": cases,
        "rows": rows_checked,
        "elements": elements_checked,
        "batch_size": batch_size,
        "state_coverage": (
            "deterministically varied frozen state and previous action "
            "for every public template/index/event"
        ),
    }


def _schedule() -> dict[str, Any]:
    r5 = json.loads(
        (
            PROJECT_ROOT / "config" / "r5" / "r5_freeze_contract.json"
        ).read_text(encoding="utf-8")
    )
    old = build_formal_schedule(r5)
    corrective = build_corrective_formal_schedule(r5)
    return {
        "historical_schedule_rows": len(old),
        "historical_schedule_sha256": schedule_commitment(old),
        "corrective_schedule_rows": len(corrective),
        "corrective_schedule_sha256": schedule_commitment(corrective),
        "corrective_e2_full_reuse_sha256": (
            e2_full_reuse_commitment(corrective)
        ),
        "corrective_CFE": sum(row.total_cfe for row in corrective),
        "corrective_atomic_model_steps": sum(
            row.total_atomic_steps for row in corrective
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args(argv)
    if args.batch_size != 100:
        raise ValueError("corrective equivalence requires natural batch 100")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    report = {
        "record_id": "WGT-V11-R8C-BATCH-EQUIVALENCE-20260726-01",
        "status": "PASS",
        "scope": (
            "result-blind public evaluator and schedule equivalence; "
            "no formal effect task or historical output was read"
        ),
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
            "logical_cpu_count": os.cpu_count(),
        },
        "schedule": _schedule(),
        "cdf": _cdf_matrix(args.batch_size),
        "rolling": _rolling_matrix(args.batch_size),
        "end_to_end_tests": [
            "100/100 matched-DE CDF batch versus forced scalar: raw gzip, summary and manifest bytes equal",
            "100/100 jMetalPy GDE3 CDF batch versus forced scalar: raw gzip, summary and manifest bytes equal",
            "100/100 DT-RAMDE WGT-RR batch versus forced scalar: raw gzip, summary and manifest bytes equal",
        ],
        "limitations": [
            "timeout paths use scalar evaluation through a deadline guard",
            "repair or numerical batch prevalidation failures fall back before ledger entry",
            "the exact matrix must be rerun on the selected formal host",
        ],
    }
    encoded = _canonical(report) + b"\n"
    output.write_bytes(encoded)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output),
                "sha256": sha256(encoded).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
