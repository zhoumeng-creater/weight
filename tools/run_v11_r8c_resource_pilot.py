"""Run the isolated result-blind R8C resource qualification pilot."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import multiprocessing
import os
from pathlib import Path
import sys


THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
for _name, _value in THREAD_ENVIRONMENT.items():
    os.environ[_name] = _value

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from resource_pilot.r8c import (
    DEFAULT_CFE_PER_EVENT,
    DEFAULT_DYNAMIC_EVENTS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REPETITIONS_PER_PROFILE,
    DEFAULT_WORKERS,
    ResourcePilotError,
    run_resource_pilot,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an R6-authorized 100/100 control-plane-only resource pilot; "
            "no formal request or effect field is consumed or persisted."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=list(DEFAULT_WORKERS),
    )
    parser.add_argument(
        "--repetitions-per-profile",
        type=int,
        default=DEFAULT_REPETITIONS_PER_PROFILE,
    )
    parser.add_argument(
        "--cfe-per-event",
        type=int,
        default=DEFAULT_CFE_PER_EVENT,
    )
    parser.add_argument(
        "--dynamic-events",
        type=int,
        default=DEFAULT_DYNAMIC_EVENTS,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Use one worker, one repetition, 100 CFE, and one event; "
            "the output root is still single-use."
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    worker_counts = args.workers
    repetitions = args.repetitions_per_profile
    cfe_per_event = args.cfe_per_event
    dynamic_events = args.dynamic_events
    if args.smoke:
        worker_counts = [1]
        repetitions = 1
        cfe_per_event = 100
        dynamic_events = 1
    try:
        report = run_resource_pilot(
            output_root=args.output_root,
            worker_counts=worker_counts,
            repetitions_per_profile=repetitions,
            cfe_per_event=cfe_per_event,
            dynamic_events=dynamic_events,
        )
    except ResourcePilotError as error:
        print(f"R8C resource pilot refused: {type(error).__name__}")
        return 2
    print(
        "R8C resource pilot "
        f"{report['status']}: {args.output_root.resolve()}"
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
