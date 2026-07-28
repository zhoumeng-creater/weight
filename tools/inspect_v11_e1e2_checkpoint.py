"""Validate compact E1/E2 evidence without exposing effect values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from formal_execution.checkpoint_data import (  # noqa: E402
    CheckpointDataError,
    read_checkpoint_file,
)
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Integrity-check one endpoint-sufficient checkpoint file. "
            "This integrity-only tool never exports objective values; use "
            "the raw-manifest-locked R9 exporter for effect access."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--input", required=True, type=Path)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    path = args.input.resolve()
    if not path.is_file():
        raise CheckpointDataError("checkpoint input does not exist")
    checkpoint_file = read_checkpoint_file(path)
    event_ids = sorted(
        {record.event_id for record in checkpoint_file.records}
    )
    regular_count = sum(
        record.kind == "checkpoint"
        for record in checkpoint_file.records
    )
    terminal_count = len(checkpoint_file.records) - regular_count
    summary: dict[str, object] = {
        "artifact_role": "R8C_E1E2_CHECKPOINT_INTEGRITY_ONLY",
        "status": "PASS",
        "task_id": checkpoint_file.metadata.task_id,
        "objective_dimension": (
            checkpoint_file.metadata.objective_dimension
        ),
        "event_count": len(event_ids),
        "checkpoint_record_count": regular_count,
        "terminal_record_count": terminal_count,
        "file_sha256": checkpoint_file.sha256,
        "effect_values_printed": False,
        "effect_values_exported": False,
        "effect_export_interface_available": False,
        "authorized_effect_access_path": (
            "tools/export_v11_r9_readable.py"
        ),
    }
    return summary


def main() -> int:
    try:
        result = run(_parser().parse_args())
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
