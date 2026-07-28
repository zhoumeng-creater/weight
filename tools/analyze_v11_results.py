"""Single read-only v1.1 analysis entrypoint.

During R2 this command performs integrity validation only. Effect analysis,
participant-data access, Results writing, and figure generation remain
unauthorized.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from evaluation.run_manifest import (
    R2_ARTIFACT_ROLE,
    validate_r2_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate an R2 correctness artifact without analysis"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--integrity-only", action="store_true")
    parser.add_argument("--effect-analysis", action="store_true")
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--participant-data", action="store_true")
    parser.add_argument("--generate-figures", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if (
        not args.integrity_only
        or args.effect_analysis
        or args.write_results
        or args.participant_data
        or args.generate_figures
    ):
        raise RuntimeError(
            "R2 analysis is integrity-only; effect/Results/data/figures "
            "remain prohibited"
        )
    requested = Path(args.manifest)
    if not requested.is_absolute():
        raise RuntimeError("R2 manifest path must be absolute")
    project_root = Path(__file__).resolve().parents[1]
    manifest_path = requested.resolve()
    if manifest_path == project_root or manifest_path.is_relative_to(
        project_root
    ):
        raise RuntimeError(
            "R2 analysis input must be outside the repository"
        )
    manifest = validate_r2_manifest(manifest_path)
    return {
        "analysis_performed": False,
        "artifact_role": R2_ARTIFACT_ROLE,
        "effect_estimation_performed": False,
        "integrity_status": "PASS",
        "run_id": manifest["run_id"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        summary = run(args)
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 2
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
