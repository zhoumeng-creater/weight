"""Authenticate an R8C E1+E2 run and print only control-plane facts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analysis.r8c_batch_outputs import (  # noqa: E402
    R8CIntegrityError,
    validate_r8c_e1e2_run,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Result-blind full-batch integrity audit for the exact 5,030-task "
            "R8C E1+E2 freeze"
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--expected-run-manifest-sha256")
    return parser


def main(arguments: list[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    try:
        report = validate_r8c_e1e2_run(
            args.run_root,
            expected_run_manifest_sha256=(
                args.expected_run_manifest_sha256
            ),
        )
    except R8CIntegrityError as error:
        print(
            json.dumps(
                {
                    "artifact_role": (
                        "R8C_E1E2_CONTROL_PLANE_INTEGRITY_NO_EFFECTS"
                    ),
                    "integrity_status": "FAIL_CLOSED",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "effect_values_emitted": False,
                    "effect_endpoints_computed": False,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            report.control_plane_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
