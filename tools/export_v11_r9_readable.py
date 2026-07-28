"""Create compact human-readable R9 tables from one locked R8C raw root."""

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
    R9ExportAuthorizationError,
    export_r9_readable_outputs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Authorization-gated compact R9 endpoint, negative-transfer, "
            "hard-violation and failure/cost tables with a README; source "
            "artifacts are never changed or deleted"
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--raw-manifest-sha256", required=True)
    parser.add_argument("--reference-catalog", required=True, type=Path)
    parser.add_argument("--reference-catalog-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--authorize-r9-export", required=True)
    parser.add_argument(
        "--include-event-diagnostics",
        action="store_true",
        help=(
            "also create compressed event_diagnostics.jsonl.gz; omitted by "
            "default"
        ),
    )
    return parser


def main(arguments: list[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    try:
        result = export_r9_readable_outputs(
            args.run_root,
            raw_manifest_sha256=args.raw_manifest_sha256,
            authorization=args.authorize_r9_export,
            reference_catalog_path=args.reference_catalog,
            reference_catalog_sha256=(
                args.reference_catalog_sha256
            ),
            output_root=args.output_root,
            include_event_diagnostics=args.include_event_diagnostics,
        )
    except (R8CIntegrityError, R9ExportAuthorizationError) as error:
        print(
            json.dumps(
                {
                    "artifact_role": (
                        "R9_AUTHORIZED_COMPACT_HUMAN_READABLE_DERIVATION"
                    ),
                    "status": "FAIL_CLOSED",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "source_artifacts_deleted": False,
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
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
