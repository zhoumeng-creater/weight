"""Validate or run the PRE-R10 R9 supporting descriptive audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analysis.r9_supporting_descriptive import (  # noqa: E402
    R9SupportingError,
    run_r9_supporting_descriptive,
    validate_r9_supporting_inputs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic PRE-R10 failure, hard-observation, and "
            "cost supporting artifacts without new confirmatory inference"
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--r5-contract", required=True, type=Path)
    parser.add_argument(
        "--implementation-contract",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--implementation-contract-sha256",
        required=True,
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--authorize-pre-r10-supporting-audit",
        required=True,
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(arguments: list[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    parameters = {
        "project_root": PROJECT_ROOT,
        "input_root": args.input_root,
        "r5_contract_path": args.r5_contract,
        "implementation_contract_path": (
            args.implementation_contract
        ),
        "implementation_contract_sha256": (
            args.implementation_contract_sha256
        ),
        "output_root": args.output_root,
        "authorization": args.authorize_pre_r10_supporting_audit,
    }
    try:
        if args.validate_only:
            result = validate_r9_supporting_inputs(**parameters)
        else:
            result = run_r9_supporting_descriptive(**parameters)
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        R9SupportingError,
    ) as error:
        print(
            json.dumps(
                {
                    "artifact_role": (
                        "R9_PRE_R10_SUPPORTING_DESCRIPTIVE_AUDIT"
                    ),
                    "status": "FAIL_CLOSED",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "r10_authorized": False,
                    "source_input_modified_or_deleted": False,
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
