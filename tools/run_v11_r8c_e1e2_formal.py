"""Enter the staged E1+E2-only R8C formal supervisor profile."""

from __future__ import annotations

import sys

from run_v11_r8_formal import main


_SUPERVISOR_INTERNAL_OPTIONS = {
    "--execution-profile",
    "--worker",
    "--schedule-index",
    "--task-id",
    "--task-directory",
    "--stop-path",
}


def _reject_internal_overrides(arguments: list[str]) -> None:
    for argument in arguments:
        option = argument.split("=", 1)[0]
        if option in _SUPERVISOR_INTERNAL_OPTIONS:
            raise ValueError(
                f"{option} is reserved for the frozen E1+E2 wrapper"
            )


if __name__ == "__main__":
    try:
        _reject_internal_overrides(sys.argv[1:])
    except ValueError as error:
        print(f"ConfigurationError: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    raise SystemExit(
        main(
            [
                "--execution-profile",
                "corrective_r8c_e1e2",
                *sys.argv[1:],
            ]
        )
    )
