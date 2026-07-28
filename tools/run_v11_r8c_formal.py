"""Enter the corrective R8C profile of the result-blind formal supervisor."""

from __future__ import annotations

import sys

from run_v11_r8_formal import main


if __name__ == "__main__":
    raise SystemExit(
        main(["--execution-profile", "corrective_r8c", *sys.argv[1:]])
    )
