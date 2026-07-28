from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "requirements-r8c-linux-x86_64.lock"
_REQUIREMENT = re.compile(
    r"^(?P<name>[A-Za-z0-9_.-]+)==(?P<version>\S+) "
    r"--hash=sha256:(?P<sha256>[0-9a-f]{64})$"
)


def _historical_pins() -> dict[str, str]:
    pins: dict[str, str] = {}
    for name in (
        "requirements-r2.lock",
        "requirements-r3-qualification.lock",
        "requirements-r4-benchmark.lock",
    ):
        for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
            match = _REQUIREMENT.fullmatch(line)
            if match is None:
                continue
            package = match.group("name").lower().replace("_", "-")
            version = match.group("version")
            previous = pins.setdefault(package, version)
            assert previous == version
    return pins


def test_r8c_linux_lock_is_complete_unique_and_hash_bound() -> None:
    raw = LOCK.read_text(encoding="utf-8")
    assert "Target: CPython 3.12, Linux x86_64" in raw
    assert "--require-hashes" in raw
    assert "--only-binary=:all:" in raw

    observed: dict[str, str] = {}
    for line in raw.splitlines():
        if not line or line.startswith(("#", "--")):
            continue
        match = _REQUIREMENT.fullmatch(line)
        assert match is not None
        package = match.group("name").lower().replace("_", "-")
        assert package not in observed
        observed[package] = match.group("version")

    assert observed == _historical_pins()
    assert len(observed) == 35
