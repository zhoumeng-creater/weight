from __future__ import annotations

import ast
import platform
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from tools import run_v11_experiment as runner_module


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
R2_DEPENDENCY_LOCK = PROJECT_ROOT / "requirements-r2.lock"
R3_QUALIFICATION_LOCK = (
    PROJECT_ROOT / "requirements-r3-qualification.lock"
)

REQUIRED_PACKAGES = (
    "dt_ramde_v11",
    "benchmark_adapters",
    "weight_application",
    "comparators",
    "evaluation",
    "analysis",
)

FORBIDDEN_LEGACY_MODULES = {
    "config",
    "data_loader",
    "de_algorithm",
    "evaluation_ledger",
    "experiment_runner",
    "fitness_evaluator",
    "font_manager",
    "main",
    "metabolic_model",
    "randomness",
    "result_schema",
    "run_expriments",
    "solution_generator",
    "statistics_utils",
    "virtual_subjects",
    "visualization",
}


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.partition(".")[0])
    return imports


def test_v11_package_roots_are_declared_and_importable() -> None:
    assert PYPROJECT.is_file(), "R2 requires a pyproject.toml package boundary"
    metadata = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    assert metadata["tool"]["setuptools"]["package-dir"] == {"" : "src"}

    for package in REQUIRED_PACKAGES:
        assert (SRC_ROOT / package / "__init__.py").is_file()

    code = "\n".join(
        [
            "import sys",
            f"sys.path.insert(0, {str(SRC_ROOT)!r})",
            *(f"import {package}" for package in REQUIRED_PACKAGES),
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_v11_source_has_no_legacy_or_path_injection_imports() -> None:
    source_files = sorted(SRC_ROOT.glob("**/*.py"))
    assert source_files, "R2 src package must not be empty"

    for path in source_files:
        source = path.read_text(encoding="utf-8")
        assert "sys.path" not in source, f"path injection is prohibited: {path}"
        assert not (_top_level_imports(path) & FORBIDDEN_LEGACY_MODULES), (
            f"v1.1 source imports a legacy root module: {path}"
        )


def test_legacy_runtime_and_tests_are_absent_from_public_release() -> None:
    legacy_root = PROJECT_ROOT / "legacy" / "weight_v0"
    assert not legacy_root.exists()

    for module in FORBIDDEN_LEGACY_MODULES:
        assert not (PROJECT_ROOT / f"{module}.py").exists(), (
            f"legacy module remains importable from the repository root: {module}"
        )

    current_tests = sorted((PROJECT_ROOT / "tests").glob("test_*.py"))
    assert current_tests
    assert all(path.name.startswith("test_v11_") for path in current_tests)
    for path in current_tests:
        assert not (_top_level_imports(path) & FORBIDDEN_LEGACY_MODULES), (
            f"current correctness test imports quarantined legacy code: {path}"
        )


def test_public_release_contains_only_authorized_markdown() -> None:
    current_readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    for stale_entrypoint in ("python main.py", "python run_expriments.py"):
        assert stale_entrypoint not in current_readme
    assert "tools/run_v11_experiment.py" in current_readme
    assert "tools/run_v11_r9_inference.py" in current_readme
    for required_argument in ("--scope", "--seed"):
        assert required_argument in current_readme
    ignored_generated_parts = {
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "build",
        "dist",
    }
    actual_markdown = {
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in PROJECT_ROOT.rglob("*.md")
        if not (set(path.relative_to(PROJECT_ROOT).parts) & ignored_generated_parts)
    }
    assert actual_markdown == {
        "README.md",
        "config/r8c_e1e2/cdf_operational_authority_audit.md",
        "data/processed/event-export-v1/README.md",
        "data/processed/inference-v2/README.md",
        "data/processed/supporting-v2/README.md",
    }
    assert not tuple((PROJECT_ROOT / "docs").rglob("*.md"))

    assert not (PROJECT_ROOT / "legacy").exists()
    for name in ("task_plan.md", "findings.md", "progress.md"):
        assert not (PROJECT_ROOT / name).exists()


def test_r2_dependency_lock_is_complete_hashed_and_scope_bounded() -> None:
    metadata = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    assert metadata["build-system"]["requires"] == [
        "setuptools==75.1.0",
        "wheel==0.44.0",
    ]
    assert metadata["project"]["dependencies"] == ["numpy==1.26.4"]
    assert metadata["project"]["optional-dependencies"]["test"] == [
        "pytest==8.3.4",
        "ruff==0.9.4",
    ]
    assert metadata["project"]["optional-dependencies"]["qualification"] == [
        "jsonschema==4.23.0",
        "pandas==2.2.2",
        "scipy==1.14.0",
    ]

    lock = R2_DEPENDENCY_LOCK.read_text(encoding="utf-8")
    assert "--require-hashes" in lock
    assert "--only-binary=:all:" in lock
    expected = {
        "colorama": "0.4.6",
        "iniconfig": "2.3.0",
        "numpy": "1.26.4",
        "packaging": "26.2",
        "pluggy": "1.6.0",
        "pytest": "8.3.4",
        "ruff": "0.9.4",
        "setuptools": "75.1.0",
        "wheel": "0.44.0",
    }
    for package, version in expected.items():
        line = next(
            item for item in lock.splitlines() if item.startswith(f"{package}==")
        )
        assert line.startswith(f"{package}=={version} ")
        assert "--hash=sha256:" in line
    assert "matplotlib" not in lock
    assert "pandas" not in lock


def test_r3_qualification_lock_is_separate_complete_and_hash_bound() -> None:
    lock = R3_QUALIFICATION_LOCK.read_text(encoding="utf-8")
    assert "--require-hashes" in lock
    assert "--only-binary=:all:" in lock
    expected = {
        "attrs": "24.2.0",
        "jsonschema-specifications": "2023.12.1",
        "jsonschema": "4.23.0",
        "numpy": "1.26.4",
        "pandas": "2.2.2",
        "python-dateutil": "2.9.0.post0",
        "pytz": "2024.1",
        "referencing": "0.35.1",
        "rpds-py": "0.20.0",
        "scipy": "1.14.0",
        "six": "1.16.0",
        "tzdata": "2024.1",
    }
    package_lines = [
        line for line in lock.splitlines() if "==" in line
    ]
    assert len(package_lines) == len(expected)
    for package, version in expected.items():
        line = next(
            item
            for item in package_lines
            if item.startswith(f"{package}==")
        )
        assert line.startswith(f"{package}=={version} ")
        assert "--hash=sha256:" in line
    assert "matplotlib" not in lock


def test_dependency_identity_measures_the_lock_and_runtime_platform(
    tmp_path: Path,
) -> None:
    if platform.system() != "Windows":
        pytest.skip("the legacy R2 dependency identity is Windows-bound")
    measured_lock = tmp_path / "requirements-r2.lock"
    measured_lock.write_bytes(R2_DEPENDENCY_LOCK.read_bytes())

    identity = runner_module._dependencies(tmp_path)
    assert identity["lock"]["sha256"] == runner_module.hashlib.sha256(
        measured_lock.read_bytes()
    ).hexdigest()
    assert identity["runtime_platform"] == {
        "system": "Windows",
        "machine": "AMD64",
    }

    measured_lock.write_bytes(measured_lock.read_bytes() + b"\n# tampered\n")
    with pytest.raises(Exception, match="dependency lock"):
        runner_module._dependencies(tmp_path)
