"""Single fail-closed v1.1 experiment entrypoint.

R2 permits only small synthetic/public correctness fixtures. This entrypoint
does not run effect experiments, read participant data, generate hidden
instances, or write scientific Results.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from benchmark_adapters.public_cmop import StaticCMOPPublicAdapter
from dt_ramde_v11.contracts import (
    AlgorithmConfig,
    ConfigurationError,
    ExecutionScope,
    R2ExecutionRequest,
)
from dt_ramde_v11.engine import DTRAMDE, SequenceRunResult
from evaluation.contracts import EvaluationResult
from evaluation.run_manifest import (
    R2_ARTIFACT_ROLE,
    R2_DEPENDENCY_LOCK,
    R2_RUNTIME_PLATFORM,
    build_r2_manifest,
    canonical_json_bytes,
    sha256_bytes,
)
from weight_application.adapter import SyntheticWeightAdapter
from weight_application.state import (
    SyntheticWeightModel,
    SyntheticWeightState,
)


class _FixtureSelector:
    selector_id = "R2-FIXTURE-MINIMUM-FIRST-OBJECTIVE"
    selector_version = "1.0.0"

    def identity(self) -> Mapping[str, Any]:
        return {
            "selector_id": self.selector_id,
            "selector_version": self.selector_version,
            "role": "R2_correctness_fixture_only",
        }

    def select(self, archive: Sequence[EvaluationResult]) -> str | None:
        if not archive:
            return None
        return min(
            archive,
            key=lambda result: (result.objectives[0], result.candidate_id),
        ).candidate_id


STATIC_FIXTURE_SPEC = {
    "fixture_id": "static_bridge_e0",
    "target_suite_id": "DAS-CMOP-PLATEMO-4.15",
    "target_problem_id": "DASCMOP1",
    "decision_dimension": 30,
    "equations": [
        "f1=sum(x_i^2)",
        "f2=sum((x_i-0.25)^2)",
        "g1=mean(x_i)-0.75<=0",
    ],
    "formal_public_instance": False,
}


def _static_fixture_evaluator(
    vector: Sequence[float],
    event_id: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if event_id != 0:
        raise FloatingPointError("static correctness fixture is TS1")
    values = np.asarray(vector, dtype=float)
    return (
        (
            float(np.sum(values**2)),
            float(np.sum((values - 0.25) ** 2)),
        ),
        (float(np.mean(values) - 0.75),),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an R2 v1.1 correctness fixture"
    )
    parser.add_argument(
        "--fixture",
        required=True,
        choices=("synthetic_weight_e0", "static_bridge_e0"),
    )
    parser.add_argument(
        "--scope",
        required=True,
        choices=tuple(scope.value for scope in ExecutionScope),
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--participant-data", action="store_true")
    parser.add_argument("--effect-estimation", action="store_true")
    parser.add_argument("--hidden-generation", action="store_true")
    parser.add_argument("--results-writing", action="store_true")
    parser.add_argument("--remote-git-mutation", action="store_true")
    parser.add_argument("--release-or-distribution", action="store_true")
    return parser


def _validate_output_root(raw_path: str, project_root: Path) -> Path:
    requested = Path(raw_path)
    if not requested.is_absolute():
        raise ConfigurationError(
            "R2 output root must be an explicit absolute path"
        )
    resolved = requested.resolve()
    if resolved == project_root or resolved.is_relative_to(project_root):
        raise ConfigurationError(
            "R2 output root must be outside the repository"
        )
    if resolved.exists():
        raise ConfigurationError(
            "R2 output root must not already exist"
        )
    return resolved


def _request(args: argparse.Namespace) -> R2ExecutionRequest:
    request = R2ExecutionRequest(
        scope=ExecutionScope(args.scope),
        participant_data_requested=args.participant_data,
        effect_estimation_requested=args.effect_estimation,
        hidden_generation_requested=args.hidden_generation,
        results_writing_requested=args.results_writing,
        remote_git_mutation_requested=args.remote_git_mutation,
        release_or_distribution_requested=args.release_or_distribution,
    )
    request.validate()
    return request


def _build_problem(
    args: argparse.Namespace,
    request: R2ExecutionRequest,
) -> tuple[
    SyntheticWeightAdapter | StaticCMOPPublicAdapter,
    AlgorithmConfig,
    dict[str, Any],
]:
    selector = _FixtureSelector()
    common = {
        "population_size": 4,
        "algorithm_seed": args.seed,
        "selector_id": selector.selector_id,
        "selector_version": selector.selector_version,
        "event_time_limit_seconds": 10.0,
        "execution_request": request,
    }
    if args.fixture == "synthetic_weight_e0":
        if request.scope is not ExecutionScope.UNIT_TEST_FIXTURE:
            raise ConfigurationError(
                "synthetic_weight_e0 requires unit_test_fixture scope"
            )
        model = SyntheticWeightModel(
            event_days=7.0,
            energy_density_kcal_per_kg=7700.0,
            fat_mass_change_fraction=0.75,
        )
        state = SyntheticWeightState(
            event_id=0,
            fat_mass_kg=24.0,
            lean_mass_kg=56.0,
            cumulative_energy_imbalance_kcal=0.0,
        )
        problem = SyntheticWeightAdapter(
            initial_state=state,
            target_mass_kg=77.0,
            model=model,
        )
        config = AlgorithmConfig(
            variant="FULL",
            cfe_per_event=8,
            max_events=2,
            timing_mode="TS2_fixed_periodic_replanning",
            method_label="DT-RAMDE_TS2_FULL",
            adapter_id=problem.adapter_id,
            adapter_version=problem.adapter_version,
            atomic_steps_per_evaluation=1,
            configuration_evidence_id="UNIT_TEST_FIXTURE",
            **common,
        )
        fixture = {
            "fixture_id": args.fixture,
            "role": "synthetic_energy_mass_correctness",
            "initial_state": state.to_dict(),
            "target_mass_kg": 77.0,
            "model": dict(model.identity()),
        }
        return problem, config, fixture

    if request.scope is not ExecutionScope.PUBLIC_CORRECTNESS_FIXTURE:
        raise ConfigurationError(
            "static_bridge_e0 requires public_correctness_fixture scope"
        )
    fixture_sha = sha256_bytes(canonical_json_bytes(STATIC_FIXTURE_SPEC))
    problem = StaticCMOPPublicAdapter(
        suite_id="DAS-CMOP-PLATEMO-4.15",
        problem_id="DASCMOP1",
        evaluator_version="STATIC-CMOP-EVAL-1.0.0",
        fixture_evaluator_sha256=fixture_sha,
        lower=(0.0,) * 30,
        upper=(1.0,) * 30,
        objective_names=("f1", "f2"),
        constraint_names=("g1",),
        evaluator=_static_fixture_evaluator,
    )
    config = AlgorithmConfig(
        variant="NO_CROSS_EVENT_MEMORY",
        cfe_per_event=4,
        max_events=1,
        timing_mode="TS1_single_event",
        method_label="F22_MG_STATIC",
        adapter_id=problem.adapter_id,
        adapter_version=problem.adapter_version,
        atomic_steps_per_evaluation=1,
        configuration_evidence_id="PUBLIC_CORRECTNESS_FIXTURE",
        **common,
    )
    return problem, config, dict(STATIC_FIXTURE_SPEC)


def _git_identity(project_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        [
            "git",
            "-C",
            str(project_root),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    return commit, bool(status.strip())


def _source_identity(project_root: Path) -> dict[str, Any]:
    paths = sorted((project_root / "src").rglob("*.py")) + [
        project_root / "tools" / "analyze_v11_results.py",
        project_root / "tools" / "run_v11_experiment.py",
    ]
    source_files = {
        path.relative_to(project_root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in paths
    }
    commit, dirty = _git_identity(project_root)
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "source_files": source_files,
    }


def _dependencies(project_root: Path) -> dict[str, Any]:
    lock_path = project_root / R2_DEPENDENCY_LOCK["path"]
    if not lock_path.is_file():
        raise ConfigurationError("R2 dependency lock file is missing")
    measured_lock_sha256 = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    if measured_lock_sha256 != R2_DEPENDENCY_LOCK["sha256"]:
        raise ConfigurationError(
            "R2 dependency lock file differs from the frozen identity"
        )
    machine = platform.machine().upper()
    if machine == "X86_64":
        machine = "AMD64"
    runtime_platform = {
        "system": platform.system(),
        "machine": machine,
    }
    if runtime_platform != R2_RUNTIME_PLATFORM:
        raise ConfigurationError(
            "runtime platform differs from the R2 dependency lock target"
        )
    try:
        package_version = metadata.version("dt-ramde-v11")
    except metadata.PackageNotFoundError:
        package_version = "0.1.0.dev0"
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "dt_ramde_v11_version": package_version,
        "lock": {
            **R2_DEPENDENCY_LOCK,
            "sha256": measured_lock_sha256,
        },
        "runtime_platform": runtime_platform,
    }


def _summaries(result: SequenceRunResult) -> tuple[dict[str, Any], dict[str, Any]]:
    terminal_counts = Counter(
        event.terminal.code.value for event in result.events
    )
    budget = {
        "event_count": len(result.events),
        "per_event": [
            {"event_id": event.event_id, **dict(event.ledger)}
            for event in result.events
        ],
        "unused_budget_transfer": False,
    }
    failures = {
        "terminal_counts": dict(sorted(terminal_counts.items())),
        "evaluation_failures": sum(
            event.ledger["evaluation_failures"] for event in result.events
        ),
        "repair_failures": sum(
            event.ledger["repair_failed"] for event in result.events
        ),
        "silent_retry": False,
    }
    return budget, failures


def run(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    request = _request(args)
    output_root = _validate_output_root(args.output_root, project_root)
    problem, config, fixture = _build_problem(args, request)
    selector = _FixtureSelector()
    config.validate()
    optimizer = DTRAMDE(config)
    result = optimizer.run_sequence(problem, selector=selector)
    raw_payload = {
        "artifact_role": R2_ARTIFACT_ROLE,
        "fixture_id": args.fixture,
        "run_result": result.to_dict(),
    }
    raw_bytes = canonical_json_bytes(raw_payload) + b"\n"
    raw_sha256 = sha256_bytes(raw_bytes)
    budget, failures = _summaries(result)
    manifest = build_r2_manifest(
        execution_scope=request.scope,
        fixture=fixture,
        method=dict(optimizer.identity()),
        adapter_identity=problem.identity(),
        selector_identity=selector.identity(),
        configuration=config.to_dict(),
        dependencies=_dependencies(project_root),
        randomness={
            "algorithm_seed": args.seed,
            "paired_fixture_seed": args.seed,
            "hidden_seed_used": False,
        },
        budget=budget,
        failures=failures,
        code=_source_identity(project_root),
        raw_sha256=raw_sha256,
        raw_bytes=len(raw_bytes),
    )
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "raw_result.json").write_bytes(raw_bytes)
    manifest_bytes = canonical_json_bytes(manifest) + b"\n"
    (output_root / "run_manifest.json").write_bytes(manifest_bytes)
    return {
        "artifact_role": R2_ARTIFACT_ROLE,
        "output_created": True,
        "run_id": manifest["run_id"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        summary = run(args)
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
