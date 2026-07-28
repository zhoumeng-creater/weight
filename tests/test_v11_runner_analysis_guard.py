from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import pytest

from dt_ramde_v11.contracts import (
    AlgorithmConfig,
    ExecutionScope,
    R2ExecutionRequest,
)
from evaluation.run_manifest import (
    R2_ARTIFACT_ROLE,
    R2_MANIFEST_SCHEMA,
    ManifestIntegrityError,
    _validate_event_ledgers,
    canonical_json_bytes,
    validate_r2_manifest,
)

pytestmark = pytest.mark.skipif(
    platform.system() != "Windows",
    reason="the legacy R2 execution and analysis runner is Windows-bound",
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
RUNNER = PROJECT_ROOT / "tools" / "run_v11_experiment.py"
ANALYZER = PROJECT_ROOT / "tools" / "analyze_v11_results.py"
MQ1_RUNNER = PROJECT_ROOT / "tools" / "run_v11_mq1_qualification.py"
R6_RUNNER = PROJECT_ROOT / "tools" / "run_v11_r6_pilot.py"
R6_VALIDATOR = PROJECT_ROOT / "tools" / "validate_r6_pilot.py"
R2_DEPENDENCY_LOCK = PROJECT_ROOT / "requirements-r2.lock"


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(SRC_ROOT)
        if not current
        else os.pathsep.join((str(SRC_ROOT), current))
    )
    return environment


def _run(
    output_root: Path,
    *,
    fixture: str = "synthetic_weight_e0",
    scope: str = "unit_test_fixture",
    extra: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--fixture",
            fixture,
            "--scope",
            scope,
            "--seed",
            "17",
            "--output-root",
            str(output_root),
            *extra,
        ],
        cwd=PROJECT_ROOT,
        env=_environment(),
        capture_output=True,
        check=False,
        text=True,
    )


def _analyze(
    manifest: Path,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            "--manifest",
            str(manifest),
            "--integrity-only",
            *extra,
        ],
        cwd=PROJECT_ROOT,
        env=_environment(),
        capture_output=True,
        check=False,
        text=True,
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reseal_manifest(
    output: Path,
    manifest: dict[str, Any],
    *,
    raw: dict[str, Any] | None = None,
) -> None:
    if raw is not None:
        raw_bytes = canonical_json_bytes(raw) + b"\n"
        (output / "raw_result.json").write_bytes(raw_bytes)
        manifest["raw_artifact"]["sha256"] = hashlib.sha256(
            raw_bytes
        ).hexdigest()
        manifest["raw_artifact"]["bytes"] = len(raw_bytes)
    binding = {
        "protocol": manifest["protocol"],
        "execution_scope": manifest["execution_scope"],
        "fixture_sha256": manifest["fixture"]["sha256"],
        "method": manifest["method"],
        "adapter_sha256": manifest["adapter"]["sha256"],
        "selector_sha256": manifest["selector"]["sha256"],
        "configuration_sha256": manifest["configuration"]["sha256"],
        "dependency_sha256": manifest["dependencies"]["sha256"],
        "randomness": manifest["randomness"],
        "budget": manifest["budget"],
        "failures": manifest["failures"],
        "code": {
            "git_commit": manifest["code"]["git_commit"],
            "git_dirty": manifest["code"]["git_dirty"],
            "source_bundle_sha256": manifest["code"][
                "source_bundle_sha256"
            ],
        },
        "permissions": manifest["permissions"],
        "raw_artifact": manifest["raw_artifact"],
        "completion": manifest["completion"],
        "parent_run_id": manifest["parent_run_id"],
        "deviation_ids": manifest["deviation_ids"],
    }
    binding_hash = hashlib.sha256(canonical_json_bytes(binding)).hexdigest()
    manifest["run_binding_sha256"] = binding_hash
    manifest["run_id"] = f"r2-{binding_hash[:24]}"
    (output / "run_manifest.json").write_bytes(
        canonical_json_bytes(manifest) + b"\n"
    )


def _event_schema_config(
    *,
    variant: str,
    method_label: str,
) -> AlgorithmConfig:
    return AlgorithmConfig(
        variant=variant,
        population_size=4,
        cfe_per_event=8,
        algorithm_seed=17,
        max_events=2,
        timing_mode="TS2_fixed_periodic_replanning",
        method_label=method_label,
        adapter_id="WGT-V11-SYNTHETIC-E0",
        adapter_version="1.1.0-r2-fixture",
        selector_id="R2-FIXTURE-MINIMUM-FIRST-OBJECTIVE",
        selector_version="1.0.0",
        atomic_steps_per_evaluation=1,
        event_time_limit_seconds=10.0,
        configuration_evidence_id="UNIT_TEST_FIXTURE",
        execution_request=R2ExecutionRequest(
            scope=ExecutionScope.UNIT_TEST_FIXTURE
        ),
    )


def test_raw_execution_feedback_nullability_is_bound_to_variant(
    tmp_path: Path,
) -> None:
    output = tmp_path / "feedback-schema"
    completed = _run(output)
    assert completed.returncode == 0, completed.stderr
    events = _read_json(output / "raw_result.json")["run_result"]["events"]

    full_events = json.loads(json.dumps(events))
    full_events[0]["execution_feedback"] = None
    with pytest.raises(
        ManifestIntegrityError,
        match="requires complete execution feedback",
    ):
        _validate_event_ledgers(
            full_events,
            config=_event_schema_config(
                variant="FULL",
                method_label="DT-RAMDE_TS2_FULL",
            ),
        )

    ablated_events = json.loads(json.dumps(events))
    with pytest.raises(
        ManifestIntegrityError,
        match="must serialize execution feedback as null",
    ):
        _validate_event_ledgers(
            ablated_events,
            config=_event_schema_config(
                variant="NO_EXECUTION_FEEDBACK",
                method_label="NO_EXECUTION_FEEDBACK",
            ),
        )

    for event in ablated_events:
        event["execution_feedback"] = None
    _validate_event_ledgers(
        ablated_events,
        config=_event_schema_config(
            variant="NO_EXECUTION_FEEDBACK",
            method_label="NO_EXECUTION_FEEDBACK",
        ),
    )


def test_unique_v11_entrypoints_exist_without_legacy_imports() -> None:
    assert RUNNER.is_file()
    assert ANALYZER.is_file()
    assert MQ1_RUNNER.is_file()
    assert R6_RUNNER.is_file()
    assert R6_VALIDATOR.is_file()
    assert sorted(
        path.name for path in (PROJECT_ROOT / "tools").glob("*v11*.py")
    ) == [
        "analyze_v11_results.py",
        "audit_v11_r8c_e1e2_run.py",
        "export_v11_r9_readable.py",
        "freeze_v11_r8c_e1e2_target_execution.py",
        "generate_v11_r8c_reference_catalog.py",
        "inspect_v11_e1e2_checkpoint.py",
        "run_v11_experiment.py",
        "run_v11_mq1_qualification.py",
        "run_v11_r6_pilot.py",
        "run_v11_r8_formal.py",
        "run_v11_r8c_e1e2_formal.py",
        "run_v11_r8c_e1e2_qualification.py",
        "run_v11_r8c_formal.py",
        "run_v11_r8c_resource_pilot.py",
        "run_v11_r9_inference.py",
        "run_v11_r9_supporting_descriptive.py",
    ]
    forbidden = (
        "experiment_runner",
        "run_expriments",
        "result_schema",
        "run_g5c",
        "g2b",
    )
    for path in (RUNNER, ANALYZER, MQ1_RUNNER):
        source = path.read_text(encoding="utf-8")
        assert "sys.path" not in source
        for name in forbidden:
            assert f"import {name}" not in source
            assert f"from {name}" not in source
    assert "--fixture" in RUNNER.read_text(encoding="utf-8")
    assert "--request" in MQ1_RUNNER.read_text(encoding="utf-8")
    qualification_source = MQ1_RUNNER.read_text(encoding="utf-8")
    assert "--effect-estimation" not in qualification_source
    assert "benchmark_effect" not in qualification_source


@pytest.mark.parametrize(
    ("scope", "extra"),
    [
        ("benchmark_effect", ()),
        ("weight_effect", ()),
        ("hidden", ()),
        ("confirmatory", ()),
        ("unit_test_fixture", ("--participant-data",)),
        ("unit_test_fixture", ("--effect-estimation",)),
        ("unit_test_fixture", ("--hidden-generation",)),
        ("unit_test_fixture", ("--results-writing",)),
        ("unit_test_fixture", ("--remote-git-mutation",)),
        ("unit_test_fixture", ("--release-or-distribution",)),
    ],
)
def test_runner_rejects_prohibited_scope_before_creating_output(
    tmp_path: Path,
    scope: str,
    extra: tuple[str, ...],
) -> None:
    output = tmp_path / f"blocked-{scope}-{len(extra)}"
    completed = _run(output, scope=scope, extra=extra)

    assert completed.returncode != 0
    assert "outside the R2 correctness scope" in completed.stderr or (
        "R2 prohibited permission requested" in completed.stderr
    )
    assert not output.exists()


def test_runner_requires_absolute_external_new_output_root(tmp_path: Path) -> None:
    relative = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--fixture",
            "synthetic_weight_e0",
            "--scope",
            "unit_test_fixture",
            "--seed",
            "17",
            "--output-root",
            "relative-output",
        ],
        cwd=PROJECT_ROOT,
        env=_environment(),
        capture_output=True,
        check=False,
        text=True,
    )
    assert relative.returncode != 0
    assert "absolute" in relative.stderr
    assert not (PROJECT_ROOT / "relative-output").exists()

    legacy = PROJECT_ROOT / "results" / "r2-forbidden"
    completed = _run(legacy)
    assert completed.returncode != 0
    assert "outside the repository" in completed.stderr
    assert not legacy.exists()

    existing = tmp_path / "already-exists"
    existing.mkdir()
    completed = _run(existing)
    assert completed.returncode != 0
    assert "must not already exist" in completed.stderr


@pytest.mark.parametrize(
    ("fixture", "scope"),
    [
        ("synthetic_weight_e0", "unit_test_fixture"),
        ("static_bridge_e0", "public_correctness_fixture"),
    ],
)
def test_runner_writes_only_r2_correctness_raw_and_manifest(
    tmp_path: Path,
    fixture: str,
    scope: str,
) -> None:
    output = tmp_path / fixture
    completed = _run(output, fixture=fixture, scope=scope)
    assert completed.returncode == 0, completed.stderr

    assert sorted(path.name for path in output.iterdir()) == [
        "raw_result.json",
        "run_manifest.json",
    ]
    manifest = _read_json(output / "run_manifest.json")
    raw = _read_json(output / "raw_result.json")
    assert manifest["schema_version"] == R2_MANIFEST_SCHEMA
    assert manifest["artifact_role"] == R2_ARTIFACT_ROLE
    assert manifest["execution_scope"] == scope
    assert manifest["fixture"]["fixture_id"] == fixture
    assert manifest["raw_artifact"]["sha256"] == _file_hash(
        output / "raw_result.json"
    )
    assert manifest["raw_artifact"]["bytes"] == (
        output / "raw_result.json"
    ).stat().st_size
    assert raw["artifact_role"] == R2_ARTIFACT_ROLE
    assert raw["fixture_id"] == fixture
    assert raw["run_result"]["effect_estimation_performed"] is False
    assert raw["run_result"]["hidden_seed_or_instance_generated"] is False
    assert raw["run_result"]["confirmatory_execution"] is False
    assert set(manifest["permissions"].values()) == {False}
    assert manifest["code"]["git_commit"]
    assert isinstance(manifest["code"]["git_dirty"], bool)
    assert len(manifest["code"]["source_bundle_sha256"]) == 64
    assert manifest["configuration"]["sha256"] == hashlib.sha256(
        canonical_json_bytes(manifest["configuration"]["value"])
    ).hexdigest()
    assert manifest["dependencies"]["sha256"] == hashlib.sha256(
        canonical_json_bytes(manifest["dependencies"]["value"])
    ).hexdigest()
    assert manifest["adapter"]["sha256"] == hashlib.sha256(
        canonical_json_bytes(manifest["adapter"]["identity"])
    ).hexdigest()
    validated = validate_r2_manifest(output / "run_manifest.json")
    assert validated["run_id"] == manifest["run_id"]


def test_manifest_binds_dependency_lock_and_rejects_resealed_substitution(
    tmp_path: Path,
) -> None:
    output = tmp_path / "dependency-lock"
    assert _run(output).returncode == 0
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dependency = manifest["dependencies"]["value"]
    assert dependency["lock"] == {
        "path": "requirements-r2.lock",
        "sha256": _file_hash(R2_DEPENDENCY_LOCK),
        "target": "CPython_3.12_Windows_AMD64",
        "install_mode": "require_hashes_only_binary_no_index_verified",
    }

    dependency["lock"]["sha256"] = "0" * 64
    manifest["dependencies"]["sha256"] = hashlib.sha256(
        canonical_json_bytes(dependency)
    ).hexdigest()
    _reseal_manifest(output, manifest)

    completed = _analyze(manifest_path)
    assert completed.returncode != 0
    assert "dependency lock" in completed.stderr.lower()


def test_clean_replay_is_byte_identical_across_external_roots(
    tmp_path: Path,
) -> None:
    first = tmp_path / "replay-a"
    second = tmp_path / "replay-b"
    assert _run(first).returncode == 0
    assert _run(second).returncode == 0

    assert (first / "raw_result.json").read_bytes() == (
        second / "raw_result.json"
    ).read_bytes()
    assert (first / "run_manifest.json").read_bytes() == (
        second / "run_manifest.json"
    ).read_bytes()


def test_analysis_is_read_only_integrity_validation(tmp_path: Path) -> None:
    output = tmp_path / "analysis-input"
    assert _run(output).returncode == 0
    manifest_path = output / "run_manifest.json"
    before = {path.name: _file_hash(path) for path in output.iterdir()}

    completed = _analyze(manifest_path)
    after = {path.name: _file_hash(path) for path in output.iterdir()}
    assert completed.returncode == 0, completed.stderr
    assert before == after
    assert sorted(path.name for path in output.iterdir()) == [
        "raw_result.json",
        "run_manifest.json",
    ]
    summary = json.loads(completed.stdout)
    assert summary == {
        "analysis_performed": False,
        "artifact_role": R2_ARTIFACT_ROLE,
        "effect_estimation_performed": False,
        "integrity_status": "PASS",
        "run_id": _read_json(manifest_path)["run_id"],
    }


@pytest.mark.parametrize(
    "extra",
    [
        ("--effect-analysis",),
        ("--write-results",),
        ("--participant-data",),
        ("--generate-figures",),
    ],
)
def test_analysis_rejects_non_integrity_actions(
    tmp_path: Path,
    extra: tuple[str, ...],
) -> None:
    output = tmp_path / f"analysis-blocked-{extra[0][2:]}"
    assert _run(output).returncode == 0
    before = {path.name: _file_hash(path) for path in output.iterdir()}

    completed = _analyze(output / "run_manifest.json", *extra)
    after = {path.name: _file_hash(path) for path in output.iterdir()}
    assert completed.returncode != 0
    assert "R2 analysis is integrity-only" in completed.stderr
    assert before == after


def test_analysis_rejects_raw_or_permission_tampering(tmp_path: Path) -> None:
    raw_output = tmp_path / "raw-tamper"
    assert _run(raw_output).returncode == 0
    raw_path = raw_output / "raw_result.json"
    raw_path.write_bytes(raw_path.read_bytes() + b" ")
    completed = _analyze(raw_output / "run_manifest.json")
    assert completed.returncode != 0
    assert "raw artifact" in completed.stderr

    manifest_output = tmp_path / "manifest-tamper"
    assert _run(manifest_output).returncode == 0
    manifest_path = manifest_output / "run_manifest.json"
    manifest = _read_json(manifest_path)
    manifest["permissions"]["effect_analysis"] = True
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    completed = _analyze(manifest_path)
    assert completed.returncode != 0
    assert "permission" in completed.stderr


@pytest.mark.parametrize(
    ("block", "key", "value"),
        [
            ("method", "role", "tampered"),
            ("budget", "event_count", 999),
            ("failures", "silent_retry", True),
            ("code", "git_dirty", None),
        ],
)
def test_analysis_recomputes_run_binding_and_raw_summaries(
    tmp_path: Path,
    block: str,
    key: str,
    value: Any,
) -> None:
    output = tmp_path / f"binding-tamper-{block}"
    assert _run(output).returncode == 0
    manifest_path = output / "run_manifest.json"
    manifest = _read_json(manifest_path)
    if block == "code" and key == "git_dirty":
        value = not bool(manifest[block][key])
    assert manifest[block][key] != value
    manifest[block][key] = value
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")

    completed = _analyze(manifest_path)
    assert completed.returncode != 0
    assert "binding" in completed.stderr or block in completed.stderr


def test_analysis_rejects_repository_and_legacy_result_paths() -> None:
    manifest = PROJECT_ROOT / "results" / "run_manifest.json"
    completed = _analyze(manifest)
    assert completed.returncode != 0
    assert "outside the repository" in completed.stderr


def test_analysis_rejects_resealed_results_payload_in_raw(
    tmp_path: Path,
) -> None:
    output = tmp_path / "resealed-results-payload"
    assert _run(output).returncode == 0
    manifest = _read_json(output / "run_manifest.json")
    raw = _read_json(output / "raw_result.json")
    raw["Results"] = {
        "effect_estimate": 1.23,
        "participant_rows": [{"participant_id": "forbidden"}],
    }
    _reseal_manifest(output, manifest, raw=raw)

    completed = _analyze(output / "run_manifest.json")
    assert completed.returncode != 0
    assert "raw artifact schema" in completed.stderr


@pytest.mark.parametrize(
    "mutation",
    [
        "scope",
        "adapter",
        "selector",
        "effect_request",
    ],
)
def test_analysis_rejects_resealed_semantic_binding_mismatch(
    tmp_path: Path,
    mutation: str,
) -> None:
    output = tmp_path / f"resealed-semantic-{mutation}"
    assert _run(output).returncode == 0
    manifest = _read_json(output / "run_manifest.json")
    raw = _read_json(output / "raw_result.json")

    if mutation == "scope":
        manifest["execution_scope"] = "public_correctness_fixture"
    elif mutation == "adapter":
        identity = manifest["adapter"]["identity"]
        identity["adapter_id"] = "TAMPERED-ADAPTER"
        manifest["adapter"]["sha256"] = hashlib.sha256(
            canonical_json_bytes(identity)
        ).hexdigest()
        raw["run_result"]["adapter_identity"] = identity
    elif mutation == "selector":
        identity = manifest["selector"]["identity"]
        identity["selector_id"] = "TAMPERED-SELECTOR"
        manifest["selector"]["sha256"] = hashlib.sha256(
            canonical_json_bytes(identity)
        ).hexdigest()
        raw["run_result"]["selector_identity"] = identity
    else:
        config = manifest["configuration"]["value"]
        config["execution_request"]["effect_estimation_requested"] = True
        manifest["configuration"]["sha256"] = hashlib.sha256(
            canonical_json_bytes(config)
        ).hexdigest()
        raw["run_result"]["config"] = config
    _reseal_manifest(output, manifest, raw=raw)

    completed = _analyze(output / "run_manifest.json")
    assert completed.returncode != 0
    assert any(
        label in completed.stderr
        for label in (
            "scope",
            "adapter",
            "selector",
            "permission",
            "prohibited",
        )
    )


def test_analysis_rejects_resealed_joint_budget_violation(
    tmp_path: Path,
) -> None:
    output = tmp_path / "resealed-joint-budget"
    assert _run(output).returncode == 0
    manifest = _read_json(output / "run_manifest.json")
    raw = _read_json(output / "raw_result.json")
    raw["run_result"]["events"][0]["ledger"]["objective_calls"] = 999
    manifest["budget"]["per_event"][0]["objective_calls"] = 999
    _reseal_manifest(output, manifest, raw=raw)

    completed = _analyze(output / "run_manifest.json")
    assert completed.returncode != 0
    assert any(
        label in completed.stderr.lower()
        for label in ("budget", "ledger", "objective")
    )


def test_analysis_rejects_noncanonical_manifest_bytes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "noncanonical-manifest"
    assert _run(output).returncode == 0
    manifest_path = output / "run_manifest.json"
    manifest = _read_json(manifest_path)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )

    completed = _analyze(manifest_path)
    assert completed.returncode != 0
    assert "canonical" in completed.stderr.lower()


def test_analysis_rejects_noncanonical_raw_bytes(tmp_path: Path) -> None:
    output = tmp_path / "noncanonical-raw"
    assert _run(output).returncode == 0
    raw_path = output / "raw_result.json"
    raw = _read_json(raw_path)
    raw_bytes = (
        json.dumps(raw, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    raw_path.write_bytes(raw_bytes)
    manifest = _read_json(output / "run_manifest.json")
    manifest["raw_artifact"]["sha256"] = hashlib.sha256(raw_bytes).hexdigest()
    manifest["raw_artifact"]["bytes"] = len(raw_bytes)
    _reseal_manifest(output, manifest)

    completed = _analyze(output / "run_manifest.json")
    assert completed.returncode != 0
    assert "canonical" in completed.stderr.lower()


def test_analysis_rejects_resealed_manifest_fixture_payload(
    tmp_path: Path,
) -> None:
    output = tmp_path / "resealed-manifest-fixture-payload"
    assert _run(output).returncode == 0
    manifest = _read_json(output / "run_manifest.json")
    manifest["fixture"]["Results"] = {
        "participant_rows": [{"participant_id": "forbidden"}],
    }
    fixture_value = {
        key: value
        for key, value in manifest["fixture"].items()
        if key != "sha256"
    }
    manifest["fixture"]["sha256"] = hashlib.sha256(
        canonical_json_bytes(fixture_value)
    ).hexdigest()
    _reseal_manifest(output, manifest)

    completed = _analyze(output / "run_manifest.json")
    assert completed.returncode != 0
    assert any(
        label in completed.stderr.lower()
        for label in ("fixture", "prohibited", "participant")
    )


@pytest.mark.parametrize(
    "mutation",
    ["transition", "evaluation_failures", "terminal"],
)
def test_analysis_rejects_resealed_event_invariant_violation(
    tmp_path: Path,
    mutation: str,
) -> None:
    output = tmp_path / f"resealed-event-{mutation}"
    assert _run(output).returncode == 0
    manifest = _read_json(output / "run_manifest.json")
    raw = _read_json(output / "raw_result.json")
    event = raw["run_result"]["events"][0]
    budget_event = manifest["budget"]["per_event"][0]
    if mutation == "transition":
        event["ledger"]["execution_transition_count"] = 0
        budget_event["execution_transition_count"] = 0
    elif mutation == "evaluation_failures":
        event["ledger"]["evaluation_failures"] = 999
        budget_event["evaluation_failures"] = 999
        manifest["failures"]["evaluation_failures"] = 999
    else:
        event["terminal"]["code"] = "FORGED_TERMINAL"
        manifest["failures"]["terminal_counts"] = {
            "ACCEPTED": 1,
            "FORGED_TERMINAL": 1,
        }
    _reseal_manifest(output, manifest, raw=raw)

    completed = _analyze(output / "run_manifest.json")
    assert completed.returncode != 0
    assert any(
        label in completed.stderr.lower()
        for label in (
            "transition",
            "failure",
            "terminal",
            "budget",
        )
    )


def test_analysis_rejects_resealed_public_adapter_target_mismatch(
    tmp_path: Path,
) -> None:
    output = tmp_path / "resealed-public-adapter-target"
    assert (
        _run(
            output,
            fixture="static_bridge_e0",
            scope="public_correctness_fixture",
        ).returncode
        == 0
    )
    manifest = _read_json(output / "run_manifest.json")
    raw = _read_json(output / "raw_result.json")
    identity = manifest["adapter"]["identity"]
    identity["target_problem_id"] = "BOGUS"
    manifest["adapter"]["sha256"] = hashlib.sha256(
        canonical_json_bytes(identity)
    ).hexdigest()
    raw["run_result"]["adapter_identity"] = identity
    _reseal_manifest(output, manifest, raw=raw)

    completed = _analyze(output / "run_manifest.json")
    assert completed.returncode != 0
    assert any(
        label in completed.stderr.lower()
        for label in ("adapter", "fixture", "target", "problem")
    )


def test_analysis_rejects_resealed_synthetic_fixture_model_mismatch(
    tmp_path: Path,
) -> None:
    output = tmp_path / "resealed-synthetic-fixture-model"
    assert _run(output).returncode == 0
    manifest = _read_json(output / "run_manifest.json")
    manifest["fixture"]["model"]["model_id"] = "FORGED-MODEL"
    fixture_value = {
        key: value
        for key, value in manifest["fixture"].items()
        if key != "sha256"
    }
    manifest["fixture"]["sha256"] = hashlib.sha256(
        canonical_json_bytes(fixture_value)
    ).hexdigest()
    _reseal_manifest(output, manifest)

    completed = _analyze(output / "run_manifest.json")
    assert completed.returncode != 0
    assert any(
        label in completed.stderr.lower()
        for label in ("fixture", "model", "adapter")
    )


def test_analysis_rejects_resealed_synthetic_model_role_escalation(
    tmp_path: Path,
) -> None:
    output = tmp_path / "resealed-synthetic-model-role"
    assert _run(output).returncode == 0
    manifest = _read_json(output / "run_manifest.json")
    raw = _read_json(output / "raw_result.json")
    identity = manifest["adapter"]["identity"]
    identity["model_role"]["effect_estimation_allowed"] = True
    identity["model_role"]["scientific_model_gate"] = "R3_QUALIFIED"
    manifest["adapter"]["sha256"] = hashlib.sha256(
        canonical_json_bytes(identity)
    ).hexdigest()
    raw["run_result"]["adapter_identity"] = identity
    _reseal_manifest(output, manifest, raw=raw)

    completed = _analyze(output / "run_manifest.json")
    assert completed.returncode != 0
    assert "model role" in completed.stderr.lower()


@pytest.mark.parametrize(
    "mutation",
    ["git_commit", "source_path", "source_hash"],
)
def test_analysis_rejects_resealed_invalid_code_provenance(
    tmp_path: Path,
    mutation: str,
) -> None:
    output = tmp_path / f"resealed-code-provenance-{mutation}"
    assert _run(output).returncode == 0
    manifest = _read_json(output / "run_manifest.json")
    code = manifest["code"]
    if mutation == "git_commit":
        code["git_commit"] = "NOT-A-COMMIT"
    else:
        source_files = code["source_files"]
        first_path = sorted(source_files)[0]
        if mutation == "source_path":
            source_files["../outside.py"] = source_files.pop(first_path)
        else:
            source_files[first_path] = "NOT-A-SHA256"
        code["source_bundle_sha256"] = hashlib.sha256(
            canonical_json_bytes(source_files)
        ).hexdigest()
    _reseal_manifest(output, manifest)

    completed = _analyze(output / "run_manifest.json")
    assert completed.returncode != 0
    assert any(
        label in completed.stderr.lower()
        for label in ("code", "commit", "source", "sha-256", "path")
    )


def test_manifest_validation_requires_external_raw_lock(tmp_path: Path) -> None:
    with pytest.raises(ManifestIntegrityError, match="manifest"):
        validate_r2_manifest(tmp_path / "missing-manifest.json")
