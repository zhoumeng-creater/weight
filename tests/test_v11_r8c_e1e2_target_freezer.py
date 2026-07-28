from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any

from jsonschema import Draft202012Validator
import pytest


ROOT = Path(__file__).resolve().parents[1]
FREEZER_PATH = (
    ROOT / "tools" / "freeze_v11_r8c_e1e2_target_execution.py"
)
SUPPORT_PATH = (
    ROOT / "tests" / "test_v11_r8c_e1e2_target_qualified_schemas.py"
)
CONTRACT_SCHEMA = (
    ROOT
    / "config"
    / "r8c_e1e2"
    / "r8c_e1e2_target_qualified_contract.schema.json"
)
REQUEST_SCHEMA = (
    ROOT
    / "config"
    / "r8c_e1e2"
    / "r8c_e1e2_target_qualified_execution_request.schema.json"
)


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def freezer() -> ModuleType:
    return _load_module(FREEZER_PATH, "_test_r8c_e1e2_target_freezer")


@pytest.fixture
def valid_report(
    tmp_path: Path,
    freezer: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    support = _load_module(
        SUPPORT_PATH,
        "_test_r8c_e1e2_target_freezer_support",
    )
    fixture_root = tmp_path / "qualification-fixture"
    fixture_root.mkdir()
    qualified = support._qualified_contract(fixture_root)
    report_path = Path(
        qualified["target_qualification_evidence"][
            "qualification_report_path"
        ]
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    runtime_environment = deepcopy(report["runtime_environment_lock"])
    monkeypatch.setattr(
        freezer.formal_runner,
        "_runtime_environment_lock_evidence",
        lambda: deepcopy(runtime_environment),
    )
    source = deepcopy(report["code_identity"])
    monkeypatch.setattr(
        freezer.formal_runner,
        "_validate_source",
        lambda _request: {
            "git_commit": source["git_commit"],
            "git_tree": source["git_tree"],
            "git_dirty": False,
        },
    )
    return report_path


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _paths(tmp_path: Path) -> dict[str, Path]:
    control = tmp_path / "control"
    scratch = tmp_path / "scratch"
    control.mkdir()
    scratch.mkdir()
    return {
        "contract_path": control / "target-qualified-contract.json",
        "request_path": control / "target-qualified-request.json",
        "request_consumption_marker": control / "request.consumed.json",
        "output_root": scratch / "r8c-e1e2-formal-20260727-01",
    }


def _freeze(
    freezer: ModuleType,
    report_path: Path,
    paths: dict[str, Path],
) -> dict[str, Any]:
    return freezer.freeze_target_execution(
        qualification_report_path=report_path,
        provider="TEST_CLOUD_PROVIDER",
        instance_type="EPYC9754_64C_80G_TEST_FIXTURE",
        author_authorization_text=(
            "Authorize only the frozen formal E1+E2 public benchmark "
            "effect execution."
        ),
        created_date="2026-07-27",
        **paths,
    )


def _validator(path: Path) -> Draft202012Validator:
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(
        schema,
        format_checker=Draft202012Validator.FORMAT_CHECKER,
    )


def test_target_freezer_publishes_canonical_valid_pair_without_launch(
    tmp_path: Path,
    freezer: ModuleType,
    valid_report: Path,
) -> None:
    paths = _paths(tmp_path)
    summary = _freeze(freezer, valid_report, paths)

    contract_payload = paths["contract_path"].read_bytes()
    request_payload = paths["request_path"].read_bytes()
    contract = json.loads(contract_payload)
    request = json.loads(request_payload)

    assert contract_payload == _canonical_bytes(contract)
    assert request_payload == _canonical_bytes(request)
    _validator(CONTRACT_SCHEMA).validate(contract)
    _validator(REQUEST_SCHEMA).validate(request)
    assert request["contracts"]["r8c_formal_contract_sha256"] == sha256(
        contract_payload
    ).hexdigest()
    assert contract["target_qualification_evidence"][
        "selected_worker_count"
    ] == 1
    assert contract["resources"]["parallelism"]["max_workers"] == 1
    assert request["frozen_exact_command"] == contract["launch"][
        "exact_command"
    ]
    expected_prefix = (
        subprocess.list2cmdline([sys.executable])
        if os.name == "nt"
        else freezer.shlex.join([sys.executable])
    )
    assert contract["launch"]["exact_command"].startswith(expected_prefix)
    assert summary["request_consumed"] is False
    assert summary["formal_execution_started"] is False
    assert not paths["request_consumption_marker"].exists()
    assert not paths["output_root"].exists()


def test_target_freezer_rejects_canonical_semantic_report_tamper(
    tmp_path: Path,
    freezer: ModuleType,
    valid_report: Path,
) -> None:
    report = json.loads(valid_report.read_text(encoding="utf-8"))
    report["e1_e2_wall_projection"]["projections"][0][
        "formal_cfe"
    ] -= 1
    valid_report.write_bytes(_canonical_bytes(report))
    paths = _paths(tmp_path)

    with pytest.raises(Exception, match="projection|qualification|inconsistent"):
        _freeze(freezer, valid_report, paths)

    assert not paths["contract_path"].exists()
    assert not paths["request_path"].exists()


def test_target_freezer_rejects_non_go_recommendation(
    tmp_path: Path,
    freezer: ModuleType,
    valid_report: Path,
) -> None:
    report = json.loads(valid_report.read_text(encoding="utf-8"))
    recommendation = report["worker_recommendation"]
    recommendation["recommended_projected_wall_hours"] = 40.0
    recommendation["recommended_projected_wall_hours_with_25_percent_headroom"] = (
        50.0
    )
    recommendation["recommended_decision_classification"] = (
        "HOLD_OPTIMIZE_CONTENTION_AND_RETEST"
    )
    selected = report["e1_e2_wall_projection"]["projections"][0]
    selected["projected_wall_seconds"] = 144000.0
    selected["projected_wall_hours"] = 40.0
    selected["projected_wall_hours_with_25_percent_headroom"] = 50.0
    selected["decision_classification"] = (
        "HOLD_OPTIMIZE_CONTENTION_AND_RETEST"
    )
    valid_report.write_bytes(_canonical_bytes(report))
    paths = _paths(tmp_path)

    with pytest.raises(freezer.FreezeError, match="<=36h GO"):
        _freeze(freezer, valid_report, paths)

    assert not paths["contract_path"].exists()
    assert not paths["request_path"].exists()


def test_target_freezer_refuses_overwrite_and_preserves_existing_file(
    tmp_path: Path,
    freezer: ModuleType,
    valid_report: Path,
) -> None:
    paths = _paths(tmp_path)
    sentinel = b"existing-author-controlled-file\n"
    paths["contract_path"].write_bytes(sentinel)

    with pytest.raises(freezer.FreezeError, match="overwrite is forbidden"):
        _freeze(freezer, valid_report, paths)

    assert paths["contract_path"].read_bytes() == sentinel
    assert not paths["request_path"].exists()
