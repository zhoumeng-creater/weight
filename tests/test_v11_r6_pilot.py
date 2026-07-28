from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from dt_ramde_v11.contracts import (
    ConfigurationError,
    ExecutionScope,
    R6ExecutionRequest,
)
from evaluation.ledger import EvaluationLedger
from weight_application.illustrative_adapter import (
    IllustrativeHallEngineeringAdapter,
    R6_E3_SCENARIOS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from validate_r6_pilot import (  # noqa: E402
    DEFAULT_CONTRACT,
    R6ValidationError,
    _validate_historical_code_identity,
    validate_contract,
    validate_output,
)


RUNNER = TOOLS_ROOT / "run_v11_r6_pilot.py"


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    src = str(PROJECT_ROOT / "src")
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        src if not current else os.pathsep.join((src, current))
    )
    return environment


def test_r6_contract_is_result_blind_and_budget_balanced() -> None:
    contract = validate_contract()
    assert contract["budget"] == {
        "scheduled_worker_processes": 20,
        "total_cfe": 728,
        "total_atomic_model_steps": 4328,
        "unused_budget_transfer_allowed": False,
        "method_specific_budget_allowed": False,
    }
    assert contract["permissions"]["r6_engineering_pilot_allowed"] is True
    assert [
        value
        for key, value in contract["permissions"].items()
        if key != "r6_engineering_pilot_allowed"
    ] == [False] * 9
    assert contract["next_gate"]["authorized"] is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scope": ExecutionScope.UNIT_TEST_FIXTURE},
        {
            "scope": ExecutionScope.ENGINEERING_PILOT,
            "nonformal_development_fixture_acknowledged": False,
        },
        {
            "scope": ExecutionScope.ENGINEERING_PILOT,
            "effect_estimation_requested": True,
        },
        {
            "scope": ExecutionScope.ENGINEERING_PILOT,
            "formal_subject_generation_requested": True,
        },
        {
            "scope": ExecutionScope.ENGINEERING_PILOT,
            "method_comparison_requested": True,
        },
    ],
)
def test_r6_request_fails_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        R6ExecutionRequest(**kwargs).validate()


@pytest.mark.parametrize("scenario", R6_E3_SCENARIOS)
def test_all_r6_scenario_branches_charge_six_atomic_steps(
    scenario: str,
) -> None:
    adapter = IllustrativeHallEngineeringAdapter(scenario=scenario)
    snapshot = adapter.freeze_information(0, None)
    assert snapshot.decision_time == 0
    ledger = EvaluationLedger(max_cfe=1)
    result = adapter.evaluate(
        (-500.0, 250.0),
        event_id=0,
        ledger=ledger,
        candidate_id="scenario-candidate",
    )
    assert result.objective_names == adapter.objective_names
    assert ledger.snapshot()["cfe"] == 1
    assert ledger.snapshot()["atomic_model_steps"] == 6
    identity = adapter.identity()
    assert identity["participant_data_used"] is False
    assert identity["effect_evidence"] is False
    assert identity["r7_gate"].startswith("BLOCKED_BY_R5A")


def test_infeasible_required_deficit_branch_has_no_feasible_action_at_cap() -> None:
    adapter = IllustrativeHallEngineeringAdapter(
        scenario=(
            "INFEASIBLE_REQUIRED_DEFICIT_1500_OVER_ACTION_CAP_1000_KCAL_DAY"
        )
    )
    adapter.freeze_information(0, None)
    for index, action in enumerate(((-1000.0, 0.0), (0.0, 0.0), (1000.0, 0.0))):
        ledger = EvaluationLedger(max_cfe=1)
        result = adapter.evaluate(
            action,
            event_id=0,
            ledger=ledger,
            candidate_id=f"candidate-{index}",
        )
        assert result.feasible is False
        assert result.constraints[-1] > 0.0


def test_observation_noise_is_deterministic_and_week_four_missingness_is_past_only() -> None:
    first = IllustrativeHallEngineeringAdapter(
        scenario="OBSERVATION_NOISE_GAUSSIAN_SD_0_5_KG"
    )
    second = IllustrativeHallEngineeringAdapter(
        scenario="OBSERVATION_NOISE_GAUSSIAN_SD_0_5_KG"
    )
    assert (
        first.freeze_information(0, None).information_hash
        == second.freeze_information(0, None).information_hash
    )

    missing = IllustrativeHallEngineeringAdapter(
        scenario="MISSINGNESS_EVERY_FOURTH_POSTBASELINE_WEEK"
    )
    feedback = None
    for event_id in range(4):
        missing.freeze_information(event_id, feedback)
        ledger = EvaluationLedger(max_cfe=1)
        feedback = missing.execute(
            (0.0, 0.0),
            event_id=event_id,
            committed=False,
            ledger=ledger,
        )
    week_four = missing.freeze_information(4, feedback)
    observation = week_four.fields["current_development_observation"].value
    assert observation["observation_available"] is False
    assert observation["source_week"] == 3


def test_supervisor_emits_only_redacted_deterministic_engineering_artifacts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "r6-test-output"
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--contract",
            str(DEFAULT_CONTRACT),
            "--output-root",
            str(output),
            "--test-mode",
        ],
        cwd=PROJECT_ROOT,
        env=_environment(),
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["total_cfe"] == 728
    assert summary["total_atomic_model_steps"] == 4328
    assert summary["effect_estimation_performed"] is False
    validated = validate_output(
        output,
        contract_path=DEFAULT_CONTRACT,
        allow_test_mode=True,
    )
    assert validated["paired_replay_hashes_match"] is True
    with gzip.open(
        output / "engineering_records.jsonl.gz",
        "rt",
        encoding="utf-8",
    ) as handle:
        records = [json.loads(line) for line in handle]
    serialized = json.dumps(records, sort_keys=True).lower()
    for prohibited in (
        '"objectives"',
        '"actions"',
        '"state_after"',
        '"participant_id"',
        '"effect_size"',
        '"p_value"',
        '"rank"',
    ):
        assert prohibited not in serialized

    repeated = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--contract",
            str(DEFAULT_CONTRACT),
            "--output-root",
            str(output),
            "--test-mode",
        ],
        cwd=PROJECT_ROOT,
        env=_environment(),
        capture_output=True,
        check=False,
        text=True,
    )
    assert repeated.returncode != 0
    assert "must not already exist" in repeated.stderr


def test_official_validator_rejects_test_mode_output(tmp_path: Path) -> None:
    output = tmp_path / "not-created"
    with pytest.raises(R6ValidationError, match="missing"):
        validate_output(output)


def test_private_r6_history_is_not_required_by_public_release() -> None:
    identity = {
        "git_dirty": False,
        "git_commit": "a878178fd11dd1dcc7d9f32d6f1332412ea3b88a",
        "git_tree": "cf6e300f94577618fdd26d3a4cf90d7f90ed78b7",
    }
    with pytest.raises(
        R6ValidationError,
        match="unavailable|not an ancestor",
    ):
        _validate_historical_code_identity(identity)
