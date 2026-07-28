from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import pandas as pd
import pytest
from jsonschema import Draft202012Validator

from weight_application.model_qualification import (
    ID_DOMAIN,
    NONPASS_CASE_NAME,
    PASS_CASE_NAME,
    SOURCE_TABLE,
    QualificationRecord,
    QualificationThresholds,
    build_qualification_records_from_rows,
    evaluate_model_qualification,
    load_pride_archive,
    month_to_model_day,
    pseudonymize_identifier,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = PROJECT_ROOT.parent / "文字稿-期刊"
GATE_ROOT = PAPER_ROOT / "项目工作区" / "09_门禁准备"


def load_runner_module() -> object:
    path = PROJECT_ROOT / "tools" / "run_v11_mq1_qualification.py"
    spec = importlib.util.spec_from_file_location("v11_mq1_runner", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def participant_rows(
    participant: object = "A",
    *,
    diet_change: float = 0.0,
    activity_change: float = 0.0,
) -> list[dict[str, object]]:
    rows = []
    for visit in (0.0, 6.0, 12.0, 18.0):
        rows.append(
            {
                "ID": participant,
                "NVISIT": visit,
                "RAGE": 55.0,
                "HEIGHT": 1.65,
                "WEIGHT0": 90.0 if visit == 0.0 else None,
                "WEIGHT": 90.0,
                "DT_KCAL": 2200.0
                + (0.0 if visit == 0.0 else diet_change),
                "kcal": 1400.0
                + (0.0 if visit == 0.0 else activity_change),
            }
        )
    return rows


def metric_rows(
    participants: int,
    *,
    error_kg: float,
) -> list[QualificationRecord]:
    return [
        QualificationRecord(
            participant_id=str(participant),
            visit_month=month,
            baseline_weight_kg=100.0,
            observed_weight_kg=observed,
            predicted_weight_kg=observed + error_kg,
        )
        for participant in range(participants)
        for month, observed in ((6.0, 90.0), (12.0, 88.0))
    ]


def small_thresholds(participants: int) -> QualificationThresholds:
    return QualificationThresholds(
        minimum_participants=participants,
        bootstrap_replicates=500,
        bootstrap_seed=17,
    )


def test_identifier_bytes_are_hashed_without_decoding() -> None:
    raw = b"\x85\xffprivate"
    expected = hashlib.sha256(ID_DOMAIN + b"B\x00" + raw).hexdigest()
    assert pseudonymize_identifier(raw) == expected
    assert pseudonymize_identifier(raw) != pseudonymize_identifier(
        raw.decode("latin-1")
    )


def test_sas_loader_tripwire_requires_encoding_none_and_preserves_bytes(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "synthetic-pride.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(SOURCE_TABLE, b"synthetic-not-real-sas")
    frame = pd.DataFrame(participant_rows(b"\x85\xffprivate"))
    touched = {"called": False}

    def read_sas_tripwire(
        payload: object,
        *,
        format: str,
        encoding: object,
    ) -> pd.DataFrame:
        touched["called"] = True
        assert payload is not None
        assert format == "sas7bdat"
        assert encoding is None
        assert isinstance(frame.loc[0, "ID"], bytes)
        return frame

    output = load_pride_archive(
        archive_path,
        read_sas=read_sas_tripwire,
    )
    assert touched["called"] is True
    assert len(output.records) == 3
    assert b"\x85\xffprivate".hex() not in str(output.audit)
    assert output.audit["raw_identifier_serialized"] is False


def test_postbaseline_outcomes_never_change_predictions() -> None:
    first = participant_rows()
    second = participant_rows()
    for row in second:
        if float(row["NVISIT"]) > 0.0:
            row["WEIGHT"] = 60.0 + float(row["NVISIT"])
    first_output = build_qualification_records_from_rows(first)
    second_output = build_qualification_records_from_rows(second)
    assert [
        record.predicted_weight_kg for record in first_output.records
    ] == [
        record.predicted_weight_kg for record in second_output.records
    ]
    assert (
        first_output.canonical_input_sha256
        != second_output.canonical_input_sha256
    )


def test_direct_diet_and_activity_exposure_have_expected_direction() -> None:
    diet = build_qualification_records_from_rows(
        participant_rows(diet_change=-300.0)
    )
    activity = build_qualification_records_from_rows(
        participant_rows(activity_change=2100.0)
    )
    neutral = build_qualification_records_from_rows(participant_rows())
    assert (
        diet.records[-1].predicted_weight_kg
        < neutral.records[-1].predicted_weight_kg
    )
    assert (
        activity.records[-1].predicted_weight_kg
        < neutral.records[-1].predicted_weight_kg
    )


def test_missing_exposure_stops_trajectory_without_interpolation() -> None:
    rows = participant_rows()
    rows[2]["DT_KCAL"] = None
    output = build_qualification_records_from_rows(rows)
    assert output.records == ()
    assert output.audit["exclusion_counts"] == {
        "missing_exposure_stopped_trajectory_before_two_visits": 1
    }


def test_month_conversion_uses_frozen_half_up_quarter_days() -> None:
    assert month_to_model_day(6.0) == 182.75
    assert month_to_model_day(12.0) == 365.25
    assert month_to_model_day(18.0) == 548.0


def test_simultaneous_decisions_and_case_names() -> None:
    passed = evaluate_model_qualification(
        metric_rows(10, error_kg=0.0),
        small_thresholds(10),
    )
    assert passed["decision"] == "MODEL_QUALIFICATION_PASSED"
    assert passed["case_name"] == PASS_CASE_NAME
    assert passed["pass"] is True

    failed = evaluate_model_qualification(
        metric_rows(10, error_kg=5.0),
        small_thresholds(10),
    )
    assert failed["decision"] == "MODEL_QUALIFICATION_FAILED"
    assert failed["case_name"] == NONPASS_CASE_NAME
    assert failed["checks"]["mae"] is False

    insufficient = evaluate_model_qualification(
        metric_rows(3, error_kg=0.0),
        small_thresholds(4),
    )
    assert insufficient["decision"] == "MODEL_INPUT_INSUFFICIENT"
    assert insufficient["case_name"] == NONPASS_CASE_NAME


def test_duplicate_visit_is_invalid_and_bootstrap_is_reproducible() -> None:
    duplicate = metric_rows(4, error_kg=0.0)
    duplicate.append(duplicate[0])
    invalid = evaluate_model_qualification(
        duplicate,
        small_thresholds(4),
    )
    assert invalid["decision"] == "QUALIFICATION_INPUT_INVALID"

    records = metric_rows(10, error_kg=0.25)
    first = evaluate_model_qualification(records, small_thresholds(10))
    second = evaluate_model_qualification(records, small_thresholds(10))
    assert first == second


def test_identifier_errors_do_not_echo_raw_value() -> None:
    with pytest.raises(ValueError) as captured:
        pseudonymize_identifier("")
    assert "PRIVATE" not in str(captured.value)


def test_execution_request_requires_separate_exact_command_confirmation() -> None:
    if not GATE_ROOT.is_dir():
        pytest.skip("journal gate workspace is not present in source-only clone")
    schema = json.loads(
        (
            GATE_ROOT / "v11_mq1_execution_request.schema.json"
        ).read_text(encoding="utf-8")
    )
    request = {
        "request_id": "WGT-V11-MQ1-REQUEST-20260724-01",
        "execution_id": "WGT-V11-MQ1-EXECUTION-20260724-01",
        "contract_id": "WGT-V11-MQ1-MODEL-QUALIFICATION-01",
        "protocol_version": "v1.2.0-r3-v11mq1-frozen",
        "archive_path": "E:/qualified/input.zip",
        "archive_sha256": "0" * 64,
        "implementation_commit": "0" * 40,
        "implementation_tree": "0" * 40,
        "qualification_lock_sha256": "0" * 64,
        "contract_sha256": "0" * 64,
        "authorization_record_sha256": "0" * 64,
        "result_path": "E:/qualified/result.json",
        "consumption_path": "E:/qualified/consumption.json",
        "author_exact_command_confirmed": False,
    }
    errors = list(Draft202012Validator(schema).iter_errors(request))
    assert errors
    assert any("True was expected" in error.message for error in errors)


def test_runner_selects_schema_by_frozen_request_identity() -> None:
    runner = load_runner_module()
    first = runner._request_schema_path(
        GATE_ROOT,
        {"request_id": "WGT-V11-MQ1-REQUEST-20260724-01"},
    )
    second = runner._request_schema_path(
        GATE_ROOT,
        {"request_id": "WGT-V11-MQ1-REQUEST-20260724-02"},
    )
    assert first.name == "v11_mq1_execution_request.schema.json"
    assert second.name == "v11_mq1_execution_request_02.schema.json"
    with pytest.raises(RuntimeError, match="unrecognized execution request"):
        runner._request_schema_path(
            GATE_ROOT,
            {"request_id": "WGT-V11-MQ1-REQUEST-UNKNOWN"},
        )


def test_launcher_defers_scientific_package_import_until_after_environment() -> None:
    source = (
        PROJECT_ROOT / "tools" / "run_v11_mq1_qualification.py"
    ).read_text(encoding="utf-8")
    environment_validation = source.index(
        "_validate_corrective_environment(\n        request,"
    )
    package_import = source.index(
        "from weight_application.model_qualification import (",
        environment_validation,
    )
    archive_resolution = source.index(
        '_absolute_file(request["archive_path"], "A1 archive")'
    )
    assert environment_validation < package_import < archive_resolution


def test_runner_consumes_identity_before_archive_parser_is_called() -> None:
    source = (
        PROJECT_ROOT / "tools" / "run_v11_mq1_qualification.py"
    ).read_text(encoding="utf-8")
    consumption_write = source.index(
        "_atomic_write(consumption_path, consumption)"
    )
    archive_parse = source.index(
        "built = load_pride_archive(archive_path)"
    )
    assert consumption_write < archive_parse


def test_execution_failure_is_distinct_and_result_schema_forbids_rows() -> None:
    if not GATE_ROOT.is_dir():
        pytest.skip("journal gate workspace is not present in source-only clone")
    schema = json.loads(
        (
            GATE_ROOT / "v11_mq1_qualification_result.schema.json"
        ).read_text(encoding="utf-8")
    )
    result = {
        "record_id": "WGT-V11-MQ1-RESULT-20260724-01",
        "execution_id": "WGT-V11-MQ1-EXECUTION-20260724-01",
        "contract_id": "WGT-V11-MQ1-MODEL-QUALIFICATION-01",
        "protocol_version": "v1.2.0-r3-v11mq1-frozen",
        "implementation": {
            "commit": "0" * 40,
            "tree": "0" * 40,
            "git_dirty": False,
            "qualification_lock_sha256": "0" * 64,
        },
        "input": {
            "archive_sha256": "0" * 64,
            "canonical_qualification_input_sha256": None,
            "source_table": SOURCE_TABLE,
        },
        "decision": "QUALIFICATION_EXECUTION_FAILED",
        "pass": False,
        "case_name": NONPASS_CASE_NAME,
        "eligible_participants": 0,
        "eligible_postbaseline_records": 0,
        "metrics": None,
        "checks": None,
        "reason": "aggregate audit required",
        "prediction_interval_status": (
            "NOT_QUALIFIED_NO_INDEPENDENT_CALIBRATION_SET"
        ),
        "audit": {
            "source_rows": 0,
            "source_participants": 0,
            "exclusion_counts": {},
            "postbaseline_outcome_used_for_prediction": False,
            "calibration_performed": False,
            "model_selection_performed": False,
            "threshold_changed": False,
            "raw_identifier_serialized": False,
        },
        "result_knowledge": {
            "effect_estimation_performed": False,
            "algorithm_or_comparator_selected": False,
            "participant_values_reported": False,
        },
    }
    validator = Draft202012Validator(schema)
    assert list(validator.iter_errors(result)) == []
    result["participant_rows"] = [{"ID": "forbidden"}]
    assert list(validator.iter_errors(result))
