from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


CODE_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = CODE_ROOT.parent / "文字稿-期刊"
VALIDATOR_ROOT = PAPER_ROOT / "项目工作区" / "05_工具与验证"
PROTOCOL_VERSION_PATH = (
    PAPER_ROOT / "项目工作区" / "01_协议与门禁" / "protocol_version.txt"
)

pytestmark = pytest.mark.skipif(
    not PAPER_ROOT.is_dir(),
    reason="journal protocol workspace is not present in source-only clone",
)


def _expected_r2_status() -> str:
    protocol_version = PROTOCOL_VERSION_PATH.read_text(encoding="utf-8")
    if "version=v1.1.8-r2-shade-success-frozen" in protocol_version:
        return "R2_STATUS=SHADE_SUCCESS_FROZEN_IMPLEMENTATION_IN_PROGRESS"
    if "version=v1.1.9-r2-shade-reclosed" in protocol_version:
        return "R2_STATUS=RECLOSED_F22_T13_SHADE_VERIFIED"
    if "version=v1.2.0-r3-v11mq1-frozen" in protocol_version:
        return "R2_STATUS=RECLOSED_F22_T13_SHADE_VERIFIED"
    if "version=v1.2.1-r3-v11mq1-launcher-reopened" in protocol_version:
        return "R2_STATUS=RECLOSED_F22_T13_SHADE_VERIFIED"
    if "version=v1.2.2-r3-v11mq1-launcher-corrective-ready" in protocol_version:
        return "R2_STATUS=RECLOSED_F22_T13_SHADE_VERIFIED"
    if "version=v1.2.3-r3-v11mq1-point-model-not-qualified" in protocol_version:
        return "R2_STATUS=RECLOSED_F22_T13_SHADE_VERIFIED"
    if "version=v1.2.4-r4-executable-bindings-closed" in protocol_version:
        return "R2_STATUS=RECLOSED_F22_T13_SHADE_VERIFIED"
    if "version=v1.3.0-r5-result-blind-design-frozen" in protocol_version:
        return "R2_STATUS=RECLOSED_F22_T13_SHADE_VERIFIED"
    if "version=v1.4.0-r6-result-blind-engineering-pilot-closed" in protocol_version:
        return "R2_STATUS=RECLOSED_F22_T13_SHADE_VERIFIED"
    if "version=v1.5.0-r5a-e3-input-contract-frozen" in protocol_version:
        return "R2_STATUS=RECLOSED_F22_T13_SHADE_VERIFIED"
    if (
        "version=v1.6.0-r7-formal-execution-authorization-frozen"
        in protocol_version
    ):
        return "R2_STATUS=RECLOSED_F22_T13_SHADE_VERIFIED"
    raise AssertionError("unexpected paper protocol version")


def _expected_r3_status() -> tuple[str, str]:
    protocol_version = PROTOCOL_VERSION_PATH.read_text(encoding="utf-8")
    if "version=v1.2.1-r3-v11mq1-launcher-reopened" in protocol_version:
        return (
            "R3=TECHNICAL_READINESS_REOPENED_PRECONSUMPTION_LAUNCH_FAILURE",
            "A1_PARTICIPANT_ACCESS=false_EXECUTION_IDENTITY_NOT_CONSUMED",
        )
    if "version=v1.2.2-r3-v11mq1-launcher-corrective-ready" in protocol_version:
        return (
            "R3=LAUNCHER_CORRECTIVE_READY_NEW_EXACT_COMMAND_PENDING",
            "A1_PARTICIPANT_ACCESS=false_PENDING_NEW_EXACT_COMMAND",
        )
    if any(
        version in protocol_version
        for version in (
            "version=v1.2.3-r3-v11mq1-point-model-not-qualified",
            "version=v1.2.4-r4-executable-bindings-closed",
            "version=v1.3.0-r5-result-blind-design-frozen",
            "version=v1.4.0-r6-result-blind-engineering-pilot-closed",
            "version=v1.5.0-r5a-e3-input-contract-frozen",
            "version=v1.6.0-r7-formal-execution-authorization-frozen",
        )
    ):
        return (
            "R3=CLOSED_POINT_MODEL_NOT_QUALIFIED",
            "A1_PARTICIPANT_OUTPUT=AGGREGATE_ONLY_NO_RAW_IDENTIFIERS",
        )
    return (
        "R3=TECHNICAL_READINESS_CLOSED_V11_MQ1_NOT_EXECUTED",
        "A1_PARTICIPANT_EXECUTION=false_PENDING_EXACT_COMMAND",
    )


@pytest.mark.parametrize(
    "script",
    [
        "validate_pre_freeze.py",
        "validate_v11_r1_proposal.py",
    ],
)
def test_protocol_validators_preserve_r1_and_accept_result_blind_shade_freeze(
    script: str,
) -> None:
    completed = subprocess.run(
        [sys.executable, str(VALIDATOR_ROOT / script)],
        cwd=PAPER_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "R1_HISTORY=VERIFIED" in completed.stdout
    assert _expected_r2_status() in completed.stdout
    r3_status, a1_status = _expected_r3_status()
    assert r3_status in completed.stdout
    assert a1_status in completed.stdout
    protocol_version = PROTOCOL_VERSION_PATH.read_text(encoding="utf-8")
    if any(
        version in protocol_version
        for version in (
            "version=v1.2.4-r4-executable-bindings-closed",
            "version=v1.3.0-r5-result-blind-design-frozen",
            "version=v1.4.0-r6-result-blind-engineering-pilot-closed",
            "version=v1.5.0-r5a-e3-input-contract-frozen",
            "version=v1.6.0-r7-formal-execution-authorization-frozen",
        )
    ):
        assert (
            "R4=CLOSED_EXECUTABLE_BINDINGS_NO_EFFECT_ESTIMATION"
            in completed.stdout
        )
    else:
        assert "R4_AUTHORIZED=false" in completed.stdout
    if any(
        version in protocol_version
        for version in (
            "version=v1.3.0-r5-result-blind-design-frozen",
            "version=v1.4.0-r6-result-blind-engineering-pilot-closed",
            "version=v1.5.0-r5a-e3-input-contract-frozen",
            "version=v1.6.0-r7-formal-execution-authorization-frozen",
        )
    ):
        assert (
            "R5=CLOSED_RESULT_BLIND_ENDPOINT_STATISTICS_SAMPLE_SEED_CFE_RESOURCE_FREEZE"
            in completed.stdout
        )
    if any(
        version in protocol_version
        for version in (
            "version=v1.4.0-r6-result-blind-engineering-pilot-closed",
            "version=v1.5.0-r5a-e3-input-contract-frozen",
            "version=v1.6.0-r7-formal-execution-authorization-frozen",
        )
    ):
        assert (
            "R6=CLOSED_ISOLATED_RESULT_BLIND_ENGINEERING_PILOT"
            in completed.stdout
        )
    if any(
        version in protocol_version
        for version in (
            "version=v1.5.0-r5a-e3-input-contract-frozen",
            "version=v1.6.0-r7-formal-execution-authorization-frozen",
        )
    ):
        assert (
            "R5A=CLOSED_RESULT_BLIND_E3_INPUT_GENERATOR_TARGET_SCENARIO_"
            "VALIDATION_FREEZE"
            in completed.stdout
        )
    if (
        "version=v1.6.0-r7-formal-execution-authorization-frozen"
        in protocol_version
    ):
        assert (
            "R7=CLOSED_FORMAL_EXECUTION_CONTRACT_AND_R8_COMMAND_FREEZE"
            in completed.stdout
        )
        assert "R8=PENDING_VERBATIM_AUTHOR_CONFIRMATION" in completed.stdout
    elif "version=v1.5.0-r5a-e3-input-contract-frozen" in protocol_version:
        assert "R7=PENDING_SEPARATE_AUTHORIZATION" in completed.stdout
