from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
import pytest

import tools.validate_r5_freeze as r5_validator
from tools.validate_r5_freeze import R5ValidationError


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "r8c_e1e2"
AMENDMENT_PATH = (
    CONFIG / "r8c_e1e2_rng_implementation_amendment.json"
)
AMENDMENT_SCHEMA_PATH = (
    CONFIG / "r8c_e1e2_rng_implementation_amendment.schema.json"
)
PENDING_PATH = CONFIG / "r8c_e1e2_formal_execution_contract.json"
PENDING_SCHEMA_PATH = (
    CONFIG / "r8c_e1e2_formal_execution_contract.schema.json"
)
QUALIFIED_SCHEMA_PATH = (
    CONFIG / "r8c_e1e2_target_qualified_contract.schema.json"
)
AMENDMENT_ID = (
    "WGT-V11-R8C-E1E2-RNG-IMPLEMENTATION-AMENDMENT-01"
)
HISTORICAL_SHA256 = (
    "1ad83dc550d283a841a69f396878487c803d5283320fbede3a7357f8bb540c5c"
)
CURRENT_SHA256 = (
    "35addcdfa1dc4053fdf787925138f264c4e5c4eaf0a72c377144cb987619a29f"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validator(path: Path) -> Draft202012Validator:
    schema = _read_json(path)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def test_rng_amendment_is_strict_result_blind_and_byte_exact_bound() -> None:
    amendment = _read_json(AMENDMENT_PATH)
    _validator(AMENDMENT_SCHEMA_PATH).validate(amendment)

    historical = amendment["historical_r5_binding"]
    current = amendment["current_equivalent_implementation"]
    evidence = amendment["byte_exact_evidence"]
    assert historical == {
        "r5_contract_id": (
            "WGT-V11-R5-ENDPOINT-STATISTICS-SAMPLE-SEED-RESOURCE-01"
        ),
        "r5_contract_path": "config/r5/r5_freeze_contract.json",
        "r5_contract_sha256": (
            "4e2dd0a0f4a97b57d71dd13eb60aa8a3c3eb34f0708aae609d50a31d155f6554"
        ),
        "r4_base_commit": "c40da5960b129a4dcec0b426e556a42272f4a028",
        "r4_base_tree": "2dff77a621e8077eb52f2837eae19deaebb446b3",
        "randomness_path": "src/evaluation/randomness.py",
        "randomness_git_blob": "a97c06ed22cc32e85c3c6c6b2284ec734bd2b741",
        "randomness_sha256": HISTORICAL_SHA256,
        "randomness_bytes": 3513,
        "derivation_domain": "WGT-V11-RNG-v1",
    }
    assert current["sha256"] == CURRENT_SHA256
    assert current["derivation_domain"] == historical["derivation_domain"]
    assert sha256(
        (ROOT / current["path"]).read_bytes()
    ).hexdigest() == CURRENT_SHA256
    assert set(evidence) == {
        "test_file",
        "test_file_sha256",
        "payload",
        "bitstream",
        "engine",
        "selection",
    }
    assert evidence["payload"]["tests"]
    assert evidence["bitstream"]["tests"]
    assert evidence["engine"]["tests"] == [
        "test_engine_rng_manifests_and_outputs_match_reference_derivation"
    ]
    assert evidence["selection"]["tests"] == [
        "test_generation_selection_cache_is_byte_exact_to_old_reference"
    ]
    assert sha256(
        (ROOT / evidence["test_file"]).read_bytes()
    ).hexdigest() == evidence["test_file_sha256"]
    assert (
        amendment["authority_boundary"][
            "authorizes_formal_effect_execution"
        ]
        is False
    )
    assert (
        amendment["authority_boundary"]["authorizes_effect_analysis"]
        is False
    )


def test_r5_validator_checks_historical_blob_before_exact_amendment() -> None:
    contract = _read_json(r5_validator.CONTRACT_PATH)
    result = r5_validator._validate_randomness_history_and_amendment(
        contract
    )

    assert result == {
        "historical_sha256": HISTORICAL_SHA256,
        "current_sha256": CURRENT_SHA256,
        "amendment_sha256": sha256(
            AMENDMENT_PATH.read_bytes()
        ).hexdigest(),
    }


def test_r5_validator_rejects_any_subsequent_randomness_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drifted = tmp_path / "randomness.py"
    drifted.write_bytes(
        r5_validator.CURRENT_RANDOMNESS_PATH.read_bytes()
        + b"\n# unbound drift\n"
    )
    monkeypatch.setattr(
        r5_validator,
        "CURRENT_RANDOMNESS_PATH",
        drifted,
    )

    with pytest.raises(
        R5ValidationError,
        match="differs from the exact amendment",
    ):
        r5_validator._validate_randomness_history_and_amendment(
            _read_json(r5_validator.CONTRACT_PATH)
        )


def test_rng_amendment_schema_rejects_semantic_or_authority_drift() -> None:
    amendment = _read_json(AMENDMENT_PATH)
    validator = _validator(AMENDMENT_SCHEMA_PATH)

    domain_drift = deepcopy(amendment)
    domain_drift["current_equivalent_implementation"][
        "derivation_domain"
    ] = "WGT-V11-RNG-v2"
    assert list(validator.iter_errors(domain_drift))

    authority_drift = deepcopy(amendment)
    authority_drift["authority_boundary"][
        "authorizes_formal_effect_execution"
    ] = True
    assert list(validator.iter_errors(authority_drift))

    extra = deepcopy(amendment)
    extra["unfrozen_permission"] = True
    assert list(validator.iter_errors(extra))


def test_pending_and_qualified_schemas_require_exact_rng_amendment() -> None:
    pending = _read_json(PENDING_PATH)
    _validator(PENDING_SCHEMA_PATH).validate(pending)
    amendment_sha256 = sha256(AMENDMENT_PATH.read_bytes()).hexdigest()
    assert pending["upstream"]["rng_implementation_amendment"] == {
        "amendment_id": AMENDMENT_ID,
        "path": (
            "config/r8c_e1e2/"
            "r8c_e1e2_rng_implementation_amendment.json"
        ),
        "sha256": amendment_sha256,
    }

    qualified_schema = _read_json(QUALIFIED_SCHEMA_PATH)
    Draft202012Validator.check_schema(qualified_schema)
    upstream = qualified_schema["properties"]["upstream"]
    assert "rng_implementation_amendment" in upstream["required"]
    assert upstream["properties"]["rng_implementation_amendment"] == {
        "$ref": "#/$defs/rng_implementation_amendment_upstream"
    }
    binding = qualified_schema["$defs"][
        "rng_implementation_amendment_upstream"
    ]
    assert binding["additionalProperties"] is False
    assert binding["properties"]["amendment_id"]["const"] == AMENDMENT_ID
    assert binding["properties"]["sha256"]["const"] == amendment_sha256
