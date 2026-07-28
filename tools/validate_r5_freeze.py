"""Validate the R5 result-blind design freeze without running an experiment."""

from __future__ import annotations

import argparse
import ast
from hashlib import sha256
import hmac
import json
from pathlib import Path
import subprocess
from typing import Any

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
R5_CONFIG = ROOT / "config" / "r5"
CONTRACT_PATH = R5_CONFIG / "r5_freeze_contract.json"
SCHEMA_PATH = R5_CONFIG / "r5_freeze_contract.schema.json"
RNG_AMENDMENT_PATH = (
    ROOT
    / "config"
    / "r8c_e1e2"
    / "r8c_e1e2_rng_implementation_amendment.json"
)
RNG_AMENDMENT_SCHEMA_PATH = (
    ROOT
    / "config"
    / "r8c_e1e2"
    / "r8c_e1e2_rng_implementation_amendment.schema.json"
)
CURRENT_RANDOMNESS_PATH = ROOT / "src" / "evaluation" / "randomness.py"
PAIRED_SEED_DOMAIN = b"WGT-V11-R5-PAIRED-MASTER-v1\0"
SUBJECT_SEED_DOMAIN = b"WGT-V11-R5-E3-PUBLIC-SUBJECT-v1\0"
ROLLING_SEED_DOMAIN = b"WGT-F23-SEED-v1\x00"


class R5ValidationError(RuntimeError):
    """An R5 design artifact is incomplete, inconsistent, or result-aware."""


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _validate_schema() -> dict[str, Any]:
    contract = _load_json(CONTRACT_PATH)
    schema = _load_json(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(contract)
    return contract


def _git_bytes(*arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), *arguments],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise R5ValidationError(
            "R5 historical Git identity could not be resolved"
        ) from error


def _git_text(*arguments: str) -> str:
    try:
        return _git_bytes(*arguments).decode("ascii").strip()
    except UnicodeError as error:
        raise R5ValidationError(
            "R5 historical Git identity is not ASCII"
        ) from error


def _derivation_domain(source: bytes, *, label: str) -> str:
    try:
        module = ast.parse(source.decode("utf-8"))
    except (UnicodeError, SyntaxError) as error:
        raise R5ValidationError(
            f"{label} randomness source cannot be parsed"
        ) from error
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name)
            and target.id == "DERIVATION_DOMAIN"
            for target in node.targets
        ):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError) as error:
            raise R5ValidationError(
                f"{label} randomness domain is not literal"
            ) from error
        if not isinstance(value, str):
            raise R5ValidationError(
                f"{label} randomness domain is not a string"
            )
        return value
    raise R5ValidationError(f"{label} randomness domain is missing")


def _validate_rng_amendment(
    contract: dict[str, Any],
    *,
    historical_blob_id: str,
    historical_blob: bytes,
) -> dict[str, str]:
    for path, label in (
        (RNG_AMENDMENT_PATH, "RNG implementation amendment"),
        (RNG_AMENDMENT_SCHEMA_PATH, "RNG amendment schema"),
        (CURRENT_RANDOMNESS_PATH, "current randomness module"),
    ):
        if not path.is_file():
            raise R5ValidationError(f"{label} is missing")

    amendment = _load_json(RNG_AMENDMENT_PATH)
    schema = _load_json(RNG_AMENDMENT_SCHEMA_PATH)
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(amendment)
    except Exception as error:
        raise R5ValidationError(
            "RNG implementation amendment schema validation failed"
        ) from error

    upstream = contract["upstream_bindings"]
    historical = amendment["historical_r5_binding"]
    binding = upstream["randomness_module"]
    expected_historical = {
        "r5_contract_id": contract["contract_id"],
        "r5_contract_path": "config/r5/r5_freeze_contract.json",
        "r5_contract_sha256": sha256(CONTRACT_PATH.read_bytes()).hexdigest(),
        "r4_base_commit": upstream["r4_base_commit"],
        "r4_base_tree": upstream["r4_base_tree"],
        "randomness_path": binding["path"],
        "randomness_git_blob": historical_blob_id,
        "randomness_sha256": binding["sha256"],
        "randomness_bytes": len(historical_blob),
        "derivation_domain": binding["derivation_domain"],
    }
    if historical != expected_historical:
        raise R5ValidationError(
            "RNG amendment historical R5 binding differs"
        )

    current = amendment["current_equivalent_implementation"]
    if current["path"] != binding["path"]:
        raise R5ValidationError(
            "RNG amendment current path differs from R5"
        )
    current_bytes = CURRENT_RANDOMNESS_PATH.read_bytes()
    current_sha256 = sha256(current_bytes).hexdigest()
    if current_sha256 == binding["sha256"]:
        raise R5ValidationError(
            "RNG amendment is present although the current file did not drift"
        )
    if (
        current["sha256"] != current_sha256
        or current["bytes"] != len(current_bytes)
    ):
        raise R5ValidationError(
            "current randomness drift differs from the exact amendment"
        )
    historical_domain = _derivation_domain(
        historical_blob,
        label="historical",
    )
    current_domain = _derivation_domain(
        current_bytes,
        label="current",
    )
    if {
        historical_domain,
        current_domain,
        current["derivation_domain"],
        amendment["unchanged_rng_contract"]["derivation_domain"],
    } != {binding["derivation_domain"]}:
        raise R5ValidationError(
            "randomness derivation domain changed across the amendment"
        )

    evidence = amendment["byte_exact_evidence"]
    evidence_path = ROOT / str(evidence["test_file"])
    if not evidence_path.is_file():
        raise R5ValidationError("RNG byte-exact evidence file is missing")
    if sha256(evidence_path.read_bytes()).hexdigest() != (
        evidence["test_file_sha256"]
    ):
        raise R5ValidationError("RNG byte-exact evidence file drifted")
    if any(
        amendment["authority_boundary"][key]
        for key in (
            "changes_seed_contract",
            "changes_derivation_domain",
            "changes_random_draws",
            "changes_algorithm",
            "changes_endpoints_samples_or_CFE",
            "authorizes_formal_effect_execution",
            "authorizes_effect_analysis",
        )
    ):
        raise R5ValidationError(
            "RNG amendment exceeds result-blind implementation authority"
        )
    return {
        "historical_sha256": binding["sha256"],
        "current_sha256": current_sha256,
        "amendment_sha256": sha256(
            RNG_AMENDMENT_PATH.read_bytes()
        ).hexdigest(),
    }


def _validate_randomness_history_and_amendment(
    contract: dict[str, Any],
) -> dict[str, str]:
    upstream = contract["upstream_bindings"]
    commit = str(upstream["r4_base_commit"])
    expected_tree = str(upstream["r4_base_tree"])
    binding = upstream["randomness_module"]
    if _git_text("rev-parse", f"{commit}^{{commit}}") != commit:
        raise R5ValidationError("R5 r4 base commit identity differs")
    if _git_text("rev-parse", f"{commit}^{{tree}}") != expected_tree:
        raise R5ValidationError("R5 r4 base tree identity differs")
    historical_spec = f"{commit}:{binding['path']}"
    historical_blob_id = _git_text("rev-parse", historical_spec)
    historical_blob = _git_bytes(
        "cat-file",
        "blob",
        historical_blob_id,
    )
    if sha256(historical_blob).hexdigest() != binding["sha256"]:
        raise R5ValidationError(
            "historical randomness blob differs from the R5 binding"
        )
    if _derivation_domain(
        historical_blob,
        label="historical",
    ) != binding["derivation_domain"]:
        raise R5ValidationError(
            "historical randomness derivation domain differs from R5"
        )
    return _validate_rng_amendment(
        contract,
        historical_blob_id=historical_blob_id,
        historical_blob=historical_blob,
    )


def _validate_upstream_hashes(
    contract: dict[str, Any],
) -> dict[str, str]:
    upstream = contract["upstream_bindings"]
    for key in (
        "r4_benchmark_registry",
        "r4_comparator_manifest",
        "dependency_lock",
    ):
        binding = upstream[key]
        path = ROOT / str(binding["path"])
        if not path.is_file():
            raise R5ValidationError(f"upstream binding is missing: {key}")
        if sha256(path.read_bytes()).hexdigest() != binding["sha256"]:
            raise R5ValidationError(f"upstream binding hash differs: {key}")
    return _validate_randomness_history_and_amendment(contract)


def _validate_paired_master_seeds(contract: dict[str, Any]) -> None:
    seed_contract = contract["seed_contract"]
    expected = [
        int.from_bytes(
            sha256(PAIRED_SEED_DOMAIN + str(index).encode("ascii")).digest()[:8],
            "big",
        )
        for index in range(10)
    ]
    recorded = [
        int(value) for value in seed_contract["paired_master_seeds_u64"]
    ]
    if recorded != expected:
        raise R5ValidationError("paired master seed table differs")
    if _canonical_sha256(expected) != (
        seed_contract["paired_master_seed_table_sha256"]
    ):
        raise R5ValidationError("paired master seed commitment differs")
    if len(set(recorded)) != len(recorded):
        raise R5ValidationError("paired master seeds are not unique")
    if max(seed_contract["use_prefix_counts"].values()) > len(recorded):
        raise R5ValidationError("a paired-seed prefix exceeds the frozen table")


def _rolling_seed(
    master_seed: bytes,
    *,
    template: str,
    index: int,
) -> int:
    message = (
        ROLLING_SEED_DOMAIN
        + f"development|WGT-RR-CMOP|{template}|{index}".encode("ascii")
    )
    return int.from_bytes(
        hmac.new(master_seed, message, sha256).digest()[:8],
        "big",
    )


def _validate_rolling_instances(contract: dict[str, Any]) -> None:
    seed_contract = contract["seed_contract"]
    generator = seed_contract["rolling_public_generator"]
    master_seed = bytes.fromhex(generator["public_master_seed_hex"])
    expected = [
        {
            "template": template,
            "index": index,
            "derived_seed_u64": _rolling_seed(
                master_seed,
                template=template,
                index=index,
            ),
        }
        for template in generator["templates"]
        for index in generator["indices_per_template"]
    ]
    recorded = [
        {
            "template": str(item["template"]),
            "index": int(item["index"]),
            "derived_seed_u64": int(item["derived_seed_u64"]),
        }
        for item in seed_contract["rolling_public_instances"]
    ]
    if recorded != expected:
        raise R5ValidationError("rolling public instance seed table differs")
    if len(recorded) != generator["instance_count"]:
        raise R5ValidationError("rolling public instance count differs")
    if _canonical_sha256(expected) != generator["instance_table_sha256"]:
        raise R5ValidationError("rolling public instance commitment differs")
    if generator["hidden"] is not False:
        raise R5ValidationError("R5 rolling sample opened hidden instances")


def _validate_e3_subjects(contract: dict[str, Any]) -> None:
    seed_contract = contract["seed_contract"]
    expected = [
        {
            "subject_id": f"VS-{index:03d}",
            "seed_u64": int.from_bytes(
                sha256(
                    SUBJECT_SEED_DOMAIN + str(index).encode("ascii")
                ).digest()[:8],
                "big",
            ),
        }
        for index in range(32)
    ]
    recorded = [
        {
            "subject_id": str(item["subject_id"]),
            "seed_u64": int(item["seed_u64"]),
        }
        for item in seed_contract["e3_public_subjects"]
    ]
    if recorded != expected:
        raise R5ValidationError("E3 public subject seed table differs")
    if _canonical_sha256(expected) != (
        seed_contract["e3_public_subject_table_sha256"]
    ):
        raise R5ValidationError("E3 public subject commitment differs")


def _expected_workloads(contract: dict[str, Any]) -> list[dict[str, int | str]]:
    design = contract["experiment_design"]
    static = design["E1_STATIC"]
    dynamic = design["E1_DYNAMIC"]
    rolling = design["E1_ROLLING"]
    e2_dynamic = design["E2_DYNAMIC"]
    e2_rolling = design["E2_ROLLING"]
    e3 = design["E3"]

    static_sequences = (
        static["top_level_unit_count"]
        * static["paired_replicates"]
        * len(static["methods"])
    )
    dynamic_sequences = (
        dynamic["top_level_unit_count"]
        * len(dynamic["profiles"])
        * dynamic["paired_replicates"]
        * len(dynamic["methods"])
    )
    rolling_sequences = (
        rolling["top_level_unit_count"]
        * rolling["paired_replicates"]
        * len(rolling["methods"])
    )
    e2_dynamic_incremental_sequences = (
        e2_dynamic["top_level_unit_count"]
        * len(e2_dynamic["profiles"])
        * e2_dynamic["paired_replicates"]
        * (len(e2_dynamic["methods"]) - 1)
    )
    e2_rolling_incremental_sequences = (
        e2_rolling["top_level_unit_count"]
        * e2_rolling["paired_replicates"]
        * (len(e2_rolling["methods"]) - 1)
    )
    stochastic_e3_methods = 2
    deterministic_e3_methods = 1
    e3_sequences = e3["top_level_unit_count"] * (
        stochastic_e3_methods * e3["paired_replicates"]
        + deterministic_e3_methods
    )

    def standard(
        workload_id: str,
        sequences: int,
        *,
        events: int,
        cfe_per_event: int,
        atomic_steps: int,
    ) -> dict[str, int | str]:
        cfe = sequences * events * cfe_per_event
        return {
            "id": workload_id,
            "method_sequences": sequences,
            "CFE": cfe,
            "atomic_model_steps": cfe * atomic_steps,
        }

    e3_cfe = e3["top_level_unit_count"] * e3["events"] * (
        stochastic_e3_methods
        * e3["paired_replicates"]
        * e3["cfe_per_event_stochastic_search"]
        + deterministic_e3_methods * e3["cfe_per_event_deterministic_policy"]
    )
    return [
        standard(
            "E1_STATIC",
            static_sequences,
            events=static["events"],
            cfe_per_event=static["cfe_per_event"],
            atomic_steps=static["atomic_steps_per_cfe"],
        ),
        standard(
            "E1_DYNAMIC",
            dynamic_sequences,
            events=dynamic["events"],
            cfe_per_event=dynamic["cfe_per_event"],
            atomic_steps=dynamic["atomic_steps_per_cfe"],
        ),
        standard(
            "E1_ROLLING",
            rolling_sequences,
            events=rolling["events"],
            cfe_per_event=rolling["cfe_per_event"],
            atomic_steps=rolling["atomic_steps_per_cfe"],
        ),
        standard(
            "E2_DYNAMIC_INCREMENTAL_AFTER_FULL_REUSE",
            e2_dynamic_incremental_sequences,
            events=e2_dynamic["events"],
            cfe_per_event=e2_dynamic["cfe_per_event"],
            atomic_steps=e2_dynamic["atomic_steps_per_cfe"],
        ),
        standard(
            "E2_ROLLING_INCREMENTAL_AFTER_FULL_REUSE",
            e2_rolling_incremental_sequences,
            events=e2_rolling["events"],
            cfe_per_event=e2_rolling["cfe_per_event"],
            atomic_steps=e2_rolling["atomic_steps_per_cfe"],
        ),
        {
            "id": "E3",
            "method_sequences": e3_sequences,
            "CFE": e3_cfe,
            "atomic_model_steps": e3_cfe * e3["atomic_steps_per_cfe"],
        },
    ]


def _validate_workload(contract: dict[str, Any]) -> None:
    expected = _expected_workloads(contract)
    workload = contract["workload_budget"]
    if workload["unique_workloads"] != expected:
        raise R5ValidationError("frozen workload table differs from design")
    if workload["total_unique_method_sequences"] != sum(
        int(item["method_sequences"]) for item in expected
    ):
        raise R5ValidationError("total method-sequence count differs")
    if workload["total_CFE"] != sum(int(item["CFE"]) for item in expected):
        raise R5ValidationError("total CFE differs")
    if workload["total_atomic_model_steps"] != sum(
        int(item["atomic_model_steps"]) for item in expected
    ):
        raise R5ValidationError("total atomic-step budget differs")


def _validate_statistics_and_permissions(contract: dict[str, Any]) -> None:
    checkpoints = contract["common_configuration"]["checkpoint_fractions"]
    expected_checkpoints = [index / 20.0 for index in range(21)]
    if checkpoints != expected_checkpoints:
        raise R5ValidationError("21-point checkpoint grid differs")
    transfer = contract["common_configuration"][
        "transfer_checkpoint_fractions"
    ]
    if transfer != expected_checkpoints[:5]:
        raise R5ValidationError("early-transfer checkpoint grid differs")

    families = contract["statistics_contract"]["multiplicity"]["families"]
    counts = {item["family_id"]: item["hypothesis_count"] for item in families}
    if counts != {
        "E1_PRIMARY_ANYTIME": 15,
        "E2_DYNAMIC_TRANSFER": 7,
        "E2_ROLLING_TRANSFER": 8,
    }:
        raise R5ValidationError("multiplicity family counts differ")
    if contract["statistics_contract"]["p_value_only_conclusion_allowed"]:
        raise R5ValidationError("R5 allowed a p-value-only conclusion")
    if contract["endpoint_contract"]["nhv"][
        "observed_method_union_reference_allowed"
    ]:
        raise R5ValidationError("R5 allowed an observed-method reference")

    permissions = contract["permissions"]
    allowed_true = {
        "r5_contract_validation_allowed",
        "synthetic_known_answer_tests_allowed",
    }
    for key, value in permissions.items():
        expected = key in allowed_true
        if value is not expected:
            raise R5ValidationError(f"permission differs: {key}")
    if contract["next_gate"]["authorized"] is not False:
        raise R5ValidationError("R5 automatically authorized R6")

    resources = contract["resource_budget"]
    if resources["parallelism"]["max_workers"] > (
        resources["freeze_hardware"]["physical_cores"]
    ):
        raise R5ValidationError("worker count exceeds physical cores")
    if resources["parallelism"]["max_pool_peak_rss_gib"] > 24:
        raise R5ValidationError("pool RSS exceeds the frozen ceiling")
    if resources["scratch"]["onedrive_path_allowed"] is not False:
        raise R5ValidationError("R5 allowed OneDrive execution scratch")
    if resources["output"]["silent_truncation_allowed"] is not False:
        raise R5ValidationError("R5 allowed silent raw-output truncation")


def validate_r5() -> dict[str, Any]:
    contract = _validate_schema()
    _validate_upstream_hashes(contract)
    _validate_paired_master_seeds(contract)
    _validate_rolling_instances(contract)
    _validate_e3_subjects(contract)
    _validate_workload(contract)
    _validate_statistics_and_permissions(contract)
    hypothesis_count = sum(
        int(item["hypothesis_count"])
        for item in contract["statistics_contract"]["multiplicity"]["families"]
    )
    return {
        "validator": "WGT-V11-R5-RESULT-BLIND-FREEZE-VALIDATOR-01",
        "status": "PASS",
        "paired_master_seed_count": len(
            contract["seed_contract"]["paired_master_seeds_u64"]
        ),
        "rolling_public_instance_count": len(
            contract["seed_contract"]["rolling_public_instances"]
        ),
        "e3_public_subject_count": len(
            contract["seed_contract"]["e3_public_subjects"]
        ),
        "confirmatory_hypothesis_count": hypothesis_count,
        "unique_method_sequences": contract["workload_budget"][
            "total_unique_method_sequences"
        ],
        "total_CFE": contract["workload_budget"]["total_CFE"],
        "total_atomic_model_steps": contract["workload_budget"][
            "total_atomic_model_steps"
        ],
        "effect_estimation_performed": False,
        "participant_data_accessed": False,
        "hidden_instance_accessed_or_generated": False,
        "results_analysis_performed": False,
        "r6_or_formal_execution_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compact",
        action="store_true",
        help="emit one-line JSON",
    )
    args = parser.parse_args()
    summary = validate_r5()
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=None if args.compact else 2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
