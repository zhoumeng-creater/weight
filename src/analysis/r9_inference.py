"""Versioned R9 confirmatory inference under the frozen R5 contract."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import gzip
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


IMPLEMENTATION_ID = (
    "WGT-V11-R9-INFERENCE-v1.0.1-result-aware-authorized"
)
AUTHORIZATION_TOKEN = "R9_RESULT_AWARE_CONFIRMATORY_INFERENCE_AUTHORIZED"
RAW_MANIFEST_SHA256 = (
    "33ab590adf809ca2b1f87c1ef225a18d43f50dbc40d7f3c2e2da7a379b1768d3"
)
R5_CONTRACT_SHA256 = (
    "4e2dd0a0f4a97b57d71dd13eb60aa8a3c3eb34f0708aae609d50a31d155f6554"
)
R9_EXPORT_MANIFEST_SHA256 = (
    "9b9761360294b6194aea05d09504223c49fafee26f9e343e5fd7e5667d0b9e94"
)
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_LEVEL = 0.95
BOOTSTRAP_SEED_U64 = 18_411_052_415_373_205_191
SIGN_FLIP_REPLICATES = 100_000
SIGN_FLIP_SEED_U64 = 12_682_445_017_195_329_024
HOLM_ALPHA = 0.05
ENDPOINT_LOWER_BOUND = 0.0
ENDPOINT_UPPER_BOUND = 1.0
FLOAT_COMPARISON_TOLERANCE = 1e-12

E1_COMPARATORS = (
    "MATCHED_FIXED_DE_PARETO",
    "MATCHED_JDE_STYLE_PARETO",
    "MATCHED_SHADE_STYLE_PARETO",
    "JMETALPY_1_7_GDE3_STANDARD_PARETO_DE",
    "JMETALPY_1_7_NSGAII_STATIC_CMOEA",
)
E1_DYNAMIC_COMPARATORS = E1_COMPARATORS[:-1] + (
    "JMETALPY_1_7_NSGAII_DYNAMIC_RESTART_BRIDGE",
)
E2_DYNAMIC_VARIANTS = (
    "NO_CROSS_EVENT_MEMORY",
    "NO_REJECTION_CREDIT",
    "NO_MEMORY_RESET_GATE",
    "NO_LINEAGE_CREDIT",
    "CROSS_EVENT_WARM_START_ONLY",
    "CROSS_EVENT_MEMORY_ONLY",
    "SHADE_ONLY",
)
E2_ROLLING_VARIANTS = (
    "NO_CROSS_EVENT_MEMORY",
    "NO_EXECUTION_FEEDBACK",
    "NO_REJECTION_CREDIT",
    "NO_MEMORY_RESET_GATE",
    "NO_LINEAGE_CREDIT",
    "CROSS_EVENT_WARM_START_ONLY",
    "CROSS_EVENT_MEMORY_ONLY",
    "SHADE_ONLY",
)

TASK_ENDPOINT_FIELDS = (
    "task_id",
    "schedule_index",
    "workload_id",
    "unit_id",
    "method_id",
    "replicate_index",
    "task_status",
    "endpoint_status",
    "anytime_nhv_auc",
    "final_nhv",
    "transfer_early_auc",
    "timeout_carried_forward_event_count",
)


class R9InferenceError(RuntimeError):
    """An R9 inference binding or statistical invariant failed."""


@dataclass(frozen=True)
class RegisteredHypothesis:
    """One result-blind R5 hypothesis in canonical RNG-consumption order."""

    hypothesis_index: int
    hypothesis_id: str
    family_id: str
    analysis_workload_id: str
    proposed_workload_id: str
    proposed_method_id: str
    comparator_workload_id: str
    comparator_method_id: str
    endpoint_id: str
    endpoint_field: str
    top_level_unit_rule: str
    expected_top_level_clusters: int
    expected_nested_pairs: int
    practical_threshold: float


@dataclass(frozen=True)
class TaskEndpoint:
    """The endpoint-bearing metadata for one scheduled method sequence."""

    task_id: str
    workload_id: str
    unit_id: str
    method_id: str
    replicate_index: int
    task_status: str
    endpoint_status: str
    anytime_nhv_auc: float | None
    transfer_early_auc: float | None

    @property
    def pair_key(self) -> tuple[str, int]:
        return (self.unit_id, self.replicate_index)


@dataclass(frozen=True)
class ClusterEffect:
    """Available-case effect and FAS bounds for one independent cluster."""

    cluster_id: str
    valid_nested_pairs: int
    expected_nested_pairs: int
    effect: float | None
    lower_bound: float
    upper_bound: float


@dataclass(frozen=True)
class BootstrapCluster:
    """Nested paired differences retained for the stratified bootstrap."""

    cluster_id: str
    top_stratum_id: str
    fixed_stratum_values: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class HypothesisComputation:
    """One hypothesis before within-family Holm adjustment."""

    hypothesis: RegisteredHypothesis
    clusters: tuple[ClusterEffect, ...]
    point_estimate: float
    ci_lower: float
    ci_upper: float
    fas_lower: float
    fas_upper: float
    available_nested_pairs: int
    sign_flip_extreme_count: int
    p_unadjusted: float


def canonical_json_bytes(value: Any) -> bytes:
    """Return the repository's canonical UTF-8 JSON representation."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    """Hash a file without loading it into memory."""

    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise R9InferenceError(f"cannot read valid JSON: {path}") from error


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise R9InferenceError(message)


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left.resolve())) == os.path.normcase(
        str(right.resolve())
    )


def _is_within(candidate: Path, parent: Path) -> bool:
    candidate_text = os.path.normcase(str(candidate.resolve()))
    parent_text = os.path.normcase(str(parent.resolve()))
    try:
        return os.path.commonpath((candidate_text, parent_text)) == parent_text
    except ValueError:
        return False


def _hypothesis_id(
    workload: str,
    proposed: str,
    comparator: str,
    endpoint: str,
) -> str:
    return f"{workload}__{proposed}__VS__{comparator}__{endpoint}"


def registered_hypotheses(
    r5_contract: Mapping[str, Any] | None = None,
) -> tuple[RegisteredHypothesis, ...]:
    """Build the immutable 30-hypothesis order used by both RNG streams."""

    hypotheses: list[RegisteredHypothesis] = []

    def add(
        *,
        family: str,
        analysis_workload: str,
        proposed_workload: str,
        proposed: str,
        comparator_workload: str,
        comparator: str,
        endpoint_id: str,
        endpoint_field: str,
        unit_rule: str,
        clusters: int,
        nested_pairs: int,
        threshold: float,
    ) -> None:
        hypotheses.append(
            RegisteredHypothesis(
                hypothesis_index=len(hypotheses) + 1,
                hypothesis_id=_hypothesis_id(
                    analysis_workload,
                    proposed,
                    comparator,
                    endpoint_id,
                ),
                family_id=family,
                analysis_workload_id=analysis_workload,
                proposed_workload_id=proposed_workload,
                proposed_method_id=proposed,
                comparator_workload_id=comparator_workload,
                comparator_method_id=comparator,
                endpoint_id=endpoint_id,
                endpoint_field=endpoint_field,
                top_level_unit_rule=unit_rule,
                expected_top_level_clusters=clusters,
                expected_nested_pairs=nested_pairs,
                practical_threshold=threshold,
            )
        )

    for comparator in E1_COMPARATORS:
        add(
            family="E1_PRIMARY_ANYTIME",
            analysis_workload="E1_STATIC",
            proposed_workload="E1_STATIC",
            proposed="F22_MG_STATIC",
            comparator_workload="E1_STATIC",
            comparator=comparator,
            endpoint_id="ANYTIME_NHV_AUC",
            endpoint_field="anytime_nhv_auc",
            unit_rule="unit_id",
            clusters=14,
            nested_pairs=140,
            threshold=0.02,
        )
    for comparator in E1_DYNAMIC_COMPARATORS:
        add(
            family="E1_PRIMARY_ANYTIME",
            analysis_workload="E1_DYNAMIC",
            proposed_workload="E1_DYNAMIC",
            proposed="DT-RAMDE_TS2_FULL",
            comparator_workload="E1_DYNAMIC",
            comparator=comparator,
            endpoint_id="ANYTIME_NHV_AUC",
            endpoint_field="anytime_nhv_auc",
            unit_rule="problem_before_slash",
            clusters=15,
            nested_pairs=150,
            threshold=0.02,
        )
    for comparator in E1_DYNAMIC_COMPARATORS:
        add(
            family="E1_PRIMARY_ANYTIME",
            analysis_workload="E1_ROLLING",
            proposed_workload="E1_ROLLING",
            proposed="DT-RAMDE_TS2_FULL",
            comparator_workload="E1_ROLLING",
            comparator=comparator,
            endpoint_id="ANYTIME_NHV_AUC",
            endpoint_field="anytime_nhv_auc",
            unit_rule="unit_id",
            clusters=32,
            nested_pairs=160,
            threshold=0.02,
        )
    for comparator in E2_DYNAMIC_VARIANTS:
        add(
            family="E2_DYNAMIC_TRANSFER",
            analysis_workload="E2_DYNAMIC",
            proposed_workload="E1_DYNAMIC",
            proposed="DT-RAMDE_TS2_FULL",
            comparator_workload=(
                "E2_DYNAMIC_INCREMENTAL_AFTER_FULL_REUSE"
            ),
            comparator=comparator,
            endpoint_id="TRANSFER_EARLY_AUC",
            endpoint_field="transfer_early_auc",
            unit_rule="problem_before_slash",
            clusters=15,
            nested_pairs=150,
            threshold=0.03,
        )
    for comparator in E2_ROLLING_VARIANTS:
        add(
            family="E2_ROLLING_TRANSFER",
            analysis_workload="E2_ROLLING",
            proposed_workload="E1_ROLLING",
            proposed="DT-RAMDE_TS2_FULL",
            comparator_workload=(
                "E2_ROLLING_INCREMENTAL_AFTER_FULL_REUSE"
            ),
            comparator=comparator,
            endpoint_id="TRANSFER_EARLY_AUC",
            endpoint_field="transfer_early_auc",
            unit_rule="unit_id",
            clusters=32,
            nested_pairs=160,
            threshold=0.03,
        )

    _require(len(hypotheses) == 30, "registered hypothesis count drifted")
    if r5_contract is not None:
        _validate_hypotheses_against_r5(
            tuple(hypotheses),
            r5_contract,
        )
    return tuple(hypotheses)


def _validate_hypotheses_against_r5(
    hypotheses: Sequence[RegisteredHypothesis],
    r5: Mapping[str, Any],
) -> None:
    design = r5["experiment_design"]
    expected_methods = {
        "E1_STATIC": (
            "F22_MG_STATIC",
            *E1_COMPARATORS,
        ),
        "E1_DYNAMIC": (
            "DT-RAMDE_TS2_FULL",
            *E1_DYNAMIC_COMPARATORS,
        ),
        "E1_ROLLING": (
            "DT-RAMDE_TS2_FULL",
            *E1_DYNAMIC_COMPARATORS,
        ),
        "E2_DYNAMIC": ("FULL", *E2_DYNAMIC_VARIANTS),
        "E2_ROLLING": ("FULL", *E2_ROLLING_VARIANTS),
    }
    for workload, methods in expected_methods.items():
        _require(
            tuple(design[workload]["methods"]) == methods,
            f"R5 method order drifted for {workload}",
        )
    _require(
        design["E1_STATIC"]["primary_endpoint"] == "ANYTIME_NHV_AUC",
        "R5 E1_STATIC endpoint drifted",
    )
    _require(
        design["E1_DYNAMIC"]["primary_endpoint"] == "ANYTIME_NHV_AUC",
        "R5 E1_DYNAMIC endpoint drifted",
    )
    _require(
        design["E1_ROLLING"]["primary_endpoint"] == "ANYTIME_NHV_AUC",
        "R5 E1_ROLLING endpoint drifted",
    )
    _require(
        design["E2_DYNAMIC"]["primary_endpoint"] == "TRANSFER_EARLY_AUC",
        "R5 E2_DYNAMIC endpoint drifted",
    )
    _require(
        design["E2_ROLLING"]["primary_endpoint"] == "TRANSFER_EARLY_AUC",
        "R5 E2_ROLLING endpoint drifted",
    )
    families = {
        row["family_id"]: int(row["hypothesis_count"])
        for row in r5["statistics_contract"]["multiplicity"]["families"]
    }
    observed = {
        family: sum(hypothesis.family_id == family for hypothesis in hypotheses)
        for family in families
    }
    _require(observed == families, "R5 multiplicity families drifted")
    thresholds = r5["endpoint_contract"]["practical_effect_thresholds"]
    _require(
        float(thresholds["ANYTIME_NHV_AUC"]) == 0.02,
        "R5 ANYTIME threshold drifted",
    )
    _require(
        float(thresholds["TRANSFER_EARLY_AUC"]) == 0.03,
        "R5 TRANSFER threshold drifted",
    )


def _validate_frozen_statistics(r5: Mapping[str, Any]) -> None:
    statistics = r5["statistics_contract"]
    bootstrap = statistics["confidence_interval"]
    sign_flip = statistics["permutation_test"]
    multiplicity = statistics["multiplicity"]
    missing = statistics["missing_and_failure_rules"]
    _require(
        bootstrap["method"]
        == "paired stratified cluster bootstrap at the highest independent unit",
        "R5 bootstrap method drifted",
    )
    _require(
        int(bootstrap["replicates"]) == BOOTSTRAP_REPLICATES,
        "R5 bootstrap replicate count drifted",
    )
    _require(
        float(bootstrap["level"]) == BOOTSTRAP_LEVEL,
        "R5 bootstrap level drifted",
    )
    _require(
        bootstrap["interval"] == "percentile",
        "R5 bootstrap interval drifted",
    )
    _require(
        int(bootstrap["seed_u64"]) == BOOTSTRAP_SEED_U64,
        "R5 bootstrap seed drifted",
    )
    _require(
        sign_flip["method"]
        == (
            "two-sided paired sign-flip preserving every nested "
            "seed/profile/event value within a top-level cluster"
        ),
        "R5 sign-flip method drifted",
    )
    _require(
        int(sign_flip["replicates"]) == SIGN_FLIP_REPLICATES,
        "R5 sign-flip replicate count drifted",
    )
    _require(
        sign_flip["p_value"] == "(b+1)/(B+1)",
        "R5 sign-flip p-value formula drifted",
    )
    _require(
        int(sign_flip["seed_u64"]) == SIGN_FLIP_SEED_U64,
        "R5 sign-flip seed drifted",
    )
    _require(
        multiplicity["procedure"] == "Holm step-down",
        "R5 multiplicity procedure drifted",
    )
    _require(
        float(multiplicity["familywise_alpha"]) == HOLM_ALPHA,
        "R5 familywise alpha drifted",
    )
    _require(
        missing["numerical_failure_continuous_endpoint"]
        == (
            "missing in VAS-Sim; report FAS-Sim risk and [0,1] "
            "worst/best bounds"
        ),
        "R5 numerical failure rule drifted",
    )
    _require(
        missing["unpaired_fallback_allowed"] is False,
        "R5 unexpectedly allows unpaired fallback",
    )


def _verify_file_commitment(
    path: Path,
    commitment: Mapping[str, Any],
    *,
    label: str,
) -> None:
    _require(path.is_file(), f"{label} is missing: {path}")
    _require(
        path.stat().st_size == int(commitment["bytes"]),
        f"{label} byte count mismatch",
    )
    _require(
        file_sha256(path) == commitment["sha256"],
        f"{label} SHA-256 mismatch",
    )


def _validate_source_artifacts(
    project_root: Path,
    contract: Mapping[str, Any],
) -> None:
    for relative, commitment in contract[
        "implementation_source_artifacts"
    ].items():
        _verify_file_commitment(
            project_root / relative,
            commitment,
            label=f"implementation artifact {relative}",
        )


def _validate_export_bindings(
    input_root: Path,
    contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    manifest_path = input_root / "r9_export_manifest.json"
    _require(manifest_path.is_file(), "R9 export manifest is missing")
    manifest_hash = file_sha256(manifest_path)
    expected = contract["input_binding"]
    _require(
        manifest_hash == R9_EXPORT_MANIFEST_SHA256,
        "R9 export manifest differs from the implementation freeze",
    )
    _require(
        manifest_hash == expected["export_manifest_sha256"],
        "R9 export manifest differs from the versioned contract",
    )
    manifest = _read_json(manifest_path)
    _require(
        manifest["raw_run_manifest_sha256"] == RAW_MANIFEST_SHA256,
        "raw manifest binding drifted",
    )
    _require(
        manifest["raw_run_manifest_sha256"]
        == expected["raw_run_manifest_sha256"],
        "raw manifest differs from the versioned contract",
    )
    _require(
        manifest["schedule_sha256"] == expected["schedule_sha256"],
        "schedule binding drifted",
    )
    _require(
        manifest["reference_catalog_sha256"]
        == expected["reference_catalog_sha256"],
        "reference catalog binding drifted",
    )
    _require(
        manifest["event_diagnostics_generated"] is True,
        "event diagnostics are required for E2 FULL reconstruction",
    )
    _require(
        manifest["raw_source_mutated_or_deleted"] is False,
        "export manifest reports raw source mutation or deletion",
    )
    _require(
        manifest["artifacts"] == expected["artifacts"],
        "input artifact commitments drifted",
    )
    for relative, commitment in manifest["artifacts"].items():
        _verify_file_commitment(
            input_root / relative,
            commitment,
            label=f"R9 export artifact {relative}",
        )
    return manifest


def validate_r9_inference_inputs(
    *,
    project_root: Path,
    input_root: Path,
    r5_contract_path: Path,
    implementation_contract_path: Path,
    implementation_contract_sha256: str,
    output_root: Path,
    authorization: str,
) -> dict[str, Any]:
    """Validate all frozen bindings without reading endpoint values."""

    project_root = project_root.resolve()
    input_root = input_root.resolve()
    r5_contract_path = r5_contract_path.resolve()
    implementation_contract_path = implementation_contract_path.resolve()
    output_root = output_root.resolve()

    _require(
        authorization == AUTHORIZATION_TOKEN,
        "exact R9 inference authorization token was not supplied",
    )
    _require(input_root.is_dir(), "R9 input root is missing")
    _require(not output_root.exists(), "R9 output root already exists")
    _require(
        not _is_within(output_root, input_root),
        "R9 output root must be outside the read-only input root",
    )

    implementation_hash = file_sha256(implementation_contract_path)
    _require(
        implementation_hash == implementation_contract_sha256,
        "implementation contract SHA-256 argument is incorrect",
    )
    contract = _read_json(implementation_contract_path)
    _require(
        contract["implementation_id"] == IMPLEMENTATION_ID,
        "implementation ID drifted",
    )
    _require(
        contract["status"] == "RESULT_AWARE_AUTHOR_APPROVED_FROZEN",
        "implementation contract is not author-approved and frozen",
    )
    _require(
        contract["authorization"]["token"] == AUTHORIZATION_TOKEN,
        "implementation authorization token drifted",
    )
    contracted_input = Path(contract["input_binding"]["input_root"])
    _require(
        _same_path(input_root, contracted_input),
        "input root differs from the versioned implementation contract",
    )

    r5_hash = file_sha256(r5_contract_path)
    _require(r5_hash == R5_CONTRACT_SHA256, "R5 contract hash drifted")
    _require(
        r5_hash == contract["r5_binding"]["sha256"],
        "R5 contract differs from the versioned implementation contract",
    )
    r5 = _read_json(r5_contract_path)
    _validate_frozen_statistics(r5)
    hypotheses = registered_hypotheses(r5)
    _require(
        contract["hypothesis_order"]
        == [row.hypothesis_id for row in hypotheses],
        "versioned hypothesis order drifted",
    )
    procedure = contract["procedure"]
    _require(
        procedure["bootstrap_replicates"] == BOOTSTRAP_REPLICATES,
        "implementation bootstrap count drifted",
    )
    _require(
        procedure["bootstrap_seed_u64"] == str(BOOTSTRAP_SEED_U64),
        "implementation bootstrap seed drifted",
    )
    _require(
        procedure["sign_flip_replicates"] == SIGN_FLIP_REPLICATES,
        "implementation sign-flip count drifted",
    )
    _require(
        procedure["sign_flip_seed_u64"] == str(SIGN_FLIP_SEED_U64),
        "implementation sign-flip seed drifted",
    )
    _require(
        procedure["holm_familywise_alpha"] == HOLM_ALPHA,
        "implementation Holm alpha drifted",
    )
    _require(
        procedure["rng_consumption_order"]
        == "one PCG64 stream per procedure; hypotheses consume sequentially",
        "implementation RNG consumption rule drifted",
    )
    _require(
        procedure["bootstrap_percentile_method"] == "linear",
        "implementation percentile interpolation drifted",
    )
    _require(
        procedure["rolling_bootstrap_stratification"]
        == "resample instances within each fixed rolling template",
        "implementation rolling bootstrap stratification drifted",
    )
    _require(
        procedure["nested_seed_bootstrap"]
        == (
            "paired seeds resampled within each selected cluster and "
            "fixed profile/template stratum"
        ),
        "implementation nested-seed bootstrap drifted",
    )
    _require(
        procedure["continuous_endpoint_bounds"] == [0.0, 1.0],
        "implementation endpoint bounds drifted",
    )

    _validate_source_artifacts(project_root, contract)
    manifest = _validate_export_bindings(input_root, contract)
    return {
        "implementation_id": IMPLEMENTATION_ID,
        "status": "PASS_VALIDATE_ONLY",
        "implementation_contract_sha256": implementation_hash,
        "r5_contract_sha256": r5_hash,
        "r9_export_manifest_sha256": R9_EXPORT_MANIFEST_SHA256,
        "raw_run_manifest_sha256": manifest[
            "raw_run_manifest_sha256"
        ],
        "hypothesis_count": len(hypotheses),
        "e3_included": False,
        "source_artifacts_modified_or_deleted": False,
    }


def paired_stratified_cluster_bootstrap(
    clusters: Sequence[BootstrapCluster],
    *,
    replicates: int,
    rng: np.random.Generator,
    level: float = BOOTSTRAP_LEVEL,
) -> tuple[float, float]:
    """Hierarchical percentile CI under the frozen R5 strata."""

    _require(bool(clusters), "no bootstrap clusters")
    _require(replicates > 0, "bootstrap replicates must be positive")
    by_top_stratum: dict[str, list[BootstrapCluster]] = {}
    for cluster in clusters:
        by_top_stratum.setdefault(cluster.top_stratum_id, []).append(
            cluster
        )

    bootstrap_sums = np.zeros(replicates, dtype=np.float64)
    sampled_cluster_count = 0
    for top_stratum in sorted(by_top_stratum):
        members = sorted(
            by_top_stratum[top_stratum],
            key=lambda row: row.cluster_id,
        )
        shape = tuple(
            len(values) for values in members[0].fixed_stratum_values
        )
        _require(bool(shape), "bootstrap cluster has no fixed strata")
        _require(
            all(count > 0 for count in shape),
            "bootstrap fixed stratum has no paired values",
        )
        for member in members:
            _require(
                tuple(
                    len(values)
                    for values in member.fixed_stratum_values
                )
                == shape,
                "unbalanced available pairs within bootstrap stratum",
            )
        _require(
            len(set(shape)) == 1,
            "fixed strata have unequal paired-seed counts",
        )
        nested = np.asarray(
            [
                member.fixed_stratum_values
                for member in members
            ],
            dtype=np.float64,
        )
        _require(
            np.isfinite(nested).all(),
            "bootstrap nested values are not finite",
        )
        cluster_count, fixed_strata, paired_seeds = nested.shape
        cluster_indices = rng.integers(
            0,
            cluster_count,
            size=(replicates, cluster_count),
            dtype=np.int64,
        )
        selected = nested[cluster_indices]
        seed_indices = rng.integers(
            0,
            paired_seeds,
            size=(
                replicates,
                cluster_count,
                fixed_strata,
                paired_seeds,
            ),
            dtype=np.int64,
        )
        resampled = np.take_along_axis(
            selected,
            seed_indices,
            axis=3,
        )
        cluster_means = resampled.mean(axis=3).mean(axis=2)
        bootstrap_sums += cluster_means.sum(axis=1)
        sampled_cluster_count += cluster_count
    bootstrap_means = bootstrap_sums / sampled_cluster_count
    tail = (1.0 - level) / 2.0
    lower, upper = np.quantile(
        bootstrap_means,
        (tail, 1.0 - tail),
        method="linear",
    )
    return float(lower), float(upper)


def paired_sign_flip(
    effects: Sequence[float],
    *,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[int, float]:
    """Two-sided cluster-level paired sign-flip with plus-one p-value."""

    values = np.asarray(effects, dtype=np.float64)
    _require(values.ndim == 1 and values.size > 0, "no cluster effects")
    _require(np.isfinite(values).all(), "cluster effects are not finite")
    _require(replicates > 0, "sign-flip replicates must be positive")
    bits = rng.integers(
        0,
        2,
        size=(replicates, values.size),
        dtype=np.int8,
    )
    signs = bits * 2 - 1
    permuted = (signs @ values) / values.size
    observed = abs(float(values.mean()))
    extreme = int(np.count_nonzero(np.abs(permuted) >= observed))
    return extreme, (extreme + 1.0) / (replicates + 1.0)


def holm_adjust(
    p_values: Sequence[float],
    *,
    alpha: float = HOLM_ALPHA,
) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    """Stable Holm step-down adjusted p-values and rejection decisions."""

    values = np.asarray(p_values, dtype=np.float64)
    _require(values.ndim == 1 and values.size > 0, "no p-values for Holm")
    _require(
        np.isfinite(values).all()
        and bool(np.all((values >= 0.0) & (values <= 1.0))),
        "Holm p-values must be finite and in [0,1]",
    )
    order = np.argsort(values, kind="stable")
    count = values.size
    adjusted_sorted = np.empty(count, dtype=np.float64)
    running = 0.0
    still_rejecting = True
    reject_sorted = np.zeros(count, dtype=bool)
    for rank, original_index in enumerate(order):
        multiplier = count - rank
        candidate = min(1.0, multiplier * float(values[original_index]))
        running = max(running, candidate)
        adjusted_sorted[rank] = running
        if still_rejecting and float(values[original_index]) <= (
            alpha / multiplier
        ):
            reject_sorted[rank] = True
        else:
            still_rejecting = False
    adjusted = np.empty(count, dtype=np.float64)
    rejected = np.zeros(count, dtype=bool)
    for rank, original_index in enumerate(order):
        adjusted[original_index] = adjusted_sorted[rank]
        rejected[original_index] = reject_sorted[rank]
    return (
        tuple(float(value) for value in adjusted),
        tuple(bool(value) for value in rejected),
    )


def _optional_float(value: str, *, label: str) -> float | None:
    if value == "":
        return None
    try:
        parsed = float(value)
    except ValueError as error:
        raise R9InferenceError(f"{label} is not numeric") from error
    _require(math.isfinite(parsed), f"{label} is not finite")
    return parsed


def _load_task_endpoints(input_root: Path) -> tuple[TaskEndpoint, ...]:
    path = input_root / "task_endpoints.csv"
    rows: list[TaskEndpoint] = []
    seen_task_ids: set[str] = set()
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            _require(
                tuple(reader.fieldnames or ()) == TASK_ENDPOINT_FIELDS,
                "task_endpoints.csv columns drifted",
            )
            for raw in reader:
                task_id = raw["task_id"]
                _require(task_id not in seen_task_ids, "duplicate task_id")
                seen_task_ids.add(task_id)
                row = TaskEndpoint(
                    task_id=task_id,
                    workload_id=raw["workload_id"],
                    unit_id=raw["unit_id"],
                    method_id=raw["method_id"],
                    replicate_index=int(raw["replicate_index"]),
                    task_status=raw["task_status"],
                    endpoint_status=raw["endpoint_status"],
                    anytime_nhv_auc=_optional_float(
                        raw["anytime_nhv_auc"],
                        label=f"{task_id} anytime_nhv_auc",
                    ),
                    transfer_early_auc=_optional_float(
                        raw["transfer_early_auc"],
                        label=f"{task_id} transfer_early_auc",
                    ),
                )
                _require(
                    row.task_status == "COMPLETE",
                    f"non-complete task in formal R9 input: {task_id}",
                )
                rows.append(row)
    except OSError as error:
        raise R9InferenceError("cannot read task_endpoints.csv") from error
    _require(len(rows) == 5_030, "task endpoint row count drifted")
    workloads = {row.workload_id for row in rows}
    _require(
        workloads
        == {
            "E1_STATIC",
            "E1_DYNAMIC",
            "E1_ROLLING",
            "E2_DYNAMIC_INCREMENTAL_AFTER_FULL_REUSE",
            "E2_ROLLING_INCREMENTAL_AFTER_FULL_REUSE",
        },
        "unexpected workload, including possible E3, in R9 input",
    )
    return tuple(rows)


def _transfer_task_ids(
    rows: Sequence[TaskEndpoint],
) -> dict[str, TaskEndpoint]:
    selected_workloads = {
        "E1_DYNAMIC",
        "E1_ROLLING",
        "E2_DYNAMIC_INCREMENTAL_AFTER_FULL_REUSE",
        "E2_ROLLING_INCREMENTAL_AFTER_FULL_REUSE",
    }
    selected: dict[str, TaskEndpoint] = {}
    for row in rows:
        if row.workload_id not in selected_workloads:
            continue
        if row.workload_id.startswith("E1_") and (
            row.method_id != "DT-RAMDE_TS2_FULL"
        ):
            continue
        selected[row.task_id] = row
    _require(len(selected) == 2_640, "transfer task selection drifted")
    return selected


def _load_transfer_values(
    input_root: Path,
    rows: Sequence[TaskEndpoint],
) -> dict[str, float | None]:
    selected = _transfer_task_ids(rows)
    event_values: dict[str, list[tuple[int, float | None]]] = {
        task_id: [] for task_id in selected
    }
    path = input_root / "event_diagnostics.jsonl.gz"
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as error:
                    raise R9InferenceError(
                        f"invalid event JSON at line {line_number}"
                    ) from error
                task_id = raw["task_id"]
                if task_id not in selected:
                    continue
                event_id = int(raw["event_id"])
                if event_id == 0:
                    continue
                value = raw["early_nhv_auc"]
                if value is None:
                    parsed = None
                else:
                    parsed = float(value)
                    _require(
                        math.isfinite(parsed),
                        f"{task_id} early_nhv_auc is not finite",
                    )
                event_values[task_id].append((event_id, parsed))
    except (OSError, UnicodeError) as error:
        raise R9InferenceError(
            "cannot read event_diagnostics.jsonl.gz"
        ) from error

    results: dict[str, float | None] = {}
    for task_id, row in selected.items():
        expected_events = 59 if "DYNAMIC" in row.workload_id else 19
        events = sorted(event_values[task_id])
        _require(
            [event_id for event_id, _ in events]
            == list(range(1, expected_events + 1)),
            f"{task_id} post-initial event coverage drifted",
        )
        if row.endpoint_status != "INCLUDED":
            results[task_id] = None
            continue
        values = [value for _, value in events]
        _require(
            all(value is not None for value in values),
            f"{task_id} included transfer endpoint has missing event value",
        )
        reconstructed = float(
            np.mean(np.asarray(values, dtype=np.float64))
        )
        if row.workload_id.startswith("E2_"):
            _require(
                row.transfer_early_auc is not None,
                f"{task_id} E2 transfer endpoint is missing",
            )
            _require(
                math.isclose(
                    reconstructed,
                    row.transfer_early_auc,
                    rel_tol=0.0,
                    abs_tol=FLOAT_COMPARISON_TOLERANCE,
                ),
                f"{task_id} transfer reconstruction mismatch",
            )
        results[task_id] = reconstructed
    return results


def _endpoint_value(
    row: TaskEndpoint,
    hypothesis: RegisteredHypothesis,
    transfer_values: Mapping[str, float | None],
) -> float | None:
    if row.endpoint_status != "INCLUDED":
        return None
    if hypothesis.endpoint_field == "anytime_nhv_auc":
        _require(
            row.anytime_nhv_auc is not None,
            f"{row.task_id} included ANYTIME endpoint is missing",
        )
        return row.anytime_nhv_auc
    _require(
        hypothesis.endpoint_field == "transfer_early_auc",
        "unknown endpoint field",
    )
    _require(
        row.task_id in transfer_values,
        f"{row.task_id} transfer value was not reconstructed",
    )
    return transfer_values[row.task_id]


def _top_cluster(unit_id: str, rule: str) -> str:
    if rule == "unit_id":
        return unit_id
    _require(rule == "problem_before_slash", "unknown top-level unit rule")
    problem, separator, _ = unit_id.partition("/")
    _require(bool(separator) and bool(problem), "dynamic unit_id drifted")
    return problem


def _fixed_stratum(unit_id: str, rule: str) -> str:
    if rule == "unit_id":
        return "ALL"
    _require(rule == "problem_before_slash", "unknown fixed stratum rule")
    _, separator, stratum = unit_id.partition("/")
    _require(
        bool(separator) and bool(stratum),
        "dynamic fixed stratum drifted",
    )
    return stratum


def _bootstrap_top_stratum(
    cluster_id: str,
    hypothesis: RegisteredHypothesis,
) -> str:
    if hypothesis.analysis_workload_id in {
        "E1_ROLLING",
        "E2_ROLLING",
    }:
        template, separator, _ = cluster_id.partition("/")
        _require(
            bool(separator) and bool(template),
            "rolling template stratum drifted",
        )
        return template
    return "ALL"


def _select_rows(
    rows: Sequence[TaskEndpoint],
    *,
    workload: str,
    method: str,
) -> dict[tuple[str, int], TaskEndpoint]:
    selected: dict[tuple[str, int], TaskEndpoint] = {}
    for row in rows:
        if row.workload_id != workload or row.method_id != method:
            continue
        _require(row.pair_key not in selected, "duplicate nested pair key")
        selected[row.pair_key] = row
    return selected


def _missing_pair_bounds(
    proposed: float | None,
    comparator: float | None,
) -> tuple[float, float]:
    if proposed is not None and comparator is not None:
        difference = proposed - comparator
        return difference, difference
    if proposed is None and comparator is None:
        return (
            ENDPOINT_LOWER_BOUND - ENDPOINT_UPPER_BOUND,
            ENDPOINT_UPPER_BOUND - ENDPOINT_LOWER_BOUND,
        )
    if proposed is None:
        assert comparator is not None
        return (
            ENDPOINT_LOWER_BOUND - comparator,
            ENDPOINT_UPPER_BOUND - comparator,
        )
    assert comparator is None
    return (
        proposed - ENDPOINT_UPPER_BOUND,
        proposed - ENDPOINT_LOWER_BOUND,
    )


def compute_cluster_effects(
    hypothesis: RegisteredHypothesis,
    rows: Sequence[TaskEndpoint],
    transfer_values: Mapping[str, float | None],
) -> tuple[tuple[ClusterEffect, ...], tuple[BootstrapCluster, ...]]:
    """Aggregate nested paired differences at the frozen independent unit."""

    proposed_rows = _select_rows(
        rows,
        workload=hypothesis.proposed_workload_id,
        method=hypothesis.proposed_method_id,
    )
    comparator_rows = _select_rows(
        rows,
        workload=hypothesis.comparator_workload_id,
        method=hypothesis.comparator_method_id,
    )
    _require(
        set(proposed_rows) == set(comparator_rows),
        f"{hypothesis.hypothesis_id} has unpaired scheduled rows",
    )
    _require(
        len(proposed_rows) == hypothesis.expected_nested_pairs,
        f"{hypothesis.hypothesis_id} nested sample size drifted",
    )

    cluster_values: dict[str, dict[str, list[float]]] = {}
    cluster_lowers: dict[str, dict[str, list[float]]] = {}
    cluster_uppers: dict[str, dict[str, list[float]]] = {}
    cluster_expected: dict[str, int] = {}
    for pair_key in sorted(proposed_rows):
        proposed_row = proposed_rows[pair_key]
        comparator_row = comparator_rows[pair_key]
        cluster = _top_cluster(
            proposed_row.unit_id,
            hypothesis.top_level_unit_rule,
        )
        _require(
            cluster
            == _top_cluster(
                comparator_row.unit_id,
                hypothesis.top_level_unit_rule,
            ),
            "paired rows resolve to different top-level clusters",
        )
        fixed_stratum = _fixed_stratum(
            proposed_row.unit_id,
            hypothesis.top_level_unit_rule,
        )
        _require(
            fixed_stratum
            == _fixed_stratum(
                comparator_row.unit_id,
                hypothesis.top_level_unit_rule,
            ),
            "paired rows resolve to different fixed strata",
        )
        proposed = _endpoint_value(
            proposed_row,
            hypothesis,
            transfer_values,
        )
        comparator = _endpoint_value(
            comparator_row,
            hypothesis,
            transfer_values,
        )
        cluster_values.setdefault(cluster, {}).setdefault(
            fixed_stratum,
            [],
        )
        cluster_lowers.setdefault(cluster, {}).setdefault(
            fixed_stratum,
            [],
        )
        cluster_uppers.setdefault(cluster, {}).setdefault(
            fixed_stratum,
            [],
        )
        cluster_expected[cluster] = cluster_expected.get(cluster, 0) + 1
        if proposed is not None and comparator is not None:
            cluster_values[cluster][fixed_stratum].append(
                proposed - comparator
            )
        lower, upper = _missing_pair_bounds(proposed, comparator)
        cluster_lowers[cluster][fixed_stratum].append(lower)
        cluster_uppers[cluster][fixed_stratum].append(upper)

    _require(
        len(cluster_expected) == hypothesis.expected_top_level_clusters,
        f"{hypothesis.hypothesis_id} top-level sample size drifted",
    )
    clusters: list[ClusterEffect] = []
    bootstrap_clusters: list[BootstrapCluster] = []
    for cluster in sorted(cluster_expected):
        fixed_strata = sorted(cluster_values[cluster])
        valid_strata = [
            cluster_values[cluster][stratum]
            for stratum in fixed_strata
            if cluster_values[cluster][stratum]
        ]
        valid_count = sum(len(values) for values in valid_strata)
        effect = (
            float(
                np.mean(
                    [
                        float(np.mean(values))
                        for values in valid_strata
                    ]
                )
            )
            if valid_strata
            else None
        )
        lower = float(
            np.mean(
                [
                    float(np.mean(cluster_lowers[cluster][stratum]))
                    for stratum in fixed_strata
                ]
            )
        )
        upper = float(
            np.mean(
                [
                    float(np.mean(cluster_uppers[cluster][stratum]))
                    for stratum in fixed_strata
                ]
            )
        )
        clusters.append(
            ClusterEffect(
                cluster_id=cluster,
                valid_nested_pairs=valid_count,
                expected_nested_pairs=cluster_expected[cluster],
                effect=effect,
                lower_bound=lower,
                upper_bound=upper,
            )
        )
        if effect is not None:
            _require(
                len(valid_strata) == len(fixed_strata),
                "available cluster is missing an entire fixed stratum",
            )
            bootstrap_clusters.append(
                BootstrapCluster(
                    cluster_id=cluster,
                    top_stratum_id=_bootstrap_top_stratum(
                        cluster,
                        hypothesis,
                    ),
                    fixed_stratum_values=tuple(
                        tuple(cluster_values[cluster][stratum])
                        for stratum in fixed_strata
                    ),
                )
            )
    return tuple(clusters), tuple(bootstrap_clusters)


def _compute_hypothesis(
    hypothesis: RegisteredHypothesis,
    rows: Sequence[TaskEndpoint],
    transfer_values: Mapping[str, float | None],
    *,
    bootstrap_rng: np.random.Generator,
    sign_flip_rng: np.random.Generator,
) -> HypothesisComputation:
    clusters, bootstrap_clusters = compute_cluster_effects(
        hypothesis,
        rows,
        transfer_values,
    )
    effects = [
        cluster.effect
        for cluster in clusters
        if cluster.effect is not None
    ]
    _require(effects, f"{hypothesis.hypothesis_id} has no available clusters")
    point = float(np.mean(np.asarray(effects, dtype=np.float64)))
    ci_lower, ci_upper = paired_stratified_cluster_bootstrap(
        bootstrap_clusters,
        replicates=BOOTSTRAP_REPLICATES,
        rng=bootstrap_rng,
    )
    extreme, p_value = paired_sign_flip(
        effects,
        replicates=SIGN_FLIP_REPLICATES,
        rng=sign_flip_rng,
    )
    return HypothesisComputation(
        hypothesis=hypothesis,
        clusters=clusters,
        point_estimate=point,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        fas_lower=float(np.mean([row.lower_bound for row in clusters])),
        fas_upper=float(np.mean([row.upper_bound for row in clusters])),
        available_nested_pairs=sum(
            row.valid_nested_pairs for row in clusters
        ),
        sign_flip_extreme_count=extreme,
        p_unadjusted=p_value,
    )


def _float_text(value: float) -> str:
    return format(value, ".17g")


def _practical_class(effect: float, threshold: float) -> str:
    if effect >= threshold:
        return "POSITIVE"
    if effect <= -threshold:
        return "NEGATIVE"
    return "SMALL_OR_NULL"


def _csv_bytes(
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> bytes:
    target = io.StringIO(newline="")
    writer = csv.DictWriter(
        target,
        fieldnames=fieldnames,
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(rows)
    return target.getvalue().encode("utf-8")


def _output_payloads(
    computations: Sequence[HypothesisComputation],
    *,
    holm_adjusted: Mapping[str, float],
    holm_rejected: Mapping[str, bool],
    implementation_contract_sha256: str,
) -> dict[str, bytes]:
    hypothesis_fields = (
        "hypothesis_index",
        "hypothesis_id",
        "family_id",
        "analysis_workload_id",
        "endpoint_id",
        "proposed_method_id",
        "comparator_method_id",
        "effect_size_type",
        "expected_top_level_clusters",
        "available_top_level_clusters",
        "missing_top_level_clusters",
        "expected_nested_pairs",
        "available_nested_pairs",
        "point_estimate",
        "ci_lower_95",
        "ci_upper_95",
        "fas_worst_case_bound",
        "fas_best_case_bound",
        "practical_threshold",
        "practical_effect_class",
        "sign_flip_extreme_count",
        "sign_flip_replicates",
        "p_unadjusted",
        "p_holm",
        "holm_reject_0_05",
    )
    hypothesis_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    for computation in computations:
        hypothesis = computation.hypothesis
        available_clusters = sum(
            cluster.effect is not None for cluster in computation.clusters
        )
        hypothesis_rows.append(
            {
                "hypothesis_index": hypothesis.hypothesis_index,
                "hypothesis_id": hypothesis.hypothesis_id,
                "family_id": hypothesis.family_id,
                "analysis_workload_id": (
                    hypothesis.analysis_workload_id
                ),
                "endpoint_id": hypothesis.endpoint_id,
                "proposed_method_id": hypothesis.proposed_method_id,
                "comparator_method_id": (
                    hypothesis.comparator_method_id
                ),
                "effect_size_type": "PAIRED_MACRO_MEAN_DIFFERENCE",
                "expected_top_level_clusters": (
                    hypothesis.expected_top_level_clusters
                ),
                "available_top_level_clusters": available_clusters,
                "missing_top_level_clusters": (
                    hypothesis.expected_top_level_clusters
                    - available_clusters
                ),
                "expected_nested_pairs": hypothesis.expected_nested_pairs,
                "available_nested_pairs": (
                    computation.available_nested_pairs
                ),
                "point_estimate": _float_text(
                    computation.point_estimate
                ),
                "ci_lower_95": _float_text(computation.ci_lower),
                "ci_upper_95": _float_text(computation.ci_upper),
                "fas_worst_case_bound": _float_text(
                    computation.fas_lower
                ),
                "fas_best_case_bound": _float_text(
                    computation.fas_upper
                ),
                "practical_threshold": _float_text(
                    hypothesis.practical_threshold
                ),
                "practical_effect_class": _practical_class(
                    computation.point_estimate,
                    hypothesis.practical_threshold,
                ),
                "sign_flip_extreme_count": (
                    computation.sign_flip_extreme_count
                ),
                "sign_flip_replicates": SIGN_FLIP_REPLICATES,
                "p_unadjusted": _float_text(
                    computation.p_unadjusted
                ),
                "p_holm": _float_text(
                    holm_adjusted[hypothesis.hypothesis_id]
                ),
                "holm_reject_0_05": str(
                    holm_rejected[hypothesis.hypothesis_id]
                ).lower(),
            }
        )
        for cluster in computation.clusters:
            cluster_rows.append(
                {
                    "hypothesis_index": hypothesis.hypothesis_index,
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "family_id": hypothesis.family_id,
                    "cluster_id": cluster.cluster_id,
                    "expected_nested_pairs": (
                        cluster.expected_nested_pairs
                    ),
                    "valid_nested_pairs": cluster.valid_nested_pairs,
                    "cluster_effect_available": str(
                        cluster.effect is not None
                    ).lower(),
                    "cluster_effect": (
                        ""
                        if cluster.effect is None
                        else _float_text(cluster.effect)
                    ),
                    "cluster_worst_case_bound": _float_text(
                        cluster.lower_bound
                    ),
                    "cluster_best_case_bound": _float_text(
                        cluster.upper_bound
                    ),
                }
            )
    cluster_fields = (
        "hypothesis_index",
        "hypothesis_id",
        "family_id",
        "cluster_id",
        "expected_nested_pairs",
        "valid_nested_pairs",
        "cluster_effect_available",
        "cluster_effect",
        "cluster_worst_case_bound",
        "cluster_best_case_bound",
    )
    readme = f"""# R9 versioned confirmatory inference

- Implementation: `{IMPLEMENTATION_ID}`
- Implementation contract SHA-256: `{implementation_contract_sha256}`
- R5 contract SHA-256: `{R5_CONTRACT_SHA256}`
- R9 event export manifest SHA-256: `{R9_EXPORT_MANIFEST_SHA256}`
- Raw run manifest SHA-256: `{RAW_MANIFEST_SHA256}`
- Scope: E1 and E2 only; E3 is excluded.
- Direction: proposed minus comparator for both higher-is-better endpoints.
- CI: paired stratified hierarchical cluster bootstrap, 20,000 replicates,
  95% linear percentile interval, frozen PCG64 seed. Rolling instances are
  resampled within each fixed template; paired seeds are resampled within
  each selected cluster and fixed profile/template stratum.
- Test: two-sided paired top-level cluster sign-flip, 100,000 replicates,
  plus-one p-value, frozen PCG64 seed.
- Multiplicity: Holm step-down within each of the three frozen families,
  familywise alpha 0.05.
- Missing numerical continuous endpoints: no imputation in confirmatory tests;
  available-case estimates are accompanied by endpoint-[0,1] FAS bounds.
- RNG consumption: one stream per procedure, consumed sequentially in the
  30-row canonical hypothesis order.
- Provenance: this implementation was created after effect visibility under
  explicit author authorization; it does not claim to be a result-blind
  pre-specified software implementation.

`confirmatory_hypotheses.csv` contains the 30 registered comparisons.
`top_level_cluster_effects.csv` contains the raw independent-unit effects and
missing-data bounds used by the inference.
"""
    return {
        "README.md": readme.encode("utf-8"),
        "confirmatory_hypotheses.csv": _csv_bytes(
            hypothesis_fields,
            hypothesis_rows,
        ),
        "top_level_cluster_effects.csv": _csv_bytes(
            cluster_fields,
            cluster_rows,
        ),
    }


def _holm_maps(
    computations: Sequence[HypothesisComputation],
) -> tuple[dict[str, float], dict[str, bool]]:
    adjusted: dict[str, float] = {}
    rejected: dict[str, bool] = {}
    families = (
        "E1_PRIMARY_ANYTIME",
        "E2_DYNAMIC_TRANSFER",
        "E2_ROLLING_TRANSFER",
    )
    for family in families:
        members = [
            row for row in computations if row.hypothesis.family_id == family
        ]
        values, decisions = holm_adjust(
            [row.p_unadjusted for row in members]
        )
        for row, value, decision in zip(
            members,
            values,
            decisions,
            strict=True,
        ):
            adjusted[row.hypothesis.hypothesis_id] = value
            rejected[row.hypothesis.hypothesis_id] = decision
    _require(len(adjusted) == 30, "Holm family coverage drifted")
    return adjusted, rejected


def _write_new_file(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except OSError as error:
        raise R9InferenceError(f"cannot create output artifact: {path}") from error


def run_r9_inference(
    *,
    project_root: Path,
    input_root: Path,
    r5_contract_path: Path,
    implementation_contract_path: Path,
    implementation_contract_sha256: str,
    output_root: Path,
    authorization: str,
) -> dict[str, Any]:
    """Run the frozen inference and publish a new immutable-style result root."""

    validation = validate_r9_inference_inputs(
        project_root=project_root,
        input_root=input_root,
        r5_contract_path=r5_contract_path,
        implementation_contract_path=implementation_contract_path,
        implementation_contract_sha256=implementation_contract_sha256,
        output_root=output_root,
        authorization=authorization,
    )
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    contract = _read_json(implementation_contract_path.resolve())
    r5 = _read_json(r5_contract_path.resolve())
    hypotheses = registered_hypotheses(r5)
    rows = _load_task_endpoints(input_root)
    transfer_values = _load_transfer_values(input_root, rows)

    bootstrap_rng = np.random.Generator(
        np.random.PCG64(BOOTSTRAP_SEED_U64)
    )
    sign_flip_rng = np.random.Generator(
        np.random.PCG64(SIGN_FLIP_SEED_U64)
    )
    computations = tuple(
        _compute_hypothesis(
            hypothesis,
            rows,
            transfer_values,
            bootstrap_rng=bootstrap_rng,
            sign_flip_rng=sign_flip_rng,
        )
        for hypothesis in hypotheses
    )
    holm_adjusted, holm_rejected = _holm_maps(computations)
    payloads = _output_payloads(
        computations,
        holm_adjusted=holm_adjusted,
        holm_rejected=holm_rejected,
        implementation_contract_sha256=(
            implementation_contract_sha256
        ),
    )

    _validate_source_artifacts(project_root.resolve(), contract)
    manifest_input = _validate_export_bindings(input_root, contract)
    _require(not output_root.exists(), "R9 output root appeared during run")
    try:
        output_root.mkdir(parents=False, exist_ok=False)
    except OSError as error:
        raise R9InferenceError(
            f"cannot create new output root: {output_root}"
        ) from error
    for relative, payload in payloads.items():
        _write_new_file(output_root / relative, payload)

    _validate_source_artifacts(project_root.resolve(), contract)
    _validate_export_bindings(input_root, contract)
    artifacts = {
        relative: {
            "bytes": (output_root / relative).stat().st_size,
            "sha256": file_sha256(output_root / relative),
        }
        for relative in sorted(payloads)
    }
    family_rejections = {
        family: sum(
            holm_rejected[row.hypothesis.hypothesis_id]
            for row in computations
            if row.hypothesis.family_id == family
        )
        for family in (
            "E1_PRIMARY_ANYTIME",
            "E2_DYNAMIC_TRANSFER",
            "E2_ROLLING_TRANSFER",
        )
    }
    manifest = {
        "artifact_role": "R9_VERSIONED_CONFIRMATORY_INFERENCE",
        "implementation_id": IMPLEMENTATION_ID,
        "status": "COMPLETE",
        "authorization": AUTHORIZATION_TOKEN,
        "provenance_status": "RESULT_AWARE_AUTHOR_APPROVED_IMPLEMENTATION",
        "implementation_contract_sha256": (
            implementation_contract_sha256
        ),
        "r5_contract_sha256": R5_CONTRACT_SHA256,
        "r9_export_manifest_sha256": R9_EXPORT_MANIFEST_SHA256,
        "raw_run_manifest_sha256": manifest_input[
            "raw_run_manifest_sha256"
        ],
        "hypothesis_count": len(computations),
        "family_rejections_holm_0_05": family_rejections,
        "bootstrap": {
            "method": (
                "paired stratified hierarchical cluster bootstrap at "
                "the highest independent unit"
            ),
            "replicates": BOOTSTRAP_REPLICATES,
            "level": BOOTSTRAP_LEVEL,
            "interval": "percentile",
            "numpy_quantile_method": "linear",
            "rolling_stratification": (
                "instances resampled within fixed template"
            ),
            "nested_seed_resampling": (
                "paired seeds resampled within selected cluster and "
                "fixed stratum"
            ),
            "bit_generator": "PCG64",
            "seed_u64": str(BOOTSTRAP_SEED_U64),
        },
        "sign_flip": {
            "method": "two-sided paired top-level cluster sign-flip",
            "replicates": SIGN_FLIP_REPLICATES,
            "p_value": "(b+1)/(B+1)",
            "bit_generator": "PCG64",
            "seed_u64": str(SIGN_FLIP_SEED_U64),
        },
        "multiplicity": {
            "procedure": "Holm step-down",
            "familywise_alpha": HOLM_ALPHA,
            "family_count": 3,
        },
        "missing_continuous_endpoint_policy": (
            "no imputation in confirmatory tests; endpoint [0,1] "
            "worst/best FAS bounds reported"
        ),
        "hypothesis_order": [
            row.hypothesis.hypothesis_id for row in computations
        ],
        "e3_included": False,
        "source_input_modified_or_deleted": False,
        "artifacts": artifacts,
        "runtime": {
            "python": (
                f"{sys.version_info.major}."
                f"{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            ),
            "numpy": np.__version__,
        },
    }
    manifest_path = output_root / "r9_inference_manifest.json"
    _write_new_file(manifest_path, canonical_json_bytes(manifest) + b"\n")
    return {
        **validation,
        "status": "COMPLETE",
        "output_artifacts": {
            **artifacts,
            "r9_inference_manifest.json": {
                "bytes": manifest_path.stat().st_size,
                "sha256": file_sha256(manifest_path),
            },
        },
        "family_rejections_holm_0_05": family_rejections,
        "source_input_modified_or_deleted": False,
    }
