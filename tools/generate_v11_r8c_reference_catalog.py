"""Materialize the result-blind compact E1/E2 reference catalog."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from analysis.reference_catalog import (  # noqa: E402
    CDF_REFERENCE_SEEDS,
    REFERENCE_CATALOG_EXPECTED_IDENTITIES,
    REFERENCE_CATALOG_ID,
    REFERENCE_CATALOG_VERSION,
    bound_file,
    load_reference_catalog,
    materialize_reference_catalog,
)


CATALOG_RELATIVE_PATH = Path(
    "config/r8c_e1e2/reference_catalog/reference_artifacts.jsonl"
)
MANIFEST_RELATIVE_PATH = Path(
    "config/r8c_e1e2/reference_catalog/reference_catalog_manifest.json"
)
SCHEMA_RELATIVE_PATH = Path(
    "config/r8c_e1e2/reference_catalog/reference_catalog_manifest.schema.json"
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _binding_set_sha256(derivations: tuple[Any, ...]) -> tuple[int, str]:
    bindings = sorted(
        {
            (
                item.extrema.identity.suite_id,
                item.extrema.identity.problem_id,
                item.extrema.identity.profile,
                item.extrema.identity.evaluator_binding_sha256,
            )
            for item in derivations
        }
    )
    payload = [
        {
            "suite_id": suite,
            "problem_id": problem,
            "profile": profile,
            "evaluator_binding_sha256": binding,
        }
        for suite, problem, profile, binding in bindings
    ]
    return len(payload), sha256(_canonical_json(payload)).hexdigest()


def _manifest(
    *,
    catalog_sha256: str,
    catalog_bytes: int,
    catalog_lines: int,
    breakdown: dict[str, int],
    binding_count: int,
    binding_set_sha256: str,
) -> dict[str, Any]:
    root = REPOSITORY_ROOT
    return {
        "catalog_id": REFERENCE_CATALOG_ID,
        "schema_version": REFERENCE_CATALOG_VERSION,
        "created_date": "2026-07-26",
        "status": (
            "FROZEN_RESULT_BLIND_REFERENCE_INPUT_NOT_EXECUTION_AUTHORITY"
        ),
        "authority_amendments": {
            "lircmop": (
                "WGT-V11-R8C-E1E2-LIRCMOP-REFERENCE-AMENDMENT-01"
            ),
            "cdf": (
                "WGT-V11-R8C-E1E2-CDF-OPERATIONAL-"
                "AUTHORITY-AMENDMENT-01"
            ),
        },
        "identity_scope": {
            "expected_total": REFERENCE_CATALOG_EXPECTED_IDENTITIES,
            "actual_total": catalog_lines,
            "lircmop_static": breakdown["lircmop_static"],
            "cdf_non_cdf13": breakdown["cdf_non_cdf13"],
            "cdf13_seed_time": breakdown["cdf13_seed_time"],
            "finite_front_records": breakdown["finite_front_records"],
            "continuous_front_records": breakdown[
                "continuous_front_records"
            ],
            "cdf13_master_seeds_u64": list(CDF_REFERENCE_SEEDS),
            "cdf_profiles": ["CDF-HARSH", "CDF-MILD"],
            "cdf_events_per_sequence": 60,
            "unique_evaluator_bindings": binding_count,
            "evaluator_binding_set_sha256": binding_set_sha256,
        },
        "representation": {
            "continuous_front": (
                "OBJECTIVE_EXTREMA_PLUS_DERIVATION_OR_ROOT_CERTIFICATE"
            ),
            "finite_front": "ALL_UNIQUE_TRUE_PARETO_POINTS",
            "finite_front_order": "LEXICOGRAPHIC",
            "float_encoding": "PYTHON_FLOAT_HEX_BINARY64",
            "record_encoding": "CANONICAL_JSON_UTF8_ONE_RECORD_PER_LF",
            "arbitrary_dense_pf_samples_stored": False,
            "historical_10000_point_target": (
                "PROVENANCE_ONLY_NOT_MATERIALIZED"
            ),
            "method_output_derived_reference_allowed": False,
        },
        "catalog_artifact": {
            "path": CATALOG_RELATIVE_PATH.as_posix(),
            "bytes": catalog_bytes,
            "lines": catalog_lines,
            "sha256": catalog_sha256,
        },
        "source_bindings": {
            "cdf_paper": {
                "doi": "10.1007/s11047-020-09799-y",
                "version_of_record_url": (
                    "https://link.springer.com/article/"
                    "10.1007/s11047-020-09799-y"
                ),
            },
            "cdf_author_oracle": {
                "repository": "https://bitbucket.org/Pag1c18/cmlsga",
                "commit": (
                    "1926a5a1c89adf0a5e5e70449adbec62750a108a"
                ),
                "path": "MLSGA/Fit_Functions.cpp",
                "bytes": 461394,
                "sha256": (
                    "48b2c256f4bdec6ed4f81f8edd82a037"
                    "53bc51550776e1ae84b2d6fcbc18fa7a"
                ),
            },
            "cdf_authority_audit": bound_file(
                root
                / "config/r8c_e1e2/cdf_operational_authority_audit.md",
                repository_root=root,
            ),
            "cdf_corrective_evaluator": bound_file(
                root / "src/benchmark_adapters/cdf_operational.py",
                repository_root=root,
            ),
            "historical_evaluator": bound_file(
                root / "src/benchmark_adapters/r4_evaluators.py",
                repository_root=root,
            ),
            "lircmop_paper_evaluator": bound_file(
                root / "src/benchmark_adapters/lircmop_paper.py",
                repository_root=root,
            ),
            "reference_derivation": bound_file(
                root / "src/analysis/reference_catalog.py",
                repository_root=root,
            ),
            "reference_identity_model": bound_file(
                root / "src/analysis/reference_fronts.py",
                repository_root=root,
            ),
            "analytic_scale": bound_file(
                root / "src/analysis/checkpoint_metrics.py",
                repository_root=root,
            ),
            "generator": bound_file(Path(__file__), repository_root=root),
        },
        "verification": {
            "all_records_reloaded_and_self_hash_validated": True,
            "all_identity_sha256_unique": True,
            "finite_front_completeness_model_validated": True,
            "source_equation_witness_tests_required": True,
            "cdf5_independent_global_minimum_crosscheck_required": True,
            "cdf11_dense_domination_crosscheck_required": True,
            "cdf8_cdf13_cdf15_endpoint_witnesses_required": True,
            "manifest_schema_validation_required": True,
        },
        "effect_boundary": {
            "effect_outputs_inspected": False,
            "effect_outputs_written": False,
            "observed_method_outputs_used": False,
            "changes_algorithms_seeds_samples_endpoints_or_cfe": False,
            "authorizes_formal_execution": False,
            "authorizes_effect_analysis": False,
            "authorizes_results_writing": False,
        },
        "validation_rule": {
            "catalog_file_hash_bytes_lines_must_match": True,
            "all_bound_source_hashes_and_bytes_must_match": True,
            "all_2294_identities_must_be_present_once": True,
            "cdf13_identity_must_bind_seed_and_full_time_vector": True,
            "any_drift": "FAIL_CLOSED",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / CATALOG_RELATIVE_PATH,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT / MANIFEST_RELATIVE_PATH,
    )
    arguments = parser.parse_args()
    catalog_path = arguments.output.resolve()
    manifest_path = arguments.manifest.resolve()
    catalog_sha, catalog_bytes, catalog_lines, breakdown = (
        materialize_reference_catalog(catalog_path)
    )
    derivations = load_reference_catalog(
        catalog_path,
        expected_sha256=catalog_sha,
        expected_lines=catalog_lines,
    )
    if len(derivations) != REFERENCE_CATALOG_EXPECTED_IDENTITIES:
        raise RuntimeError("reference catalog identity count drifted")
    binding_count, binding_hash = _binding_set_sha256(derivations)
    manifest = _manifest(
        catalog_sha256=catalog_sha,
        catalog_bytes=catalog_bytes,
        catalog_lines=catalog_lines,
        breakdown=breakdown,
        binding_count=binding_count,
        binding_set_sha256=binding_hash,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "catalog": bound_file(
                    catalog_path,
                    repository_root=REPOSITORY_ROOT,
                ),
                "manifest": bound_file(
                    manifest_path,
                    repository_root=REPOSITORY_ROOT,
                ),
                "identity_count": len(derivations),
                "binding_count": binding_count,
                "breakdown": breakdown,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
