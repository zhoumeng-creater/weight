"""Validate the R4 comparator/benchmark bindings without estimating effects."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark_adapters.r4_public import (  # noqa: E402
    make_r4_cdf_adapter,
    make_r4_lircmop_adapter,
)
from benchmark_adapters.r4_wgt_rr import WGTRRPublicAdapter  # noqa: E402
from comparators import (  # noqa: E402
    ConventionalRollingPlannerBaseline,
    FixedEnergyDeficitBaseline,
    JMetalComparator,
    MatchedParetoDE,
)
from evaluation.ledger import EvaluationLedger  # noqa: E402


R4_CONFIG = ROOT / "config" / "r4"

EXPECTED_METHOD_CATEGORIES = {
    "matched_fixed_de_pareto",
    "matched_jde_style_pareto",
    "matched_shade_style_pareto",
    "standard_pareto_de_no_cross_event",
    "external_non_de_static_cmoea",
    "external_dynamic_constrained_moea",
    "domain_fixed_energy_deficit",
    "domain_conventional_rolling_planner",
}
EXPECTED_BENCHMARK_CATEGORIES = {
    "public_static_constrained_mo",
    "public_dynamic_constrained_mo",
    "public_rolling_receding_horizon_constrained_mo",
}


class R4ValidationError(RuntimeError):
    """An R4 artifact is incomplete, inconsistent, or effect-authorized."""


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_document(name: str) -> dict[str, Any]:
    document_path = R4_CONFIG / f"{name}.json"
    schema_path = R4_CONFIG / f"{name}.schema.json"
    document = _load_json(document_path)
    schema = _load_json(schema_path)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(document)
    return document


def _require_unique(values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise R4ValidationError(f"{label} values are not unique")


def _validate_semantics(
    comparators: dict[str, Any],
    benchmarks: dict[str, Any],
    licenses: dict[str, Any],
) -> None:
    methods = comparators["methods"]
    categories = [str(item["category"]) for item in methods]
    method_ids = [str(item["method_id"]) for item in methods]
    _require_unique(categories, "comparator category")
    _require_unique(method_ids, "method_id")
    if set(categories) != EXPECTED_METHOD_CATEGORIES:
        raise R4ValidationError("minimum comparator categories are incomplete")
    if comparators["effect_estimation_allowed"] is not False or any(
        item["effect_execution_allowed"] is not False for item in methods
    ):
        raise R4ValidationError("comparator manifest opened effect execution")

    suites = benchmarks["benchmarks"]
    suite_categories = [str(item["category"]) for item in suites]
    suite_ids = [str(item["suite_id"]) for item in suites]
    _require_unique(suite_categories, "benchmark category")
    _require_unique(suite_ids, "suite_id")
    if set(suite_categories) != EXPECTED_BENCHMARK_CATEGORIES:
        raise R4ValidationError("minimum benchmark categories are incomplete")
    if benchmarks["effect_estimation_allowed"] is not False or any(
        item["formal_effect_execution_allowed"] is not False
        or item["registered_effect_instance"] is not False
        for item in suites
    ):
        raise R4ValidationError("benchmark registry opened an effect instance")

    lock = licenses["dependency_lock"]
    lock_path = ROOT / str(lock["path"])
    actual_lock_hash = sha256(lock_path.read_bytes()).hexdigest()
    if actual_lock_hash != lock["sha256"]:
        raise R4ValidationError("R4 dependency lock hash differs")
    project_license = licenses["project_license"]
    project_license_path = ROOT / str(project_license["path"])
    actual_license_hash = sha256(project_license_path.read_bytes()).hexdigest()
    if actual_license_hash != project_license["sha256"]:
        raise R4ValidationError("project license hash differs")
    if (
        licenses["distribution_allowed"] is not True
        or project_license["spdx_identifier"] != "MIT"
        or licenses["release_blockers"]
    ):
        raise R4ValidationError("MIT distribution authorization is incomplete")


def _exercise_benchmark_bridges() -> dict[str, int]:
    static_count = 0
    for index in range(1, 15):
        adapter = make_r4_lircmop_adapter(index)
        adapter.freeze_information(0, None)
        ledger = EvaluationLedger(max_cfe=1)
        midpoint = (
            np.asarray(adapter.lower_bounds)
            + np.asarray(adapter.upper_bounds)
        ) / 2.0
        adapter.evaluate(midpoint, 0, ledger, f"r4-static-{index}")
        if ledger.snapshot()["cfe"] != 1:
            raise R4ValidationError("static bridge bypassed the shared ledger")
        static_count += 1

    dynamic_count = 0
    for profile in ("CDF-HARSH", "CDF-MILD"):
        for index in range(1, 16):
            adapter = make_r4_cdf_adapter(
                index,
                profile=profile,
                environment_seed=0,
            )
            adapter.freeze_information(7, None)
            ledger = EvaluationLedger(max_cfe=1)
            lower = np.asarray(adapter.lower_bounds)
            upper = np.asarray(adapter.upper_bounds)
            vector = lower + 0.25 * (upper - lower)
            adapter.evaluate(vector, 7, ledger, f"r4-cdf-{profile}-{index}")
            if ledger.snapshot()["cfe"] != 1:
                raise R4ValidationError(
                    "dynamic bridge bypassed the shared ledger"
                )
            dynamic_count += 1

    rolling = WGTRRPublicAdapter.from_known_answer()
    rolling.freeze_information(0, None)
    rolling_ledger = EvaluationLedger(max_cfe=1)
    result = rolling.evaluate(
        np.zeros(rolling.decision_dimension),
        0,
        rolling_ledger,
        "r4-rolling",
    )
    rolling.execute(
        rolling.first_action(np.zeros(rolling.decision_dimension)),
        0,
        result.feasible,
        rolling_ledger,
    )
    rolling_snapshot = rolling_ledger.snapshot()
    if (
        rolling_snapshot["cfe"] != 1
        or rolling_snapshot["atomic_model_steps"] != 6
        or rolling_snapshot["execution_transition_count"] != 1
    ):
        raise R4ValidationError("rolling bridge accounting differs")

    return {
        "static_problem_bindings_exercised": static_count,
        "dynamic_profile_problem_bindings_exercised": dynamic_count,
        "rolling_known_answer_bindings_exercised": 1,
    }


def _exercise_comparator_construction() -> int:
    methods = [
        MatchedParetoDE(mode="fixed"),
        MatchedParetoDE(mode="jde"),
        MatchedParetoDE(mode="shade"),
        JMetalComparator(mode="gde3"),
        JMetalComparator(mode="nsgaii_static"),
        JMetalComparator(mode="nsgaii_dynamic_restart"),
        FixedEnergyDeficitBaseline(),
        ConventionalRollingPlannerBaseline(),
    ]
    identities = [method.identity() for method in methods]
    if any(
        identity.get("effect_execution_allowed") is not False
        for identity in identities
    ):
        raise R4ValidationError("executable comparator opened effect execution")
    _require_unique(
        [str(identity["method_id"]) for identity in identities],
        "constructed method_id",
    )
    return len(methods)


def validate_r4() -> dict[str, Any]:
    comparators = _validate_document("comparator_manifest")
    benchmarks = _validate_document("benchmark_registry")
    licenses = _validate_document("license_record")
    _validate_semantics(comparators, benchmarks, licenses)
    bridge_counts = _exercise_benchmark_bridges()
    method_count = _exercise_comparator_construction()
    return {
        "validator": "WGT-V11-R4-EXECUTABLE-BINDING-VALIDATOR-01",
        "status": "PASS",
        "method_categories_bound": method_count,
        **bridge_counts,
        "effect_estimation_performed": False,
        "participant_data_accessed": False,
        "hidden_instance_accessed_or_generated": False,
        "results_analysis_performed": False,
        "distribution_authorized": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compact",
        action="store_true",
        help="emit one-line JSON",
    )
    args = parser.parse_args()
    summary = validate_r4()
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
