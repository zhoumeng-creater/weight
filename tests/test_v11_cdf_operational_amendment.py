from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator
import numpy as np
import pytest

from benchmark_adapters.cdf_operational import (
    CDFDomainUndefinedError,
    CDFOperationalEvaluator,
    CDF_OPERATIONAL_AUTHORITY_ID,
    CDF_OPERATIONAL_SUITE_ID,
)
from benchmark_adapters.r4_evaluators import CDFEvaluator
from evaluation.contracts import NumericalEvaluationError
from evaluation.evaluator import BatchEvaluationUnavailableBeforeEntry
from evaluation.ledger import EvaluationLedger
from formal_execution.adapters import make_corrective_cdf_adapter


ROOT = Path(__file__).resolve().parents[1]
AMENDMENT_PATH = (
    ROOT
    / "config"
    / "r8c_e1e2"
    / "r8c_e1e2_cdf_operational_authority_amendment.json"
)
AMENDMENT_SCHEMA_PATH = AMENDMENT_PATH.with_suffix(".schema.json")


_KAT_EVENT_7_QUARTER_BOUNDS = {
    1: (
        (0.411890912788997, 0.7287756576655868),
        (0.29468519119636477, 0.27788708199919787),
    ),
    2: (
        (18.376616247739932, 22.42172942406179),
        (0.028662844557722514,),
    ),
    3: (
        (0.820778849246565, 1.328469550255483),
        (0.5814447780382868,),
    ),
    4: (
        (2.5331153969862594, 3.2513782010219328),
        (-4.241714177066079,),
    ),
    5: (
        (32.85965241524137, 35.27701984023672),
        (1.522213595499958,),
    ),
    6: (
        (16.363401266761816, 14.374554648812023),
        (0.4051838992327912, 0.5400839926159475),
    ),
    7: (
        (9.222689905851944, 9.9102092737056),
        (-16.13274397212583,),
    ),
    8: (
        (0.5073159294558478, 1.064403169595413),
        (-1.577697398979474,),
    ),
    9: (
        (6.326017727512383, 6.8992604249388085),
        (0.38420659964791537, 0.5319841599961043),
    ),
    10: (
        (29.97932756110928, 22.647009457496488),
        (-0.24202969626716686, 0.29287039711598944),
    ),
    11: (
        (7.004380436663206, 6.437166556824805),
        (0.2131966011250105,),
    ),
    12: (
        (2.0232740095538397, 3.568183858793528),
        (-2.342245754729499e-08,),
    ),
    13: (
        (6.4049377857689915, 6.32783472958726),
        (-10.576928457076937,),
    ),
    14: (
        (0.2801364758968926, 0.800499151250937),
        (-0.2926879573949944,),
    ),
    15: (
        (3.998802267701416, 4.737499999999999),
        (-19.348509190143485,),
    ),
}

_FIRST_FIVE_DYNAMIC_SEEDS = (
    "1814705672717120344",
    "11510044127855585889",
    "2013063862857590834",
    "9940308221477475016",
    "10545341458691982268",
)

_SCHEDULE_COMMITMENTS = {
    "CDF-HARSH": (
        "be9d3b19fedb1edaf1b7bc858b74f155227a206a1953b604945096d6d4f14a14",
        "1a11f4c5122c42b80b1f82f0f3e3e947232e5f364533c93d4365f2d43649cca7",
        "93c5197e1c8005f7b4b47f8c06d1a2337855a28387fd7200f469b1c2a5127429",
        "93612bee4abfbbf083ef22c897448646e18814223040496efbfd666152ec0410",
        "f49c73fe2eca597f7484d44e35bc0c6dce127bfa6d0f9bab47e460847ab9cc6c",
    ),
    "CDF-MILD": (
        "3ba14455f1318bca905a9befa20e60948cce3f1f1674b73e41a0c37f3d8989ee",
        "db6f043b1e16a9d0a9158885968c287252b8d6d037823aaaa5b0e6cfe2797c18",
        "b397cfc16fc567e58edd4a7eaead3a80422c7e137492d53ea37b50ff522991db",
        "a6b0d822dea7580b71d93e778e43a394d4dd43783dd8d9af390c38f7401066eb",
        "a83c86ac488d21c4d99e5f806ba064b415847c6cd40b591a61780f93242c9f70",
    ),
}


@pytest.mark.parametrize("problem_index", range(1, 16))
def test_all_fifteen_operational_equations_have_frozen_known_answers(
    problem_index: int,
) -> None:
    evaluator = CDFOperationalEvaluator(
        problem_index=problem_index,
        profile="CDF-HARSH",
        environment_seed=17,
    )
    lower = np.asarray(evaluator.lower_bounds)
    upper = np.asarray(evaluator.upper_bounds)
    vector = lower + 0.25 * (upper - lower)

    assert evaluator(vector, 7) == _KAT_EVENT_7_QUARTER_BOUNDS[problem_index]


def test_cdf1_corrective_constraint_matches_oracle_exponent_placement() -> None:
    vector = np.asarray([0.37, 0.23, 0.41, 0.67, 0.19] * 2)
    evaluator = CDFOperationalEvaluator(1, "CDF-HARSH", 17)
    objectives, project_constraints = evaluator(vector, 7)
    gt = math.sin(0.5 * math.pi * 7 / 5)
    k1 = 0.5 * (1.0 - vector[0]) - (1.0 - vector[0]) ** 2
    k2 = (
        0.25 * math.sqrt(1.0 - vector[0])
        - 0.5 * (1.0 - vector[0])
    )
    expected_upstream = (
        vector[1]
        - vector[0]
        ** (
            1.0
            + abs(gt)
            - math.copysign(math.sqrt(abs(k1)), k1)
        ),
        vector[3]
        - vector[0]
        ** (
            1.375
            + abs(gt)
            - math.copysign(math.sqrt(abs(k2)), k2)
        ),
    )

    assert objectives == CDFEvaluator(1, "CDF-HARSH", 17)(vector, 7)[0]
    assert project_constraints == tuple(-value for value in expected_upstream)
    assert project_constraints != CDFEvaluator(
        1, "CDF-HARSH", 17
    )(vector, 7)[1]


def test_corrective_factory_uses_new_suite_and_preserves_historical_binding() -> None:
    corrective = make_corrective_cdf_adapter(
        1,
        profile="CDF-HARSH",
        environment_seed=17,
    )
    historical = CDFEvaluator(1, "CDF-HARSH", 17)

    assert corrective.identity()["target_suite_id"] == CDF_OPERATIONAL_SUITE_ID
    assert corrective.fixture_evaluator_sha256 != historical.binding_sha256
    assert CDF_OPERATIONAL_AUTHORITY_ID


def test_cdf9_nonreal_source_domain_is_a_charged_typed_failure() -> None:
    adapter = make_corrective_cdf_adapter(
        9,
        profile="CDF-HARSH",
        environment_seed=17,
    )
    adapter.freeze_information(5, None)
    vector = np.zeros(10)
    vector[0] = 1.0
    ledger = EvaluationLedger(max_cfe=1)

    with pytest.raises(NumericalEvaluationError, match="nonrecoverable"):
        adapter.evaluate(vector, 5, ledger, "cdf9-domain")

    assert ledger.snapshot()["cfe"] == 1
    assert ledger.snapshot()["evaluation_failures"] == 1
    assert ledger.evaluation_failures[0].failure_type == (
        "CDFDomainUndefinedError"
    )
    assert "outside its real domain" in ledger.evaluation_failures[0].reason


def test_cdf9_invalid_batch_falls_back_before_any_ledger_entry() -> None:
    adapter = make_corrective_cdf_adapter(
        9,
        profile="CDF-HARSH",
        environment_seed=17,
    )
    adapter.freeze_information(5, None)
    valid = np.zeros(10)
    invalid = np.zeros(10)
    invalid[0] = 1.0
    ledger = EvaluationLedger(max_cfe=2)

    with pytest.raises(BatchEvaluationUnavailableBeforeEntry):
        adapter.evaluate_batch(
            [valid, invalid],
            5,
            ledger,
            ["valid", "invalid"],
        )

    assert ledger.snapshot()["cfe"] == 0
    assert ledger.snapshot()["evaluation_failures"] == 0


@pytest.mark.parametrize("profile", ("CDF-HARSH", "CDF-MILD"))
def test_cdf13_first_five_seed_schedules_are_frozen(profile: str) -> None:
    severity = 5 if profile == "CDF-HARSH" else 10
    for seed, expected_commitment in zip(
        _FIRST_FIVE_DYNAMIC_SEEDS,
        _SCHEDULE_COMMITMENTS[profile],
        strict=True,
    ):
        evaluator = CDFOperationalEvaluator(13, profile, int(seed))
        assert evaluator.environment_schedule_commitment == expected_commitment
        previous = evaluator._time_vector(0)
        assert previous == (0.0,) * 5
        for event_id in range(1, 60):
            current = evaluator._time_vector(event_id)
            increments = [
                (current[index] - previous[index]) * severity
                for index in range(5)
            ]
            assert sum(
                math.isclose(value, 1.0, abs_tol=1e-12)
                for value in increments
            ) == 1
            assert sum(
                math.isclose(value, 0.0, abs_tol=1e-12)
                for value in increments
            ) == 4
            previous = current
        assert math.isclose(sum(previous), 59 / severity)


@pytest.mark.parametrize("profile", ("CDF-HARSH", "CDF-MILD"))
def test_all_problem_event_batches_are_exact_ordered_scalar_kernels(
    profile: str,
) -> None:
    for problem_index in range(1, 16):
        evaluator = CDFOperationalEvaluator(problem_index, profile, 17)
        lower = np.asarray(evaluator.lower_bounds)
        upper = np.asarray(evaluator.upper_bounds)
        for event_id in range(60):
            vectors = []
            for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
                vector = lower + fraction * (upper - lower)
                if problem_index == 9:
                    gt = math.sin(
                        0.5 * math.pi * event_id / evaluator.severity_ns
                    )
                    mt = 0.5 + abs(gt)
                    vector[0] = 0.5 * min(1.0, 1.0 / mt)
                vectors.append(vector)
            expected = tuple(
                evaluator(vector, event_id) for vector in vectors
            )
            assert evaluator.evaluate_batch(vectors, event_id) == expected


def test_cdf9_evaluator_exposes_specific_domain_error() -> None:
    evaluator = CDFOperationalEvaluator(9, "CDF-HARSH", 17)
    vector = np.zeros(10)
    vector[0] = 1.0

    with pytest.raises(CDFDomainUndefinedError):
        evaluator(vector, 5)


def test_cdf_operational_amendment_is_strict_and_source_bound() -> None:
    amendment = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    schema = json.loads(
        AMENDMENT_SCHEMA_PATH.read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    validator.validate(amendment)

    authority = amendment["authority_binding"]
    oracle = authority["author_oracle"]
    assert oracle == {
        "repository": "https://bitbucket.org/Pag1c18/cmlsga",
        "commit": "1926a5a1c89adf0a5e5e70449adbec62750a108a",
        "path": "MLSGA/Fit_Functions.cpp",
        "bytes": 461394,
        "sha256": (
            "48b2c256f4bdec6ed4f81f8edd82a037"
            "53bc51550776e1ae84b2d6fcbc18fa7a"
        ),
    }
    for binding in (
        authority["audit_document"],
        amendment["historical_freeze"]["r5_contract"],
        amendment["historical_freeze"]["r4_benchmark_registry"],
        *amendment["implementation_bindings"].values(),
        amendment["reference_catalog_binding"]["manifest"],
        amendment["reference_catalog_binding"]["manifest_schema"],
        amendment["reference_catalog_binding"]["artifact"],
    ):
        path = ROOT / binding["path"]
        assert path.stat().st_size == binding["bytes"]
        assert sha256(path.read_bytes()).hexdigest() == binding["sha256"]
    artifact = amendment["reference_catalog_binding"]["artifact"]
    assert len((ROOT / artifact["path"]).read_bytes().splitlines()) == (
        artifact["lines"]
    )
    for binding in amendment["verification"]["test_files"]:
        path = ROOT / binding["path"]
        assert sha256(path.read_bytes()).hexdigest() == binding["sha256"]

    assert amendment["cdf9_failure_policy"][
        "undefined_real_domain_action"
    ] == "CHARGE_ONCE_AND_RAISE_CDFDomainUndefinedError"
    assert amendment["cdf9_failure_policy"][
        "external_terminal"
    ] == "REJECT_NUMERICAL"
    assert amendment["effect_boundary"]["effect_outputs_inspected"] is False

    authority_drift = deepcopy(amendment)
    authority_drift["authority_binding"]["author_oracle"][
        "commit"
    ] = "0" * 40
    assert list(validator.iter_errors(authority_drift))

    extension_drift = deepcopy(amendment)
    extension_drift["cdf9_failure_policy"][
        "domain_extension_allowed"
    ] = True
    assert list(validator.iter_errors(extension_drift))
