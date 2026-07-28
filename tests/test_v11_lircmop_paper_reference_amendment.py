from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from math import cos, pi, sin
from pathlib import Path

from jsonschema import Draft202012Validator
import numpy as np
import pytest

from analysis.checkpoint_metrics import AnalyticReferenceScale
from analysis.reference_fronts import (
    FINITE_FRONT_COMPLETENESS_ASSERTION,
    ExactReferenceExtrema,
    FiniteParetoFront,
    ReferenceArtifactError,
    ReferenceIdentity,
)
from benchmark_adapters.lircmop_paper import (
    LIRCMOP_PAPER_SUITE_ID,
    LIRCMOPPaperEvaluator,
)
from benchmark_adapters.r4_evaluators import CDFEvaluator
from formal_execution.adapters import (
    make_corrective_lircmop_adapter,
    make_formal_lircmop_adapter,
)

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT_PATH = (
    ROOT
    / "config"
    / "r8c_e1e2"
    / "r8c_e1e2_lircmop_reference_amendment.json"
)
AMENDMENT_SCHEMA_PATH = AMENDMENT_PATH.with_suffix(".schema.json")


KNOWN_ANSWERS = {
    1: ((0.25, 0.9375), (0.255, 0.255)),
    2: ((0.25, 0.5), (0.255, 0.255)),
    3: ((0.25, 0.9375), (0.255, 0.255, 0.4999999999999994)),
    4: ((0.25, 0.5), (0.255, 0.255, 0.4999999999999994)),
    5: (
        (0.9557, 1.2057),
        (-0.03678937000000007, -0.9076945262500004),
    ),
    6: (
        (0.9557, 1.6432),
        (-0.028967778203125044, -1.0295177782031248),
    ),
    7: (
        (0.9557, 1.2057),
        (
            0.09201569944444446,
            -0.33774101068888895,
            -1.7732764967999994,
        ),
    ),
    8: (
        (0.9557, 1.6432),
        (
            0.08849017861111111,
            -0.1907756643347223,
            -1.4519377780499996,
        ),
    ),
    9: (
        (0.426425, 1.59909375),
        (-0.05239293476182719, 1.4066544402853678),
    ),
    10: (
        (0.426425, 0.85285),
        (-0.03356473072265628, 0.6986262793979707),
    ),
    11: (
        (0.426425, 0.85285),
        (-0.18275332686249993, 1.7986262793979708),
    ),
    12: (
        (0.426425, 1.59909375),
        (-0.22563404587293848, 1.9066544402853678),
    ),
    13: (
        (
            0.6030560183349472,
            1.455906018334947,
            0.6527431305851337,
        ),
        (-6.6423186669679986, -0.23160548046799997),
    ),
    14: (
        (
            0.6030560183349472,
            1.455906018334947,
            0.6527431305851337,
        ),
        (
            -6.6423186669679986,
            -0.23160548046799997,
            0.053490688056999844,
        ),
    ),
}


def _known_vector(problem_index: int) -> tuple[float, ...]:
    if problem_index <= 4:
        x1 = 0.25
        values = [x1] * 30
        for index in range(2, 29, 2):
            values[index] = sin(0.5 * pi * x1)
        for index in range(1, 30, 2):
            values[index] = cos(0.5 * pi * x1)
        return tuple(values)
    if problem_index <= 12:
        x1 = 0.25
        values = [x1] * 30
        for index in range(2, 29, 2):
            values[index] = sin(0.5 * (index + 1) * pi * x1 / 30.0)
        for index in range(1, 30, 2):
            values[index] = cos(0.5 * (index + 1) * pi * x1 / 30.0)
        return tuple(values)
    values = [0.5] * 30
    values[0] = 0.25
    values[1] = 0.75
    return tuple(values)


@pytest.mark.parametrize("problem_index", range(1, 15))
def test_all_lircmop_paper_equations_match_known_answers(
    problem_index: int,
) -> None:
    evaluator = LIRCMOPPaperEvaluator(problem_index)
    actual = evaluator(_known_vector(problem_index), 0)
    expected = KNOWN_ANSWERS[problem_index]
    np.testing.assert_allclose(
        actual[0],
        expected[0],
        rtol=0.0,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        actual[1],
        expected[1],
        rtol=0.0,
        atol=2e-15,
    )
    assert len(actual[0]) == len(evaluator.objective_names)
    assert len(actual[1]) == len(evaluator.constraint_names)


@pytest.mark.parametrize("problem_index", range(1, 15))
def test_lircmop_paper_batch_is_byte_exact_to_scalar(
    problem_index: int,
) -> None:
    evaluator = LIRCMOPPaperEvaluator(problem_index)
    rng = np.random.default_rng(20260727 + problem_index)
    vectors = (
        _known_vector(problem_index),
        tuple(0.0 for _ in range(30)),
        tuple(1.0 for _ in range(30)),
        *(
            tuple(float(value) for value in row)
            for row in rng.random((4, 30))
        ),
    )
    scalar = tuple(evaluator(vector, 0) for vector in vectors)
    batch = evaluator.evaluate_batch(vectors, 0)
    assert batch == scalar
    assert json.dumps(batch, separators=(",", ":")).encode() == (
        json.dumps(scalar, separators=(",", ":")).encode()
    )


def test_corrective_factory_uses_paper_identity_without_rewriting_r4() -> None:
    corrective = make_corrective_lircmop_adapter(5)
    historical = make_formal_lircmop_adapter(5)
    assert corrective.identity()["target_suite_id"] == LIRCMOP_PAPER_SUITE_ID
    assert (
        historical.identity()["target_suite_id"]
        == "LIR-CMOP-JMETALPY-1.7.0"
    )
    assert corrective.fixture_evaluator_sha256 != (
        historical.fixture_evaluator_sha256
    )


def _static_identity() -> ReferenceIdentity:
    evaluator = LIRCMOPPaperEvaluator(1)
    return ReferenceIdentity(
        suite_id=LIRCMOP_PAPER_SUITE_ID,
        problem_id=evaluator.problem_id,
        evaluator_binding_sha256=evaluator.binding_sha256,
    )


def test_exact_extrema_feed_existing_nhv_scale_without_10000_points() -> None:
    extrema = ExactReferenceExtrema(
        identity=_static_identity(),
        minima=(0.5, 0.5),
        maxima=(1.5, 1.5),
        derivation_id="LIRCMOP1_CLOSED_FORM_EXTREMA_V1",
    )
    scale = extrema.to_analytic_reference_scale()
    assert isinstance(scale, AnalyticReferenceScale)
    assert scale.minima == (0.5, 0.5)
    assert scale.maxima == (1.5, 1.5)
    assert scale.point_count is None


def test_finite_true_pf_stores_every_unique_point_once() -> None:
    front = FiniteParetoFront.from_points(
        identity=_static_identity(),
        points=[
            (0.5, 1.5),
            (1.0, 1.0),
            (0.5, 1.5),
            (1.5, 0.5),
        ],
        derivation_id="FINITE_TEST_PF_V1",
    )
    assert front.points == (
        (0.5, 1.5),
        (1.0, 1.0),
        (1.5, 0.5),
    )
    assert (
        front.completeness_assertion
        == FINITE_FRONT_COMPLETENESS_ASSERTION
    )
    extrema = front.extrema()
    assert extrema.minima == (0.5, 0.5)
    assert extrema.maxima == (1.5, 1.5)
    scale = extrema.to_analytic_reference_scale()
    assert scale.point_count == 3


def test_cdf13_reference_identity_binds_schedule_seed_and_evaluator() -> None:
    first = CDFEvaluator(
        problem_index=13,
        profile="CDF-HARSH",
        environment_seed=20260727,
    )
    second = CDFEvaluator(
        problem_index=13,
        profile="CDF-HARSH",
        environment_seed=20260728,
    )
    identity = ReferenceIdentity.for_cdf13(
        first,
        event_id=17,
        master_seed_u64="20260727",
    )
    other_seed = ReferenceIdentity.for_cdf13(
        second,
        event_id=17,
        master_seed_u64="20260728",
    )
    assert identity.profile == "CDF-HARSH"
    assert identity.event_id == 17
    assert identity.master_seed_u64 == "20260727"
    assert identity.time_vector == tuple(
        first.release_metadata(17)["current_time_vector"]
    )
    assert identity.evaluator_binding_sha256 == first.binding_sha256
    assert identity.identity_sha256 != other_seed.identity_sha256


def test_cdf13_reference_identity_fails_closed_on_seed_mismatch() -> None:
    evaluator = CDFEvaluator(
        problem_index=13,
        profile="CDF-MILD",
        environment_seed=7,
    )
    with pytest.raises(ReferenceArtifactError, match="master seed"):
        ReferenceIdentity.for_cdf13(
            evaluator,
            event_id=3,
            master_seed_u64="8",
        )


def test_lircmop_reference_amendment_is_strict_and_source_bound() -> None:
    contract = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    schema = json.loads(
        AMENDMENT_SCHEMA_PATH.read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    validator.validate(contract)

    bindings = contract["implementation_bindings"]
    for binding in bindings.values():
        path = ROOT / binding["path"]
        assert path.stat().st_size == binding["bytes"]
        assert sha256(path.read_bytes()).hexdigest() == binding["sha256"]
    verification = contract["verification"]
    test_path = ROOT / verification["test_file"]
    assert sha256(test_path.read_bytes()).hexdigest() == (
        verification["test_file_sha256"]
    )
    assert contract["reference_representation"][
        "arbitrary_continuous_pf_point_count_required"
    ] is False
    assert contract["reference_representation"][
        "exact_extrema_required"
    ] is True
    assert contract["reference_representation"][
        "finite_pf_store_all_unique_points"
    ] is True
    catalog = contract["reference_catalog_binding"]
    for key in ("manifest", "manifest_schema", "artifact"):
        binding = catalog[key]
        path = ROOT / binding["path"]
        assert path.stat().st_size == binding["bytes"]
        assert sha256(path.read_bytes()).hexdigest() == binding["sha256"]
    assert len(
        (ROOT / catalog["artifact"]["path"]).read_bytes().splitlines()
    ) == catalog["artifact"]["lines"]
    assert catalog["identity_scope"]["actual_total"] == 2294
    assert contract["effect_boundary"]["effect_outputs_inspected"] is False

    authority_drift = deepcopy(contract)
    authority_drift["effect_boundary"][
        "authorizes_formal_effect_execution"
    ] = True
    assert list(validator.iter_errors(authority_drift))

    sample_count_drift = deepcopy(contract)
    sample_count_drift["reference_representation"][
        "arbitrary_continuous_pf_point_count_required"
    ] = True
    assert list(validator.iter_errors(sample_count_drift))
