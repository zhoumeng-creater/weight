from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator
import numpy as np
import pytest

from analysis.reference_catalog import (
    CDF_REFERENCE_SEEDS,
    REFERENCE_CATALOG_EXPECTED_IDENTITIES,
    _cdf11_finite_front,
    _linked_w_minimum,
    derive_cdf_reference,
    derive_lircmop_reference,
    iter_formal_reference_derivations,
    load_reference_catalog,
    reference_derivation_from_record,
)
from analysis.reference_fronts import ReferenceArtifactError
from benchmark_adapters.cdf_operational import CDFOperationalEvaluator
from benchmark_adapters.lircmop_paper import LIRCMOPPaperEvaluator


ROOT = Path(__file__).resolve().parents[1]
CATALOG = (
    ROOT
    / "config/r8c_e1e2/reference_catalog/reference_artifacts.jsonl"
)
MANIFEST = (
    ROOT
    / "config/r8c_e1e2/reference_catalog/reference_catalog_manifest.json"
)
MANIFEST_SCHEMA = MANIFEST.with_suffix(".schema.json")


def _assert_feasible(
    result: tuple[Sequence[float], Sequence[float]],
    *,
    tolerance: float = 1e-10,
) -> tuple[float, ...]:
    objectives, constraints = result
    assert max(constraints, default=-math.inf) <= tolerance
    return tuple(float(value) for value in objectives)


def _cdf5_or_11_decision(
    x_value: float,
    *,
    gt: float,
    decision_bound: float,
) -> np.ndarray:
    vector = np.zeros(10)
    vector[0] = x_value
    for index in range(2, 11):
        wave = (
            math.cos(6.0 * math.pi * x_value + index * math.pi / 10.0)
            if index % 2 == 1
            else math.sin(
                6.0 * math.pi * x_value + index * math.pi / 10.0
            )
        )
        base = 0.8 * x_value * wave + gt
        residual = 0.0
        if index == 2:
            _minimum, residual = _linked_w_minimum(
                x_value,
                gt=gt,
                decision_bound=decision_bound,
                return_arg=True,
            )
        vector[index - 1] = base + residual
    return vector


def _cdf8_ideal_decision(x_value: float) -> np.ndarray:
    vector = np.zeros(10)
    vector[0] = x_value
    for index in range(2, 11):
        exponent = 0.5 * (2.0 + 3.0 * (index - 2) / 8.0)
        vector[index - 1] = x_value**exponent
    return vector


def _cdf13_ideal_decision(
    evaluator: CDFOperationalEvaluator,
    event_id: int,
    x_value: float,
) -> np.ndarray:
    time_vector = evaluator._time_vector(event_id)
    g = tuple(math.sin(0.5 * math.pi * value) for value in time_vector)
    k_t1 = math.ceil(10.0 * g[0])
    vector = np.zeros(10)
    vector[0] = x_value
    for index in range(2, 11):
        vector[index - 1] = (
            math.sin(
                6.0 * math.pi * x_value
                + (index + k_t1) * math.pi / 10.0
            )
            + g[1]
        )
    return vector


def _cdf15_ideal_decision(x_value: float) -> np.ndarray:
    vector = np.zeros(10)
    vector[0] = x_value
    for index in range(2, 11):
        vector[index - 1] = math.sin(
            6.0 * math.pi * x_value + index * math.pi / 10.0
        )
    return vector


def _lir_indexed_ideal_decision(x_value: float) -> np.ndarray:
    vector = np.zeros(30)
    vector[0] = x_value
    for zero_based in range(1, 30):
        one_based = zero_based + 1
        angle = 0.5 * one_based * math.pi * x_value / 30.0
        vector[zero_based] = (
            math.sin(angle)
            if zero_based % 2 == 0
            else math.cos(angle)
        )
    return vector


def test_all_2294_formal_reference_identities_are_unique() -> None:
    derivations = tuple(iter_formal_reference_derivations())
    identities = [
        item.extrema.identity.identity_sha256 for item in derivations
    ]

    assert len(derivations) == REFERENCE_CATALOG_EXPECTED_IDENTITIES
    assert len(identities) == len(set(identities))
    assert sum(
        item.extrema.identity.problem_id.startswith("LIRCMOP")
        for item in derivations
    ) == 14
    assert sum(
        item.extrema.identity.problem_id == "CDF13"
        for item in derivations
    ) == 600


@pytest.mark.parametrize(
    ("problem_index", "expected_minima", "expected_maxima"),
    (
        (1, (0.5, 0.5), (1.5, 1.5)),
        (
            3,
            (0.5 + 1 / 120, 0.5 + 1 - (113 / 120) ** 2),
            (0.5 + 113 / 120, 0.5 + 1 - (1 / 120) ** 2),
        ),
        (5, (0.7057, 0.7057), (1.7057, 1.7057)),
        (13, (0.0, 0.0, 0.0), (1.7057, 1.7057, 1.7057)),
        (14, (0.0, 0.0, 0.0), (1.75, 1.75, 1.75)),
    ),
)
def test_lircmop_closed_form_extrema(
    problem_index: int,
    expected_minima: tuple[float, ...],
    expected_maxima: tuple[float, ...],
) -> None:
    reference = derive_lircmop_reference(problem_index).extrema

    assert reference.minima == expected_minima
    assert reference.maxima == expected_maxima


@pytest.mark.parametrize("problem_index", (7, 8, 9, 10, 11, 12))
def test_lircmop_numerical_axis_extrema_have_reachable_witnesses(
    problem_index: int,
) -> None:
    derivation = derive_lircmop_reference(problem_index)
    evaluator = LIRCMOPPaperEvaluator(problem_index)
    maximum_x, maximum_y = derivation.extrema.maxima

    y_vector = _lir_indexed_ideal_decision(0.0)
    if problem_index in {7, 8}:
        target_g2 = (maximum_y - 1.7057) / 10.0
    else:
        target_g2 = (maximum_y / 1.7057 - 1.0) / 10.0
    y_vector[1] -= math.sqrt(max(0.0, target_g2))
    y_objectives = _assert_feasible(evaluator(y_vector, 0))
    expected_y_f1 = 0.7057 if problem_index in {7, 8} else 0.0
    assert y_objectives[0] == expected_y_f1
    assert math.isclose(y_objectives[1], maximum_y, abs_tol=2e-10)

    x_vector = _lir_indexed_ideal_decision(1.0)
    if problem_index in {7, 8}:
        target_g1 = (maximum_x - 1.7057) / 10.0
    else:
        target_g1 = (maximum_x / 1.7057 - 1.0) / 10.0
    x_vector[2] += math.sqrt(max(0.0, target_g1))
    x_objectives = _assert_feasible(evaluator(x_vector, 0))
    assert math.isclose(x_objectives[0], maximum_x, abs_tol=2e-10)
    expected_x_f2 = 0.7057 if problem_index in {7, 8} else 0.0
    assert x_objectives[1] == expected_x_f2


@pytest.mark.parametrize("profile", ("CDF-HARSH", "CDF-MILD"))
def test_cdf5_global_minimum_is_below_independent_dense_grid(
    profile: str,
) -> None:
    severity = 5 if profile == "CDF-HARSH" else 10
    grid = np.linspace(0.0, 1.0, (1 << 19) + 1)
    for event_id in range(60):
        gt = math.sin(0.5 * math.pi * event_id / severity)
        shift = abs(gt)
        dense = (
            1.0
            - grid
            + shift
            + np.asarray(
                _linked_w_minimum(
                    grid,
                    gt=gt,
                    decision_bound=2.0,
                )
            )
        )
        derivation = derive_cdf_reference(
            5,
            profile=profile,
            event_id=event_id,
        )
        minimum = derivation.extrema.minima[1]
        x_at_minimum = (
            derivation.extrema.maxima[0]
            - derivation.extrema.minima[0]
        )

        assert minimum <= float(np.min(dense)) + 1e-12
        spacing = 1.0 / (1 << 19)
        for delta in (-spacing, spacing):
            nearby = min(1.0, max(0.0, x_at_minimum + delta))
            nearby_value = (
                1.0
                - nearby
                + shift
                + float(
                    _linked_w_minimum(
                        nearby,
                        gt=gt,
                        decision_bound=2.0,
                    )
                )
            )
            assert minimum <= nearby_value + 1e-12


@pytest.mark.parametrize("profile", ("CDF-HARSH", "CDF-MILD"))
def test_cdf5_extreme_points_are_reachable(profile: str) -> None:
    severity = 5 if profile == "CDF-HARSH" else 10
    for event_id in range(60):
        gt = math.sin(0.5 * math.pi * event_id / severity)
        derivation = derive_cdf_reference(
            5,
            profile=profile,
            event_id=event_id,
        )
        x_at_minimum = (
            derivation.extrema.maxima[0]
            - derivation.extrema.minima[0]
        )
        evaluator = CDFOperationalEvaluator(5, profile, 0)
        objectives = _assert_feasible(
            evaluator(
                _cdf5_or_11_decision(
                    x_at_minimum,
                    gt=gt,
                    decision_bound=2.0,
                ),
                event_id,
            )
        )
        assert np.allclose(
            objectives,
            (
                derivation.extrema.maxima[0],
                derivation.extrema.minima[1],
            ),
            rtol=0.0,
            atol=2e-10,
        )


@pytest.mark.parametrize("profile", ("CDF-HARSH", "CDF-MILD"))
def test_cdf11_complete_points_dominate_dense_reachable_envelope(
    profile: str,
) -> None:
    severity = 5 if profile == "CDF-HARSH" else 10
    grid = np.linspace(0.0, 1.0, 20_001)
    for event_id in range(60):
        gt = math.sin(0.5 * math.pi * event_id / severity)
        finite = np.asarray(_cdf11_finite_front(gt))
        ripple = 0.15 * np.abs(np.sin(math.pi * (20.0 * grid + gt)))
        f1 = grid + ripple
        f2 = (
            1.0
            - grid
            + np.asarray(
                _linked_w_minimum(
                    grid,
                    gt=0.0,
                    decision_bound=1.0,
                )
            )
            + ripple
        )
        dominated = (
            (finite[:, None, 0] <= f1[None, :] + 1e-12)
            & (finite[:, None, 1] <= f2[None, :] + 1e-12)
        ).any(axis=0)

        assert bool(np.all(dominated))
        derivation = derive_cdf_reference(
            11,
            profile=profile,
            event_id=event_id,
        )
        assert derivation.finite_front is not None
        assert np.allclose(
            derivation.finite_front.points,
            finite,
            rtol=0.0,
            atol=2e-15,
        )


def test_cdf11_all_finite_points_have_decision_witnesses() -> None:
    profile = "CDF-HARSH"
    event_id = 7
    gt = math.sin(0.5 * math.pi * event_id / 5)
    evaluator = CDFOperationalEvaluator(11, profile, 0)
    expected = derive_cdf_reference(
        11,
        profile=profile,
        event_id=event_id,
    ).finite_front
    assert expected is not None

    candidates = {0.0, 1.0}
    for integer in range(math.floor(gt) - 1, math.ceil(gt) + 22):
        value = (integer - gt) / 20.0
        if 0.0 <= value <= 1.0:
            candidates.add(value)
    evaluated = []
    for x_value in candidates:
        evaluated.append(
            _assert_feasible(
                evaluator(
                    _cdf5_or_11_decision(
                        x_value,
                        gt=0.0,
                        decision_bound=1.0,
                    ),
                    event_id,
                )
            )
        )
    for point in expected.points:
        assert any(
            np.allclose(point, value, rtol=0.0, atol=2e-12)
            for value in evaluated
        )


@pytest.mark.parametrize(
    ("problem_index", "profile", "event_id", "seed"),
    (
        (8, "CDF-HARSH", 5, None),
        (8, "CDF-MILD", 7, None),
        (13, "CDF-HARSH", 17, CDF_REFERENCE_SEEDS[0]),
        (13, "CDF-MILD", 43, CDF_REFERENCE_SEEDS[-1]),
        (15, "CDF-HARSH", 7, None),
        (15, "CDF-MILD", 11, None),
    ),
)
def test_cdf_constrained_curve_extrema_have_evaluator_witnesses(
    problem_index: int,
    profile: str,
    event_id: int,
    seed: str | None,
) -> None:
    derivation = derive_cdf_reference(
        problem_index,
        profile=profile,
        event_id=event_id,
        master_seed_u64=seed,
    )
    evaluator = CDFOperationalEvaluator(
        problem_index,
        profile,
        0 if seed is None else int(seed),
    )
    if problem_index == 8:
        minimum_x = derivation.extrema.minima[0]
        maximum_x = derivation.extrema.maxima[0]
        decision = _cdf8_ideal_decision
    elif problem_index == 13:
        time_vector = evaluator._time_vector(event_id)
        random_shift = abs(math.sin(0.5 * math.pi * time_vector[2]))
        minimum_x = derivation.extrema.minima[0] - random_shift
        maximum_x = derivation.extrema.maxima[0] - random_shift
        decision = lambda value: _cdf13_ideal_decision(  # noqa: E731
            evaluator,
            event_id,
            value,
        )
    else:
        minimum_x = derivation.extrema.minima[0]
        maximum_x = derivation.extrema.maxima[0]
        decision = _cdf15_ideal_decision

    minimum_objectives = _assert_feasible(
        evaluator(decision(minimum_x), event_id),
        tolerance=3e-10,
    )
    maximum_objectives = _assert_feasible(
        evaluator(decision(maximum_x), event_id),
        tolerance=3e-10,
    )
    assert np.allclose(
        (
            minimum_objectives[0],
            maximum_objectives[1],
        ),
        derivation.extrema.minima,
        rtol=0.0,
        atol=3e-10,
    )
    assert np.allclose(
        (
            maximum_objectives[0],
            minimum_objectives[1],
        ),
        derivation.extrema.maxima,
        rtol=0.0,
        atol=3e-10,
    )


@pytest.mark.parametrize("profile", ("CDF-HARSH", "CDF-MILD"))
def test_cdf9_real_domain_extrema_and_no_extension(profile: str) -> None:
    severity = 5 if profile == "CDF-HARSH" else 10
    for event_id in range(60):
        gt = math.sin(0.5 * math.pi * event_id / severity)
        multiplier = 0.5 + abs(gt)
        expected_maximum_x = min(1.0, 1.0 / multiplier)
        derivation = derive_cdf_reference(
            9,
            profile=profile,
            event_id=event_id,
        )

        assert math.isclose(
            derivation.extrema.maxima[0] - abs(gt),
            expected_maximum_x,
            abs_tol=1e-15,
        )
        assert (
            derivation.certificate["q_below_zero_policy"]
            == "CHARGED_TYPED_CDFDomainUndefinedError_NO_EXTENSION"
        )


@pytest.mark.parametrize("profile", ("CDF-HARSH", "CDF-MILD"))
def test_cdf14_finite_classification_uses_exact_event_arithmetic(
    profile: str,
) -> None:
    severity = 5 if profile == "CDF-HARSH" else 10
    for event_id in range(60):
        derivation = derive_cdf_reference(
            14,
            profile=profile,
            event_id=event_id,
        )
        if event_id % (2 * severity) == 0:
            assert derivation.finite_front is not None
            assert len(derivation.finite_front.points) == 21
        else:
            assert derivation.finite_front is None


def test_catalog_records_fail_closed_on_hash_drift() -> None:
    record = derive_cdf_reference(
        13,
        profile="CDF-HARSH",
        event_id=3,
        master_seed_u64=CDF_REFERENCE_SEEDS[0],
    ).canonical_record()
    assert reference_derivation_from_record(record).canonical_record() == record

    drifted = json.loads(json.dumps(record))
    drifted["extrema"]["maxima_hex"][0] = 0.0.hex()
    with pytest.raises(ReferenceArtifactError, match="SHA-256"):
        reference_derivation_from_record(drifted)


def test_materialized_catalog_matches_its_manifest() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    schema = json.loads(MANIFEST_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(manifest)
    artifact = manifest["catalog_artifact"]

    assert artifact["path"] == (
        "config/r8c_e1e2/reference_catalog/reference_artifacts.jsonl"
    )
    assert artifact["bytes"] == CATALOG.stat().st_size
    assert artifact["sha256"] == sha256(CATALOG.read_bytes()).hexdigest()
    derivations = load_reference_catalog(
        CATALOG,
        expected_sha256=artifact["sha256"],
        expected_lines=artifact["lines"],
    )
    assert len(derivations) == REFERENCE_CATALOG_EXPECTED_IDENTITIES
    assert manifest["identity_scope"]["actual_total"] == len(derivations)
    assert manifest["representation"]["arbitrary_dense_pf_samples_stored"] is False
    for binding in manifest["source_bindings"].values():
        if "path" not in binding or binding["path"].startswith("MLSGA/"):
            continue
        path = ROOT / binding["path"]
        assert path.stat().st_size == binding["bytes"]
        assert sha256(path.read_bytes()).hexdigest() == binding["sha256"]
