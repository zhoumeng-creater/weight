from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import numpy as np
import pytest

from analysis.r9_inference import (
    BootstrapCluster,
    ClusterEffect,
    R9InferenceError,
    TaskEndpoint,
    _missing_pair_bounds,
    holm_adjust,
    paired_stratified_cluster_bootstrap,
    paired_sign_flip,
    registered_hypotheses,
)


ROOT = Path(__file__).resolve().parents[1]


def _bootstrap_clusters(
    values: list[float],
) -> tuple[BootstrapCluster, ...]:
    return tuple(
        BootstrapCluster(
            cluster_id=f"cluster-{index}",
            top_stratum_id="ALL",
            fixed_stratum_values=((value,),),
        )
        for index, value in enumerate(values)
    )


def test_registered_hypothesis_order_and_family_sizes() -> None:
    hypotheses = registered_hypotheses()

    assert len(hypotheses) == 30
    assert [row.hypothesis_index for row in hypotheses] == list(
        range(1, 31)
    )
    assert sum(
        row.family_id == "E1_PRIMARY_ANYTIME" for row in hypotheses
    ) == 15
    assert sum(
        row.family_id == "E2_DYNAMIC_TRANSFER" for row in hypotheses
    ) == 7
    assert sum(
        row.family_id == "E2_ROLLING_TRANSFER" for row in hypotheses
    ) == 8
    assert hypotheses[0].analysis_workload_id == "E1_STATIC"
    assert hypotheses[14].analysis_workload_id == "E1_ROLLING"
    assert hypotheses[15].comparator_method_id == (
        "NO_CROSS_EVENT_MEMORY"
    )
    assert hypotheses[-1].comparator_method_id == "SHADE_ONLY"


def test_zero_effect_known_answer_for_bootstrap_and_sign_flip() -> None:
    effects = [0.0, 0.0, 0.0, 0.0]
    bootstrap_rng = np.random.Generator(np.random.PCG64(11))
    sign_rng = np.random.Generator(np.random.PCG64(12))

    assert paired_stratified_cluster_bootstrap(
        _bootstrap_clusters(effects),
        replicates=500,
        rng=bootstrap_rng,
    ) == (0.0, 0.0)
    extreme, p_value = paired_sign_flip(
        effects,
        replicates=1_000,
        rng=sign_rng,
    )
    assert extreme == 1_000
    assert p_value == 1.0


def test_rng_streams_are_deterministic_and_sequential() -> None:
    first = np.random.Generator(np.random.PCG64(91))
    second = np.random.Generator(np.random.PCG64(91))
    effects = [-0.2, 0.1, 0.4]

    clusters = _bootstrap_clusters(effects)
    first_ci = paired_stratified_cluster_bootstrap(
        clusters,
        replicates=250,
        rng=first,
    )
    second_ci = paired_stratified_cluster_bootstrap(
        clusters,
        replicates=250,
        rng=second,
    )
    assert first_ci == second_ci
    assert paired_stratified_cluster_bootstrap(
        clusters,
        replicates=250,
        rng=first,
    ) == paired_stratified_cluster_bootstrap(
        clusters,
        replicates=250,
        rng=second,
    )


def test_bootstrap_resamples_within_fixed_top_strata() -> None:
    clusters = (
        BootstrapCluster("A-1", "A", ((0.0,),)),
        BootstrapCluster("A-2", "A", ((0.0,),)),
        BootstrapCluster("B-1", "B", ((10.0,),)),
        BootstrapCluster("B-2", "B", ((10.0,),)),
    )
    rng = np.random.Generator(np.random.PCG64(8))

    assert paired_stratified_cluster_bootstrap(
        clusters,
        replicates=500,
        rng=rng,
    ) == (5.0, 5.0)


def test_bootstrap_resamples_paired_seeds_inside_selected_cluster() -> None:
    clusters = (
        BootstrapCluster(
            "only",
            "ALL",
            ((0.0, 2.0),),
        ),
    )
    rng = np.random.Generator(np.random.PCG64(9))

    lower, upper = paired_stratified_cluster_bootstrap(
        clusters,
        replicates=2_000,
        rng=rng,
    )
    assert lower == 0.0
    assert upper == 2.0


def test_holm_known_answer_and_step_down_decisions() -> None:
    adjusted, rejected = holm_adjust([0.01, 0.04, 0.03])

    assert adjusted == pytest.approx((0.03, 0.06, 0.06))
    assert rejected == (True, False, False)


@pytest.mark.parametrize(
    ("proposed", "comparator", "expected"),
    [
        (0.8, 0.3, (0.5, 0.5)),
        (None, None, (-1.0, 1.0)),
        (None, 0.3, (-0.3, 0.7)),
        (0.8, None, (-0.2, 0.8)),
    ],
)
def test_missing_pair_endpoint_bounds(
    proposed: float | None,
    comparator: float | None,
    expected: tuple[float, float],
) -> None:
    assert _missing_pair_bounds(proposed, comparator) == pytest.approx(
        expected
    )


def test_statistical_functions_fail_closed_on_empty_input() -> None:
    rng = np.random.Generator(np.random.PCG64(1))

    with pytest.raises(R9InferenceError, match="no bootstrap clusters"):
        paired_stratified_cluster_bootstrap(
            [],
            replicates=10,
            rng=rng,
        )
    with pytest.raises(R9InferenceError, match="no cluster effects"):
        paired_sign_flip([], replicates=10, rng=rng)
    with pytest.raises(R9InferenceError, match="no p-values"):
        holm_adjust([])


def test_frozen_hypothesis_is_immutable() -> None:
    hypothesis = registered_hypotheses()[0]

    modified = replace(hypothesis, practical_threshold=0.5)
    assert hypothesis.practical_threshold == 0.02
    assert modified.practical_threshold == 0.5


def test_result_data_models_are_frozen() -> None:
    task = TaskEndpoint(
        task_id="task",
        workload_id="E1_STATIC",
        unit_id="LIRCMOP1",
        method_id="F22_MG_STATIC",
        replicate_index=0,
        task_status="COMPLETE",
        endpoint_status="INCLUDED",
        anytime_nhv_auc=0.2,
        transfer_early_auc=None,
    )
    cluster = ClusterEffect(
        cluster_id="LIRCMOP1",
        valid_nested_pairs=10,
        expected_nested_pairs=10,
        effect=0.1,
        lower_bound=0.1,
        upper_bound=0.1,
    )

    with pytest.raises(Exception):
        task.task_id = "other"  # type: ignore[misc]
    with pytest.raises(Exception):
        cluster.effect = 0.2  # type: ignore[misc]


def test_versioned_contract_schema_and_hypothesis_order() -> None:
    contract = json.loads(
        (
            ROOT
            / "config"
            / "r9"
            / "r9_inference_implementation_v1_0_1.json"
        ).read_text(encoding="utf-8")
    )
    schema = json.loads(
        (
            ROOT
            / "config"
            / "r9"
            / "r9_inference_implementation_v1_0_1.schema.json"
        ).read_text(encoding="utf-8")
    )

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(contract)
    assert contract["hypothesis_order"] == [
        row.hypothesis_id for row in registered_hypotheses()
    ]
