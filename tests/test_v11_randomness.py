from __future__ import annotations

import numpy as np
import pytest

from evaluation.randomness import RandomStream, derive_rng


def test_rng_derivation_is_stable_and_replayable() -> None:
    first, first_manifest = derive_rng(
        20260724,
        stream=RandomStream.ALGORITHM,
        experiment_id="E0",
        unit_id="fixture-1",
        method_id="FULL",
        event_id=2,
        generation=3,
        target_index=4,
        substream="parameter",
    )
    second, second_manifest = derive_rng(
        20260724,
        stream=RandomStream.ALGORITHM,
        experiment_id="E0",
        unit_id="fixture-1",
        method_id="FULL",
        event_id=2,
        generation=3,
        target_index=4,
        substream="parameter",
    )

    np.testing.assert_array_equal(first.random(8), second.random(8))
    assert first_manifest == second_manifest
    assert first_manifest["derivation_domain"] == "WGT-V11-RNG-v1"
    assert len(first_manifest["sha256"]) == 64


def test_algorithm_streams_are_method_and_counter_isolated() -> None:
    base = dict(
        stream=RandomStream.ALGORITHM,
        experiment_id="E0",
        unit_id="fixture-1",
        event_id=0,
        generation=0,
        target_index=0,
        substream="operator",
    )
    full, _ = derive_rng(7, method_id="FULL", **base)
    shade, _ = derive_rng(7, method_id="SHADE_ONLY", **base)
    next_target, _ = derive_rng(7, method_id="FULL", **{**base, "target_index": 1})

    assert full.random() != shade.random()
    replay, _ = derive_rng(7, method_id="FULL", **base)
    assert replay.random() != next_target.random()


def test_algorithm_substreams_are_purpose_isolated_and_auditable() -> None:
    base = dict(
        stream=RandomStream.ALGORITHM,
        experiment_id="E0",
        unit_id="fixture-1",
        method_id="FULL",
        event_id=0,
        generation=0,
        target_index=0,
    )
    parameter, parameter_manifest = derive_rng(
        7,
        substream="parameter",
        **base,
    )
    operator, operator_manifest = derive_rng(
        7,
        substream="operator",
        **base,
    )

    assert parameter.random() != operator.random()
    assert parameter_manifest["tokens"]["substream"] == "parameter"
    assert operator_manifest["tokens"]["substream"] == "operator"


def test_initialization_stream_excludes_method_identity() -> None:
    shared, manifest = derive_rng(
        19,
        stream=RandomStream.INITIALIZATION,
        experiment_id="E0",
        unit_id="fixture-1",
        event_id=0,
        substream="initialization",
    )
    replay, replay_manifest = derive_rng(
        19,
        stream=RandomStream.INITIALIZATION,
        experiment_id="E0",
        unit_id="fixture-1",
        event_id=0,
        substream="initialization",
    )

    np.testing.assert_array_equal(shared.random(4), replay.random(4))
    assert manifest == replay_manifest
    assert "method_id" not in manifest["tokens"]
    with pytest.raises(ValueError, match="method_id"):
        derive_rng(
            19,
            stream=RandomStream.INITIALIZATION,
            experiment_id="E0",
            unit_id="fixture-1",
            method_id="FULL",
            event_id=0,
            substream="initialization",
        )


@pytest.mark.parametrize(
    "stream",
    [
        RandomStream.INSTANCE,
        RandomStream.OBSERVATION,
        RandomStream.EXECUTION,
    ],
)
def test_paired_environment_streams_exclude_method_identity(
    stream: RandomStream,
) -> None:
    shared, manifest = derive_rng(
        11,
        stream=stream,
        experiment_id="E0",
        unit_id="fixture-1",
        event_id=1,
    )
    replay, replay_manifest = derive_rng(
        11,
        stream=stream,
        experiment_id="E0",
        unit_id="fixture-1",
        event_id=1,
    )
    np.testing.assert_array_equal(shared.random(4), replay.random(4))
    assert manifest == replay_manifest
    assert "method_id" not in manifest["tokens"]

    with pytest.raises(ValueError, match="method_id"):
        derive_rng(
            11,
            stream=stream,
            experiment_id="E0",
            unit_id="fixture-1",
            method_id="FULL",
        )


def test_algorithm_stream_requires_method_identity_and_nonnegative_seed() -> None:
    with pytest.raises(ValueError, match="method_id"):
        derive_rng(
            1,
            stream=RandomStream.ALGORITHM,
            experiment_id="E0",
            unit_id="fixture-1",
        )
    with pytest.raises(ValueError, match="nonnegative"):
        derive_rng(
            -1,
            stream=RandomStream.ALGORITHM,
            experiment_id="E0",
            unit_id="fixture-1",
            method_id="FULL",
        )
    with pytest.raises(ValueError, match="substream"):
        derive_rng(
            1,
            stream=RandomStream.ALGORITHM,
            experiment_id="E0",
            unit_id="fixture-1",
            method_id="FULL",
        )
