from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np
import pytest

import dt_ramde_v11.engine as engine_module
from benchmark_adapters.r4_public import make_r4_cdf_adapter
from dt_ramde_v11.contracts import (
    AlgorithmConfig,
    ExecutionScope,
    R6ExecutionRequest,
)
from dt_ramde_v11.engine import DTRAMDE
from dt_ramde_v11.engine import (
    _serializable_vector,
    _vector_audit_material,
    _vector_hash,
)
from evaluation.randomness import (
    DERIVATION_DOMAIN,
    RandomStream,
    _algorithm_payload_segments,
    derive_rng,
)


def _reference_rng(
    master_seed: int,
    *,
    stream: RandomStream,
    experiment_id: str,
    unit_id: str,
    method_id: str | None = None,
    event_id: int | None = None,
    generation: int | None = None,
    target_index: int | None = None,
    substream: str | None = None,
) -> tuple[np.random.Generator, dict[str, Any]]:
    tokens: dict[str, Any] = {
        "stream": stream.value,
        "experiment_id": experiment_id,
        "unit_id": unit_id,
    }
    if method_id is not None:
        tokens["method_id"] = method_id
    if event_id is not None:
        tokens["event_id"] = int(event_id)
    if generation is not None:
        tokens["generation"] = int(generation)
    if target_index is not None:
        tokens["target_index"] = int(target_index)
    if substream is not None:
        tokens["substream"] = substream
    payload = json.dumps(
        {
            "derivation_domain": DERIVATION_DOMAIN,
            "master_seed": int(master_seed),
            "tokens": tokens,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    entropy = np.frombuffer(digest, dtype="<u4").tolist()
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    return rng, {
        "derivation_domain": DERIVATION_DOMAIN,
        "tokens": tokens,
        "sha256": digest.hex(),
    }


@pytest.mark.parametrize(
    "arguments",
    [
        {
            "master_seed": 0,
            "stream": RandomStream.INSTANCE,
            "experiment_id": "实验/α",
            "unit_id": 'unit-"quoted"',
        },
        {
            "master_seed": 2**63 + 17,
            "stream": RandomStream.INITIALIZATION,
            "experiment_id": "E0\nline",
            "unit_id": "fixture-1",
            "event_id": 0,
            "substream": "initialization",
        },
        {
            "master_seed": 20260726,
            "stream": RandomStream.ALGORITHM,
            "experiment_id": "实验 E0\nα",
            "unit_id": 'fixture-"1"',
            "method_id": "DT-RAMDE_TS2_FULL/β",
            "event_id": 59,
            "generation": 999,
            "target_index": 99,
            "substream": "parameter/参数",
        },
        {
            "master_seed": 20260726,
            "stream": RandomStream.ALGORITHM,
            "experiment_id": "E0",
            "unit_id": "fixture-1",
            "method_id": "DT-RAMDE_TS2_FULL",
            "event_id": 59,
            "generation": 999,
            "target_index": 100,
            "substream": "archive",
        },
    ],
)
def test_cached_rng_payload_matches_reference_byte_for_byte(
    arguments: dict[str, Any],
) -> None:
    actual_rng, actual_manifest = derive_rng(**arguments)
    expected_rng, expected_manifest = _reference_rng(**arguments)

    assert actual_manifest == expected_manifest
    np.testing.assert_array_equal(
        actual_rng.bit_generator.random_raw(32),
        expected_rng.bit_generator.random_raw(32),
    )


@pytest.mark.parametrize("case_seed", range(50))
def test_algorithm_fast_path_matches_reference_for_random_counters(
    case_seed: int,
) -> None:
    rng = np.random.default_rng(case_seed)
    arguments = {
        "master_seed": int(rng.integers(0, 2**63, dtype=np.uint64)),
        "stream": RandomStream.ALGORITHM,
        "experiment_id": f"E-{case_seed}-实验",
        "unit_id": f"unit/{int(rng.integers(0, 1000))}",
        "method_id": f"method-{int(rng.integers(0, 20))}",
        "event_id": (
            None if case_seed % 7 == 0 else int(rng.integers(0, 100))
        ),
        "generation": int(rng.integers(0, 100_000)),
        "target_index": int(rng.integers(0, 10_000)),
        "substream": ("parameter", "operator", "j_rand", "archive")[
            case_seed % 4
        ],
    }
    actual_rng, actual_manifest = derive_rng(**arguments)
    expected_rng, expected_manifest = _reference_rng(**arguments)

    assert actual_manifest == expected_manifest
    np.testing.assert_array_equal(
        actual_rng.bit_generator.random_raw(32),
        expected_rng.bit_generator.random_raw(32),
    )


def test_algorithm_payload_segments_are_reused_with_exact_dynamic_hashes() -> None:
    _algorithm_payload_segments.cache_clear()
    common = {
        "master_seed": 17,
        "stream": RandomStream.ALGORITHM,
        "experiment_id": "E0",
        "unit_id": "fixture-1",
        "method_id": "FULL",
        "event_id": 2,
        "substream": "operator",
    }
    for generation, target_index in ((0, 0), (0, 1), (1, 0)):
        actual_rng, actual_manifest = derive_rng(
            **common,
            generation=generation,
            target_index=target_index,
        )
        expected_rng, expected_manifest = _reference_rng(
            **common,
            generation=generation,
            target_index=target_index,
        )
        assert actual_manifest == expected_manifest
        np.testing.assert_array_equal(
            actual_rng.bit_generator.random_raw(8),
            expected_rng.bit_generator.random_raw(8),
        )

    cache = _algorithm_payload_segments.cache_info()
    assert cache.misses == 1
    assert cache.hits == 2


def _run_public_engine_fixture(*, batched: bool = True) -> dict[str, Any]:
    problem = make_r4_cdf_adapter(
        1,
        profile="CDF-HARSH",
        environment_seed=20260726,
    )
    if not batched:
        base_problem = problem

        class ScalarOnlyProblem:
            evaluate_batch = None

            def __getattr__(self, name: str) -> Any:
                return getattr(base_problem, name)

        problem = ScalarOnlyProblem()

    class Selector:
        selector_id = "RNG-CACHE-TEST-SELECTOR"
        selector_version = "1.0.0"

        def identity(self) -> dict[str, str]:
            return {
                "selector_id": self.selector_id,
                "selector_version": self.selector_version,
            }

        def select(self, archive: Any) -> str | None:
            return min(
                (candidate.candidate_id for candidate in archive),
                default=None,
            )

    selector = Selector()
    config = AlgorithmConfig(
        variant="FULL",
        population_size=10,
        cfe_per_event=30,
        algorithm_seed=20260726,
        max_events=2,
        timing_mode="TS2_fixed_periodic_replanning",
        method_label="DT-RAMDE_TS2_FULL",
        adapter_id=problem.adapter_id,
        adapter_version=problem.adapter_version,
        selector_id=selector.selector_id,
        selector_version=selector.selector_version,
        atomic_steps_per_evaluation=1,
        event_time_limit_seconds=3600.0,
        configuration_evidence_id="RNG_CACHE_END_TO_END_PILOT",
        execution_request=R6ExecutionRequest(
            scope=ExecutionScope.ENGINEERING_PILOT
        ),
    )
    return DTRAMDE(config).run_sequence(problem, selector=selector).to_dict()


def test_engine_rng_manifests_and_outputs_match_reference_derivation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual = _run_public_engine_fixture()
    monkeypatch.setattr(engine_module, "derive_rng", _reference_rng)
    expected = _run_public_engine_fixture()

    assert actual == expected


def _reference_trial_vector(
    self: DTRAMDE,
    population: list[Any],
    inferior_archive: list[Any],
    ranked_population: list[Any],
    *,
    problem: Any,
    event_id: int,
    generation: int,
    target_index: int,
    f_value: float,
    cr_value: float,
    selection_cache: Any = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    size = len(population)
    p_count = max(
        1,
        math.ceil(min(1.0, max(0.10, 2.0 / size)) * size),
    )
    rng, operator_manifest = engine_module.derive_rng(
        self.config.algorithm_seed,
        stream=RandomStream.ALGORITHM,
        experiment_id=self.config.configuration_evidence_id,
        unit_id=problem.adapter_id,
        method_id=self.method_id,
        event_id=event_id,
        generation=generation,
        target_index=target_index,
        substream="operator",
    )
    pbest = ranked_population[int(rng.integers(0, p_count))]
    r1_indices = [index for index in range(size) if index != target_index]
    r1 = population[int(rng.choice(r1_indices))]
    target = population[target_index]
    excluded = {target.candidate_id, r1.candidate_id}
    r2_pool = [
        candidate
        for candidate in population + inferior_archive
        if candidate.candidate_id not in excluded
    ]
    if not r2_pool:
        raise engine_module.StateIntegrityError("no legal r2 candidate")
    r2 = r2_pool[int(rng.integers(0, len(r2_pool)))]
    mutant = (
        target.vector
        + f_value * (pbest.vector - target.vector)
        + f_value * (r1.vector - r2.vector)
    )
    j_rand_rng, j_rand_manifest = engine_module.derive_rng(
        self.config.algorithm_seed,
        stream=RandomStream.ALGORITHM,
        experiment_id=self.config.configuration_evidence_id,
        unit_id=problem.adapter_id,
        method_id=self.method_id,
        event_id=event_id,
        generation=generation,
        target_index=target_index,
        substream="j_rand",
    )
    j_rand = int(j_rand_rng.integers(0, len(mutant)))
    mask = rng.random(len(mutant)) < cr_value
    mask[j_rand] = True
    return np.where(mask, mutant, target.vector), {
        "pbest": pbest,
        "r1": r1,
        "r2": r2,
        "j_rand": j_rand,
        "rng": {
            "operator": operator_manifest,
            "j_rand": j_rand_manifest,
        },
    }


def _canonical_result_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@pytest.mark.parametrize("batched", [True, False])
def test_generation_selection_cache_is_byte_exact_to_old_reference(
    monkeypatch: pytest.MonkeyPatch,
    batched: bool,
) -> None:
    actual = _run_public_engine_fixture(batched=batched)
    assert all(
        {
            audit["node_id"].split(":")[1]
            for audit in event["trial_audit"]
        }
        == {"g000000", "g000001"}
        for event in actual["events"]
    )

    monkeypatch.setattr(DTRAMDE, "_trial_vector", _reference_trial_vector)
    expected = _run_public_engine_fixture(batched=batched)

    assert _canonical_result_bytes(actual) == _canonical_result_bytes(expected)


@pytest.mark.parametrize(
    "vector",
    [
        np.asarray([0.0, -0.0, 1.25, -3.5]),
        np.asarray([np.inf, -np.inf, np.nan]),
        [1, 2, 3],
    ],
)
def test_vector_audit_material_reuses_serialization_without_hash_drift(
    vector: Any,
) -> None:
    serialized, digest = _vector_audit_material(vector)

    assert serialized == _serializable_vector(vector)
    assert digest == _vector_hash(vector)
