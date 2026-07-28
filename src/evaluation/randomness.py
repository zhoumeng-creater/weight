"""Hierarchical, paired, domain-separated v1.1 random streams.

Rewrite provenance:
    LEGACY_WEIGHT/randomness.py at commit
    a00adadc6bfaa753e81abd91723992ed868ead0c
    SHA-256 a77a1e310d1b43729b940fa71c49f0f8b0f394464ff0b36abdf8cd2a51b33293

The implementation also preserves the formal SHA-256/SeedSequence derivation
idea while making shared environment streams and method streams explicit.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from functools import lru_cache
from typing import Any

import numpy as np


DERIVATION_DOMAIN = "WGT-V11-RNG-v1"


class RandomStream(str, Enum):
    INSTANCE = "instance"
    INITIALIZATION = "initialization"
    OBSERVATION = "observation"
    EXECUTION = "execution"
    ALGORITHM = "algorithm"
    BOOTSTRAP = "bootstrap"
    PERMUTATION = "permutation"


_PAIRED_ENVIRONMENT_STREAMS = {
    RandomStream.INSTANCE,
    RandomStream.INITIALIZATION,
    RandomStream.OBSERVATION,
    RandomStream.EXECUTION,
}
_METHOD_STREAMS = {RandomStream.ALGORITHM}


def _encoded_json_scalar(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@lru_cache(maxsize=2048)
def _algorithm_payload_segments(
    master_seed: int,
    experiment_id: str,
    unit_id: str,
    method_id: str,
    event_id: int | None,
    substream: str,
) -> tuple[bytes, bytes, bytes]:
    """Cache canonical bytes around generation and target placeholders."""

    event_fragment = (
        b""
        if event_id is None
        else b'"event_id":' + str(event_id).encode("ascii") + b","
    )
    before_generation = b"".join(
        (
            b'{"derivation_domain":',
            _encoded_json_scalar(DERIVATION_DOMAIN),
            b',"master_seed":',
            str(master_seed).encode("ascii"),
            b',"tokens":{',
            event_fragment,
            b'"experiment_id":',
            _encoded_json_scalar(experiment_id),
            b',"generation":',
        )
    )
    before_target = b"".join(
        (
            b',"method_id":',
            _encoded_json_scalar(method_id),
            b',"stream":"algorithm","substream":',
            _encoded_json_scalar(substream),
            b',"target_index":',
        )
    )
    suffix = b',"unit_id":' + _encoded_json_scalar(unit_id) + b"}}"
    return before_generation, before_target, suffix


def derive_rng(
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
    include_manifest: bool = True,
) -> tuple[np.random.Generator, dict[str, Any]]:
    """Return a replayable generator and its non-secret derivation manifest."""

    if master_seed < 0:
        raise ValueError("master_seed must be nonnegative")
    if not experiment_id or not unit_id:
        raise ValueError("experiment_id and unit_id must be nonempty")
    if stream in _PAIRED_ENVIRONMENT_STREAMS and method_id is not None:
        raise ValueError("paired environment stream must not include method_id")
    if stream in _METHOD_STREAMS and not method_id:
        raise ValueError("algorithm stream requires method_id")
    if stream in _METHOD_STREAMS and not substream:
        raise ValueError("algorithm stream requires substream")
    if substream is not None and not substream.strip():
        raise ValueError("substream must be nonempty when provided")
    for name, value in {
        "event_id": event_id,
        "generation": generation,
        "target_index": target_index,
    }.items():
        if value is not None and value < 0:
            raise ValueError(f"{name} must be nonnegative when provided")

    tokens: dict[str, Any] | None = None
    if include_manifest or not (
        stream is RandomStream.ALGORITHM
        and method_id is not None
        and generation is not None
        and target_index is not None
        and substream is not None
    ):
        tokens = {
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

    if (
        stream is RandomStream.ALGORITHM
        and method_id is not None
        and generation is not None
        and target_index is not None
        and substream is not None
    ):
        segments = _algorithm_payload_segments(
            int(master_seed),
            experiment_id,
            unit_id,
            method_id,
            None if event_id is None else int(event_id),
            substream,
        )
        payload = b"".join(
            (
                segments[0],
                str(int(generation)).encode("ascii"),
                segments[1],
                str(int(target_index)).encode("ascii"),
                segments[2],
            )
        )
    else:
        assert tokens is not None
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
    entropy = np.frombuffer(digest, dtype="<u4")
    rng = np.random.default_rng(entropy)
    if not include_manifest:
        return rng, {}
    assert tokens is not None
    manifest = {
        "derivation_domain": DERIVATION_DOMAIN,
        "tokens": tokens,
        "sha256": digest.hex(),
    }
    return rng, manifest
