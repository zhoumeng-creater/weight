"""Public-development-only port of the frozen WGT-RR generator.

The source algorithm is bound to
``项目工作区/05_工具与验证/f23_hidden_instance_generator.py`` with
SHA-256 ``d553e237df7f18a8cfe9f02931e32dc185368b7504a3247a106f0571f9ad8dd2``.
This port intentionally exposes no hidden split, salt, or caller-supplied
master seed.
"""

from __future__ import annotations

import hashlib
import hmac
import math
from dataclasses import dataclass
from typing import Any


MASK64 = (1 << 64) - 1
GENERATOR_ID = "WGT-F23-RRGEN-01"
GENERATOR_VERSION = "1.0.0"
DOMAIN_SEED = b"WGT-F23-SEED-v1\x00"
TEMPLATES = ("RR-SMOOTH", "RR-SHOCK", "RR-REJECTION", "RR-INTERMITTENT")
PUBLIC_DEVELOPMENT_MASTER_SEED_HEX = (
    "7d717dd3458bd5c78201839faf6ef4abc7628e94e7672bbb79743298da835c2f"
)


class PublicRollingGeneratorError(ValueError):
    """The requested public rolling instance violates the frozen contract."""


def _u64(value: int) -> int:
    return value & MASK64


@dataclass
class _SplitMix64:
    state: int

    def next_u64(self) -> int:
        self.state = _u64(self.state + 0x9E3779B97F4A7C15)
        value = self.state
        value = _u64((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9)
        value = _u64((value ^ (value >> 27)) * 0x94D049BB133111EB)
        return _u64(value ^ (value >> 31))

    def uniform01(self) -> float:
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))

    def uniform(self, low: float, high: float) -> float:
        return low + (high - low) * self.uniform01()

    def integer(self, low: int, high_inclusive: int) -> int:
        width = high_inclusive - low + 1
        return low + int(self.uniform01() * width)


def _derive_seed(template: str, index: int) -> int:
    if template not in TEMPLATES:
        raise PublicRollingGeneratorError("unknown public rolling template")
    if type(index) is not int or not 0 <= index <= 7:
        raise PublicRollingGeneratorError(
            "public rolling index must be an integer in 0..7"
        )
    message = (
        DOMAIN_SEED
        + f"development|WGT-RR-CMOP|{template}|{index}".encode("ascii")
    )
    master_seed = bytes.fromhex(PUBLIC_DEVELOPMENT_MASTER_SEED_HEX)
    return int.from_bytes(
        hmac.new(master_seed, message, hashlib.sha256).digest()[:8],
        "big",
    )


def _common(rng: _SplitMix64) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "events": 20,
        "planning_horizon": 6,
        "state_dimension": 2,
        "action_dimension": 2,
        "a_diagonal": [
            rng.uniform(0.78, 0.94),
            rng.uniform(0.78, 0.94),
        ],
        "b_diagonal": [
            rng.uniform(0.18, 0.32),
            rng.uniform(0.18, 0.32),
        ],
        "b_rotation_radians": rng.uniform(-0.35, 0.35),
        "initial_state": [
            rng.uniform(-0.4, 0.4),
            rng.uniform(-0.4, 0.4),
        ],
        "reference_amplitude": [
            rng.uniform(0.25, 0.75),
            rng.uniform(0.25, 0.75),
        ],
        "reference_phase": [
            rng.uniform(0.0, 2 * math.pi),
            rng.uniform(0.0, 2 * math.pi),
        ],
        "reference_period_events": [
            rng.uniform(12.0, 24.0),
            rng.uniform(12.0, 24.0),
        ],
        "drift_amplitude": [
            rng.uniform(0.0, 0.05),
            rng.uniform(0.0, 0.05),
        ],
        "state_bound": [
            rng.uniform(1.20, 1.80),
            rng.uniform(1.20, 1.80),
        ],
        "rate_limit": rng.uniform(0.35, 0.65),
        "obstacle_center": [
            rng.uniform(-0.30, 0.30),
            rng.uniform(-0.30, 0.30),
        ],
        "obstacle_radius": rng.uniform(0.15, 0.35),
        "disturbance_bound": [
            rng.uniform(0.01, 0.08),
            rng.uniform(0.01, 0.08),
        ],
        "safety_margin_fraction": 0.05,
        "feedback_missing_events": [],
        "shock": None,
        "temporary_narrowing": None,
    }
    parameters["disturbance_sequence"] = [
        [
            rng.uniform(
                -parameters["disturbance_bound"][0],
                parameters["disturbance_bound"][0],
            ),
            rng.uniform(
                -parameters["disturbance_bound"][1],
                parameters["disturbance_bound"][1],
            ),
        ]
        for _ in range(parameters["events"])
    ]
    return parameters


def generate_public_instance(template: str, index: int) -> dict[str, Any]:
    """Generate one of the 32 frozen public development instances."""

    seed = _derive_seed(template, index)
    rng = _SplitMix64(seed)
    parameters = _common(rng)
    if template == "RR-SHOCK":
        parameters["shock"] = {
            "event": rng.integer(7, 13),
            "a_delta": [
                rng.uniform(-0.12, 0.12),
                rng.uniform(-0.12, 0.12),
            ],
            "bound_multiplier": rng.uniform(0.65, 0.85),
            "reference_phase_shift": rng.uniform(-math.pi, math.pi),
        }
    elif template == "RR-REJECTION":
        start = rng.integer(5, 10)
        parameters["safety_margin_fraction"] = rng.uniform(0.08, 0.15)
        parameters["temporary_narrowing"] = {
            "start_event": start,
            "duration_events": rng.integer(2, 4),
            "bound_multiplier": rng.uniform(0.45, 0.65),
        }
    elif template == "RR-INTERMITTENT":
        missing = [
            event for event in range(1, 20) if rng.uniform01() < 0.25
        ]
        if not missing:
            missing = [rng.integer(1, 19)]
        parameters["feedback_missing_events"] = missing

    return {
        "generator_id": GENERATOR_ID,
        "generator_version": GENERATOR_VERSION,
        "suite_id": "WGT-RR-CMOP",
        "split": "development",
        "template": template,
        "index": index,
        "derived_seed_u64": seed,
        "parameters": parameters,
    }
