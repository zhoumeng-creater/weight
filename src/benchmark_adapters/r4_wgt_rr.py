"""R4 public-development binding for the WGT-RR rolling benchmark.

This is a current-interface rewrite of the frozen public F23 equations.  It
does not import the historical runtime and rejects every split except
``development``.  The bundled RR-SMOOTH/index-0 known answer is public and is
used only for bridge correctness; no hidden instance is read or generated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from math import cos, pi, sin
from pathlib import Path
from typing import Any

import numpy as np

from dt_ramde_v11.core import Candidate
from evaluation.contracts import EvaluationResult
from evaluation.evaluator import SharedEvaluator
from evaluation.firewall import (
    PROHIBITED_FIELDS,
    InformationBoundaryError,
    InformationField,
    InformationSnapshot,
    freeze_information,
)
from evaluation.ledger import EvaluationLedger


WGT_RR_GENERATOR_ID = "WGT-F23-RRGEN-01"
WGT_RR_GENERATOR_VERSION = "1.0.0"
WGT_RR_GENERATOR_SHA256 = (
    "d553e237df7f18a8cfe9f02931e32dc185368b7504a3247a106f0571f9ad8dd2"
)
WGT_RR_PUBLIC_KNOWN_COMMITMENT = (
    "25c3aea2bb3e08e1435f1b68e61f2ad98a3085dc38c7f21af6a7a5eccfa077fa"
)


class WGTRRBindingError(ValueError):
    """A public WGT-RR instance or call violates the R4 binding."""


def _find_prohibited_key(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key)
            if name in PROHIBITED_FIELDS:
                return name
            nested = _find_prohibited_key(item)
            if nested is not None:
                return nested
    elif isinstance(value, list | tuple):
        for item in value:
            nested = _find_prohibited_key(item)
            if nested is not None:
                return nested
    return None


def load_public_wgt_rr_known_answer() -> Mapping[str, Any]:
    """Load the bundled public RR-SMOOTH/index-0 bridge instance."""

    path = Path(__file__).with_name("data") / "wgt_rr_rr_smooth_00.json"
    return json.loads(path.read_text(encoding="utf-8"))


class WGTRRPublicAdapter:
    """Public-development WGT-RR adapter on the shared v1.1 ledger."""

    adapter_id = "BIND-ROLLING-01/R4-PUBLIC-EXECUTABLE"
    adapter_version = "1.1.0-r4-binding"
    decision_dimension = 12
    atomic_steps_per_evaluation = 6
    lower_bounds = (-1.0,) * 12
    upper_bounds = (1.0,) * 12
    objective_names = (
        "mean_normalized_state_tracking_squared_error",
        "mean_normalized_action_effort",
        "mean_normalized_action_rate_squared_error",
    )
    constraint_names = tuple(
        f"horizon_{horizon}_{name}"
        for horizon in range(1, 7)
        for name in (
            "state_1_bound",
            "state_2_bound",
            "action_1_rate",
            "action_2_rate",
            "obstacle_exclusion",
        )
    )
    constraint_scales = (1.0,) * 30

    def __init__(self, instance: Mapping[str, Any]) -> None:
        self._validate_instance(instance)
        self._instance = json.loads(
            json.dumps(instance, sort_keys=True, separators=(",", ":"))
        )
        self._parameters = self._instance["parameters"]
        self._state = np.asarray(
            self._parameters["initial_state"], dtype=float
        )
        self._previous_action = np.zeros(2, dtype=float)
        self._released_history: list[dict[str, Any]] = []
        self._information: InformationSnapshot | None = None
        self._frozen_state: np.ndarray | None = None
        self._evaluator = SharedEvaluator(
            objective_names=self.objective_names,
            constraint_names=self.constraint_names,
            evaluate_joint=self._evaluate_joint,
            evaluate_joint_batch=self._evaluate_joint_batch,
        )

    @staticmethod
    def _validate_instance(instance: Mapping[str, Any]) -> None:
        if instance.get("suite_id") != "WGT-RR-CMOP":
            raise WGTRRBindingError("adapter accepts WGT-RR-CMOP only")
        if instance.get("split") != "development":
            raise WGTRRBindingError(
                "R4 public adapter rejects every non-development split"
            )
        if instance.get("generator_id") != WGT_RR_GENERATOR_ID:
            raise WGTRRBindingError("rolling generator identity differs")
        if instance.get("generator_version") != WGT_RR_GENERATOR_VERSION:
            raise WGTRRBindingError("rolling generator version differs")
        if instance.get("seed_commitment_sha256") != (
            WGT_RR_PUBLIC_KNOWN_COMMITMENT
        ):
            raise WGTRRBindingError(
                "public instance commitment differs from the known answer"
            )
        parameters = instance.get("parameters")
        if not isinstance(parameters, Mapping):
            raise WGTRRBindingError("rolling parameters must be a mapping")
        if (
            parameters.get("events") != 20
            or parameters.get("planning_horizon") != 6
            or len(parameters.get("disturbance_sequence", ())) != 20
        ):
            raise WGTRRBindingError(
                "rolling instance shape differs from the public contract"
            )

    @classmethod
    def from_known_answer(cls) -> WGTRRPublicAdapter:
        return cls(load_public_wgt_rr_known_answer())

    def identity(self) -> Mapping[str, Any]:
        return {
            "suite_id": "WGT-RR-CMOP",
            "split": "public_development_bridge",
            "template": self._instance["template"],
            "index": self._instance["index"],
            "derived_seed_u64": self._instance["derived_seed_u64"],
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "generator_id": WGT_RR_GENERATOR_ID,
            "generator_version": WGT_RR_GENERATOR_VERSION,
            "generator_sha256": WGT_RR_GENERATOR_SHA256,
            "public_commitment_sha256": WGT_RR_PUBLIC_KNOWN_COMMITMENT,
            "registered_benchmark_evaluator": True,
            "registered_effect_instance": False,
            "formal_effect_execution_allowed": False,
            "execution_authority": "R4_BINDING_ONLY_NO_EFFECT",
        }

    def _rotation_b(self) -> np.ndarray:
        theta = float(self._parameters["b_rotation_radians"])
        rotation = np.asarray(
            [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]],
            dtype=float,
        )
        return rotation @ np.diag(
            np.asarray(self._parameters["b_diagonal"], dtype=float)
        )

    def _released_parameters(
        self, event_id: int
    ) -> tuple[np.ndarray, np.ndarray, float]:
        a = np.asarray(self._parameters["a_diagonal"], dtype=float).copy()
        bounds = np.asarray(
            self._parameters["state_bound"], dtype=float
        ).copy()
        phase_shift = 0.0
        shock = self._parameters.get("shock")
        if shock is not None and event_id >= int(shock["event"]):
            a += np.asarray(shock["a_delta"], dtype=float)
            bounds *= float(shock["bound_multiplier"])
            phase_shift = float(shock["reference_phase_shift"])
        narrowing = self._parameters.get("temporary_narrowing")
        if narrowing is not None:
            start = int(narrowing["start_event"])
            duration = int(narrowing["duration_events"])
            if start <= event_id < start + duration:
                bounds *= float(narrowing["bound_multiplier"])
        return a, bounds, phase_shift

    def _reference(self, time_index: int, phase_shift: float) -> np.ndarray:
        amplitude = np.asarray(
            self._parameters["reference_amplitude"], dtype=float
        )
        period = np.asarray(
            self._parameters["reference_period_events"], dtype=float
        )
        phase = (
            np.asarray(self._parameters["reference_phase"], dtype=float)
            + phase_shift
        )
        return amplitude * np.sin(
            2.0 * pi * time_index / period + phase
        )

    def _nominal_drift(self, time_index: int) -> np.ndarray:
        amplitude = np.asarray(
            self._parameters["drift_amplitude"], dtype=float
        )
        phase = np.asarray(
            self._parameters["reference_phase"], dtype=float
        )
        return amplitude * np.sin(
            2.0 * pi * time_index / 19.0 + phase / 2.0
        )

    def freeze_information(
        self, event_id: int, feedback: Mapping[str, Any] | None
    ) -> InformationSnapshot:
        if not 0 <= event_id < 20:
            raise WGTRRBindingError("rolling event must be in 0..19")
        prohibited = _find_prohibited_key(feedback)
        if prohibited is not None:
            raise InformationBoundaryError(
                f"prohibited information field: {prohibited}"
            )
        if event_id == 0 and feedback is not None:
            raise WGTRRBindingError(
                "initial rolling event cannot receive prior feedback"
            )
        if feedback is not None:
            released_at = feedback.get("released_at")
            if type(released_at) is not int or released_at != event_id:
                raise InformationBoundaryError(
                    "rolling feedback must be released at the current event"
                )
        a, state_bound, phase_shift = self._released_parameters(event_id)
        fields = {
            "current_observed_state": InformationField(
                available_at=event_id,
                value=self._state.tolist(),
            ),
            "released_history_through_k": InformationField(
                available_at=event_id,
                value=list(self._released_history),
            ),
            "released_a_diagonal": InformationField(
                available_at=event_id,
                value=a.tolist(),
            ),
            "released_state_bound": InformationField(
                available_at=event_id,
                value=state_bound.tolist(),
            ),
            "released_reference_phase_shift": InformationField(
                available_at=event_id,
                value=phase_shift,
            ),
            "previous_action": InformationField(
                available_at=event_id,
                value=self._previous_action.tolist(),
            ),
            "frozen_interface_versions_and_hashes": InformationField(
                available_at=0,
                value={
                    "evaluation": self.adapter_version,
                    "generator": (
                        f"{WGT_RR_GENERATOR_ID}/{WGT_RR_GENERATOR_VERSION}"
                    ),
                    "generator_sha256": WGT_RR_GENERATOR_SHA256,
                    "binding": "BIND-ROLLING-01",
                    "execution_authority": "R4_BINDING_ONLY_NO_EFFECT",
                },
            ),
        }
        if feedback is not None:
            fields["prior_execution_feedback"] = InformationField(
                available_at=event_id,
                value=dict(feedback),
            )
        self._information = freeze_information(
            decision_time=event_id,
            fields=fields,
        )
        self._frozen_state = self._state.copy()
        return self._information

    def _evaluate_joint(
        self,
        vector: Sequence[float],
        information: InformationSnapshot,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if information is not self._information:
            raise WGTRRBindingError(
                "rolling evaluator received an unbound snapshot"
            )
        values = np.asarray(vector, dtype=float)
        if (
            values.shape != (12,)
            or not np.all(np.isfinite(values))
            or np.any(values < -1.0)
            or np.any(values > 1.0)
        ):
            raise FloatingPointError(
                "rolling candidate is nonfinite, out of bounds, or wrong shape"
            )
        event_id = information.decision_time
        actions = values.reshape(6, 2)
        a, state_bound, phase_shift = self._released_parameters(event_id)
        matrix_b = self._rotation_b()
        state = self._frozen_state.copy()
        prior_action = self._previous_action.copy()
        tracking: list[float] = []
        effort: list[float] = []
        rate: list[float] = []
        constraints: list[float] = []
        rate_limit = float(self._parameters["rate_limit"])
        center = np.asarray(
            self._parameters["obstacle_center"], dtype=float
        )
        radius = float(self._parameters["obstacle_radius"])
        for horizon, action in enumerate(actions):
            state = (
                a * state
                + matrix_b @ action
                + self._nominal_drift(event_id + horizon)
            )
            reference = self._reference(
                event_id + horizon + 1, phase_shift
            )
            delta = action - prior_action
            tracking.append(
                float(np.mean(((state - reference) / state_bound) ** 2))
            )
            effort.append(float(np.mean(action**2)))
            rate.append(float(np.mean((delta / rate_limit) ** 2)))
            constraints.extend(
                (np.abs(state) / state_bound - 1.0).tolist()
            )
            constraints.extend(
                (np.abs(delta) / rate_limit - 1.0).tolist()
            )
            constraints.append(
                float(
                    1.0
                    - np.sum((state - center) ** 2) / radius**2
                )
            )
            prior_action = action
        return (
            (
                float(np.mean(tracking)),
                float(np.mean(effort)),
                float(np.mean(rate)),
            ),
            tuple(constraints),
        )

    def _evaluate_joint_batch(
        self,
        vectors: Sequence[Sequence[float]],
        information: InformationSnapshot,
    ) -> tuple[
        tuple[tuple[float, ...], tuple[float, ...]],
        ...,
    ]:
        """Evaluate an ordered batch with the exact scalar numerical kernel.

        NumPy reductions and matrix products may differ by one or more ULPs
        between a two-dimensional vectorized expression and the corresponding
        one-candidate expression, with the difference depending on the host
        SIMD/BLAS implementation.  The formal contract requires batch and
        scalar paths to produce identical scientific bytes, so the batch
        interface retains atomic ledger charging while reusing the frozen
        scalar kernel for each candidate in order.
        """

        if information is not self._information:
            raise WGTRRBindingError(
                "rolling evaluator received an unbound snapshot"
            )
        values = np.asarray(vectors, dtype=float)
        if (
            values.ndim != 2
            or values.shape[1:] != (12,)
            or not np.all(np.isfinite(values))
            or np.any(values < -1.0)
            or np.any(values > 1.0)
        ):
            raise FloatingPointError(
                "rolling candidate batch is nonfinite, out of bounds, "
                "or wrong shape"
            )
        return tuple(
            self._evaluate_joint(vector, information)
            for vector in values
        )

    def evaluate(
        self,
        vector: Sequence[float],
        event_id: int,
        ledger: EvaluationLedger,
        candidate_id: str,
    ) -> EvaluationResult:
        information = self._information
        if information is None or information.decision_time != event_id:
            raise WGTRRBindingError(
                "freeze_information must bind the rolling event"
            )
        if self._frozen_state is None or not np.array_equal(
            self._state, self._frozen_state
        ):
            raise WGTRRBindingError(
                "outer rolling state changed during inner evaluation"
            )
        return self._evaluator.evaluate(
            vector=vector,
            event_id=event_id,
            candidate_id=candidate_id,
            information=information,
            ledger=ledger,
            atomic_steps=self.atomic_steps_per_evaluation,
            origin="wgt_rr_public_joint_evaluator",
        )

    def evaluate_batch(
        self,
        vectors: Sequence[Sequence[float]],
        event_id: int,
        ledger: EvaluationLedger,
        candidate_ids: Sequence[str],
    ) -> tuple[EvaluationResult, ...]:
        information = self._information
        if information is None or information.decision_time != event_id:
            raise WGTRRBindingError(
                "freeze_information must bind the rolling event"
            )
        if self._frozen_state is None or not np.array_equal(
            self._state, self._frozen_state
        ):
            raise WGTRRBindingError(
                "outer rolling state changed during inner evaluation"
            )
        return self._evaluator.evaluate_batch(
            vectors=vectors,
            event_id=event_id,
            candidate_ids=candidate_ids,
            information=information,
            ledger=ledger,
            atomic_steps=self.atomic_steps_per_evaluation,
            origin="wgt_rr_public_joint_evaluator",
        )

    def safety_filter(
        self, result: EvaluationResult, event_id: int
    ) -> bool:
        del event_id
        margin = float(self._parameters["safety_margin_fraction"])
        return result.feasible and all(
            value <= -margin for value in result.constraints[:5]
        )

    @staticmethod
    def select_candidate(candidates: Sequence[Candidate]) -> Candidate:
        """Apply the frozen equal-weight augmented Tchebycheff selector."""

        values = tuple(candidates)
        if not values:
            raise WGTRRBindingError("selector requires a nonempty archive")
        objective_matrix = np.asarray(
            [candidate.evaluation.objectives for candidate in values],
            dtype=float,
        )
        if objective_matrix.ndim != 2 or objective_matrix.shape[1] != 3:
            raise WGTRRBindingError(
                "rolling selector requires three objectives"
            )
        if np.any(objective_matrix < 0.0) or not np.all(
            np.isfinite(objective_matrix)
        ):
            raise WGTRRBindingError(
                "rolling selector objectives must be finite and nonnegative"
            )
        transformed = objective_matrix / (1.0 + objective_matrix)
        ideal = np.min(transformed, axis=0)
        weighted_distance = (transformed - ideal) / 3.0
        scores = np.max(weighted_distance, axis=1) + 1e-6 * np.sum(
            weighted_distance, axis=1
        )
        ranked = sorted(
            zip(scores.tolist(), values, strict=True),
            key=lambda pair: (pair[0], pair[1].candidate_id),
        )
        return ranked[0][1]

    @staticmethod
    def first_action(vector: Sequence[float]) -> np.ndarray:
        values = np.asarray(vector, dtype=float)
        if values.shape != (12,) or not np.all(np.isfinite(values)):
            raise WGTRRBindingError("rolling vector has wrong shape")
        return values.reshape(6, 2)[0].copy()

    def shift_solution(self, vector: Sequence[float]) -> np.ndarray:
        values = np.asarray(vector, dtype=float)
        if values.shape != (12,) or not np.all(np.isfinite(values)):
            raise WGTRRBindingError("rolling shift vector has wrong shape")
        actions = values.reshape(6, 2)
        return np.clip(
            np.vstack([actions[1:], actions[-1]]), -1.0, 1.0
        ).reshape(-1)

    def fallback_action(self, event_id: int) -> np.ndarray:
        if not 0 <= event_id < 20:
            raise WGTRRBindingError("rolling event must be in 0..19")
        return np.zeros(2, dtype=float)

    @staticmethod
    def _transform(value: float) -> float:
        value = max(float(value), 0.0)
        return value / (1.0 + value)

    def _realized_components(
        self,
        action: np.ndarray,
        prior_action: np.ndarray,
        state_after: np.ndarray,
        event_id: int,
    ) -> tuple[list[float], bool]:
        _, bounds, phase_shift = self._released_parameters(event_id)
        reference = self._reference(event_id + 1, phase_shift)
        tracking = float(np.mean(((state_after - reference) / bounds) ** 2))
        effort = float(np.mean(action**2))
        delta = action - prior_action
        rate_limit = float(self._parameters["rate_limit"])
        rate = float(np.mean((delta / rate_limit) ** 2))
        center = np.asarray(
            self._parameters["obstacle_center"], dtype=float
        )
        radius = float(self._parameters["obstacle_radius"])
        hard = np.concatenate(
            [
                np.abs(state_after) / bounds - 1.0,
                np.abs(delta) / rate_limit - 1.0,
                np.asarray(
                    [
                        1.0
                        - np.sum((state_after - center) ** 2) / radius**2
                    ]
                ),
            ]
        )
        return (
            [self._transform(value) for value in (tracking, effort, rate)],
            bool(np.any(hard > 0.0)),
        )

    def execute(
        self,
        action: Sequence[float],
        event_id: int,
        committed: bool,
        ledger: EvaluationLedger,
    ) -> Mapping[str, Any]:
        values = np.asarray(action, dtype=float)
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise WGTRRBindingError("invalid rolling execution action")
        if self._information is None or self._information.decision_time != event_id:
            raise WGTRRBindingError(
                "rolling event must be frozen before execution"
            )
        before = self._state.copy()
        prior = self._previous_action.copy()
        a, _, _ = self._released_parameters(event_id)
        disturbance = np.asarray(
            self._parameters["disturbance_sequence"][event_id], dtype=float
        )
        after = (
            a * before
            + self._rotation_b() @ values
            + self._nominal_drift(event_id)
            + disturbance
        )
        exec_components, hard_violation = self._realized_components(
            values, prior, after, event_id
        )
        hold = (
            a * before
            + self._rotation_b() @ prior
            + self._nominal_drift(event_id)
            + disturbance
        )
        ref_components, _ = self._realized_components(
            prior, prior, hold, event_id
        )
        ell_exec = float(np.mean(exec_components))
        ell_ref = float(np.mean(ref_components))
        scale = max(1e-12, abs(ell_ref) + 0.1)
        self._state = after
        self._previous_action = values.copy()
        self._information = None
        self._frozen_state = None
        ledger.record_execution()
        missing = event_id in set(
            self._parameters.get("feedback_missing_events", ())
        )
        self._released_history.append(
            {
                "event_id": event_id,
                "committed": bool(committed),
                "realized_next_state": after.tolist(),
                "feedback_available": not missing,
            }
        )
        available = bool(committed and not missing)
        return {
            "available": available,
            "ell_exec": ell_exec if available else None,
            "ell_ref": ell_ref if available else None,
            "s_exec": scale if available else None,
            "hard_constraint_violation": (
                hard_violation if available else None
            ),
            "released_at": event_id + 1,
        }
