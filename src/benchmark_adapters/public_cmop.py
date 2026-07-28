"""Result-blind F23 bridges for public constrained multi-objective evaluators.

Port provenance:
    FORMAL_V1/dt_ramde_formal/adapters/public_cmop.py
    SHA-256 b7ba30e8626bd4ada74e3421caf9e906d603cb65700379b16287578fd9bc8183

Only the registered bridge semantics are ported. Public problem equations are
deliberately not vendored here: callers must supply an explicit, versioned,
hash-bound external evaluator.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import re
from typing import Any

import numpy as np

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


Evaluation = Callable[
    [Sequence[float], int],
    tuple[Sequence[float], Sequence[float]],
]
ReleaseMetadata = Callable[[int], Mapping[str, Any]]


class PublicAdapterContractError(ValueError):
    """A public evaluator bridge differs from the registered F23 contract."""


def _validate_bounds(
    lower: Sequence[float], upper: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    if (
        lower_array.ndim != 1
        or lower_array.size < 1
        or upper_array.shape != lower_array.shape
        or not np.all(np.isfinite(lower_array))
        or not np.all(np.isfinite(upper_array))
        or not np.all(lower_array <= upper_array)
    ):
        raise PublicAdapterContractError("public adapter bounds are invalid")
    lower_array.setflags(write=False)
    upper_array.setflags(write=False)
    return lower_array, upper_array


def _validate_names(names: Sequence[str], *, label: str) -> tuple[str, ...]:
    values = tuple(str(name) for name in names)
    if not values or any(not name for name in values):
        raise PublicAdapterContractError(f"{label} names must be explicit")
    if len(set(values)) != len(values):
        raise PublicAdapterContractError(f"{label} names must be unique")
    return values


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


class _PublicCMOPBase:
    """Shared result-blind bridge bound to the v1.1 evaluation ledger."""

    atomic_steps_per_evaluation = 1
    records_execution_transition = False
    adapter_version = "1.0.0-r2-fixture"
    bridge_role = "r2_result_blind_fixture_bridge"
    bridge_stage = "R2"
    execution_authority = "R2_CORRECTNESS_ONLY"
    registered_benchmark_evaluator = False
    registered_effect_instance = False
    formal_effect_execution_allowed = False
    evaluator_interface_version: str

    def __init__(
        self,
        *,
        suite_id: str,
        problem_id: str,
        evaluator_version: str,
        fixture_evaluator_sha256: str,
        lower: Sequence[float],
        upper: Sequence[float],
        objective_names: Sequence[str],
        constraint_names: Sequence[str],
        evaluator: Evaluation,
    ) -> None:
        if not suite_id or not problem_id or not evaluator_version:
            raise PublicAdapterContractError(
                "public evaluator identity must be explicit"
            )
        if evaluator_version != self.evaluator_interface_version:
            raise PublicAdapterContractError(
                "evaluator interface version differs from the frozen F23 binding"
            )
        normalized_hash = fixture_evaluator_sha256.lower()
        if len(normalized_hash) != 64 or any(
            character not in "0123456789abcdef"
            for character in normalized_hash
        ):
            raise PublicAdapterContractError(
                "public evaluator requires a SHA-256 identity"
            )
        if not callable(evaluator):
            raise PublicAdapterContractError("public evaluator must be callable")

        self.suite_id = suite_id
        self.problem_id = problem_id
        self.fixture_evaluator_sha256 = normalized_hash
        self.lower_bounds, self.upper_bounds = _validate_bounds(lower, upper)
        self.decision_dimension = int(self.lower_bounds.size)
        self.objective_names = _validate_names(
            objective_names, label="objective"
        )
        self.constraint_names = _validate_names(
            constraint_names, label="constraint"
        )
        self.constraint_scales = (1.0,) * len(self.constraint_names)
        self._external_evaluator = evaluator
        self._information: InformationSnapshot | None = None

        self._evaluator = SharedEvaluator(
            objective_names=self.objective_names,
            constraint_names=self.constraint_names,
            evaluate_joint=self._evaluate_external,
            evaluate_joint_batch=self._evaluate_external_batch,
        )

    def _evaluate_external(
        self,
        vector: Sequence[float],
        information: InformationSnapshot,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        values = np.asarray(vector, dtype=float)
        if (
            values.shape != (self.decision_dimension,)
            or not np.all(np.isfinite(values))
            or np.any(values < self.lower_bounds)
            or np.any(values > self.upper_bounds)
        ):
            raise FloatingPointError(
                "public candidate is non-finite, out of bounds, or wrong shape"
            )
        objectives, constraints = self._external_evaluator(
            values.copy(), information.decision_time
        )
        return (
            tuple(float(value) for value in objectives),
            tuple(float(value) for value in constraints),
        )

    def _evaluate_external_batch(
        self,
        vectors: Sequence[Sequence[float]],
        information: InformationSnapshot,
    ) -> tuple[tuple[Sequence[float], Sequence[float]], ...]:
        values = np.asarray(vectors, dtype=float)
        if (
            values.ndim != 2
            or values.shape[1:] != (self.decision_dimension,)
            or not np.all(np.isfinite(values))
            or np.any(values < self.lower_bounds)
            or np.any(values > self.upper_bounds)
        ):
            raise FloatingPointError(
                "public candidate batch is non-finite, out of bounds, "
                "or wrong shape"
            )
        batch_evaluator = getattr(
            self._external_evaluator, "evaluate_batch", None
        )
        if not callable(batch_evaluator):
            raise NotImplementedError(
                "external evaluator has no ordered batch kernel"
            )
        rows = tuple(
            batch_evaluator(values.copy(), information.decision_time)
        )
        return tuple(
            (tuple(objectives), tuple(constraints))
            for objectives, constraints in rows
        )

    def identity(self) -> Mapping[str, Any]:
        return {
            "target_suite_id": self.suite_id,
            "target_problem_id": self.problem_id,
            "split": "r2_public_bridge_correctness_fixture",
            "target_registered_split": "public_fixed_confirmatory",
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "evaluator_interface_version": self.evaluator_interface_version,
            "fixture_evaluator_sha256": self.fixture_evaluator_sha256,
            "bridge_role": self.bridge_role,
            "registered_effect_instance": self.registered_effect_instance,
            "formal_effect_execution_allowed": (
                self.formal_effect_execution_allowed
            ),
        }

    def evaluate(
        self,
        vector: Sequence[float],
        event_id: int,
        ledger: EvaluationLedger,
        candidate_id: str,
    ) -> EvaluationResult:
        information = self._information
        if information is None or information.decision_time != event_id:
            raise PublicAdapterContractError(
                "freeze_information must bind the current event before evaluate"
            )
        return self._evaluator.evaluate(
            vector=vector,
            event_id=event_id,
            candidate_id=candidate_id,
            information=information,
            ledger=ledger,
            atomic_steps=self.atomic_steps_per_evaluation,
            origin="public_external_evaluator",
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
            raise PublicAdapterContractError(
                "freeze_information must bind the current event before evaluate"
            )
        return self._evaluator.evaluate_batch(
            vectors=vectors,
            event_id=event_id,
            candidate_ids=candidate_ids,
            information=information,
            ledger=ledger,
            atomic_steps=self.atomic_steps_per_evaluation,
            origin="public_external_evaluator",
        )

    @staticmethod
    def safety_filter(result: EvaluationResult, event_id: int) -> bool:
        del event_id
        return result.feasible

    def shift_solution(self, vector: Sequence[float]) -> np.ndarray:
        values = np.asarray(vector, dtype=float)
        if (
            values.shape != (self.decision_dimension,)
            or not np.all(np.isfinite(values))
        ):
            raise PublicAdapterContractError("shift input has wrong shape")
        return values.copy()

    def first_action(self, vector: Sequence[float]) -> np.ndarray:
        return self.shift_solution(vector)

    def fallback_action(self, event_id: int) -> np.ndarray:
        del event_id
        return (self.lower_bounds + self.upper_bounds) / 2.0

    def execute(
        self,
        action: Sequence[float],
        event_id: int,
        committed: bool,
        ledger: EvaluationLedger,
    ) -> Mapping[str, Any]:
        del committed
        self.shift_solution(action)
        if self.records_execution_transition:
            ledger.record_execution()
        return {
            "available": False,
            "ell_exec": None,
            "ell_ref": None,
            "s_exec": None,
            "hard_constraint_violation": None,
            "released_at": event_id + 1,
            "reason": "MISSING_BY_DESIGN_PUBLIC_BENCHMARK",
        }


class StaticCMOPPublicAdapter(_PublicCMOPBase):
    """R2 TS1 correctness bridge targeting the F23 static interface."""

    adapter_id = "BIND-STATIC-CMOP-01/R2-CORRECTNESS-BRIDGE"
    evaluator_interface_version = "STATIC-CMOP-EVAL-1.0.0"
    _REGISTRY = {
        "DAS-CMOP-PLATEMO-4.15": (r"DASCMOP([1-9])", 30, {2, 3}),
        "LIR-CMOP-PLATEMO-4.15": (
            r"LIRCMOP([1-9]|1[0-4])",
            30,
            {2, 3},
        ),
        "LIR-CMOP-JMETALPY-1.7.0": (
            r"LIRCMOP([1-9]|1[0-4])",
            30,
            {2, 3},
        ),
        "LIR-CMOP-PAPER-2019-TABLE-8": (
            r"LIRCMOP([1-9]|1[0-4])",
            30,
            {2, 3},
        ),
        "MW-PLATEMO-4.15": (r"MW([1-9]|1[0-4])", 15, {2, 3}),
        "RWCMOP-PLATEMO-4.15": (r"RWMOP([1-9]|[1-4][0-9]|50)", None, {2, 3, 4, 5}),
    }

    def __init__(self, **kwargs: Any) -> None:
        suite_id = str(kwargs.get("suite_id", ""))
        if suite_id not in self._REGISTRY:
            raise PublicAdapterContractError(
                "static suite is not registered in F23"
            )
        pattern, dimension, objective_counts = self._REGISTRY[suite_id]
        problem_id = str(kwargs.get("problem_id", ""))
        if re.fullmatch(pattern, problem_id) is None:
            raise PublicAdapterContractError(
                "static problem is not registered for the selected F23 suite"
            )
        if dimension is None:
            raise PublicAdapterContractError(
                "RWCMOP requires R4 problem-specific dimension binding"
            )
        super().__init__(**kwargs)
        if self.decision_dimension != dimension:
            raise PublicAdapterContractError(
                "static decision dimension differs from the F23 registry"
            )
        if len(self.objective_names) not in objective_counts:
            raise PublicAdapterContractError(
                "static objective count differs from the F23 registry"
            )

    def freeze_information(
        self, event_id: int, feedback: Mapping[str, Any] | None
    ) -> InformationSnapshot:
        if event_id != 0 or feedback is not None:
            raise PublicAdapterContractError(
                "static public CMOP bridge is TS1 with no prior feedback"
            )
        self._information = freeze_information(
            decision_time=0,
            fields={
                "current_problem_identity": InformationField(
                    available_at=0,
                    value=dict(self.identity()),
                ),
                "frozen_interface_versions_and_hashes": InformationField(
                    available_at=0,
                    value={
                        "binding": "BIND-STATIC-CMOP-01",
                        "evaluator_interface_version": (
                            self.evaluator_interface_version
                        ),
                        "fixture_evaluator_sha256": (
                            self.fixture_evaluator_sha256
                        ),
                        "execution_authority": self.execution_authority,
                    },
                ),
            },
        )
        return self._information


class CDFPublicAdapter(_PublicCMOPBase):
    """R2 TS2 correctness bridge targeting the F23 CDF interface."""

    adapter_id = "BIND-CDF-DYNAMIC-01/R2-CORRECTNESS-BRIDGE"
    evaluator_interface_version = "CDF-EVAL-1.0.0"
    records_execution_transition = True

    def __init__(
        self,
        *,
        profile: str,
        release_metadata: ReleaseMetadata,
        **kwargs: Any,
    ) -> None:
        if kwargs.get("suite_id") not in {
            "CDF-1-15",
            "CDF-1-15-CMLSGA-1926A5A1-OPERATIONAL",
        } or kwargs.get("problem_id") not in {
            f"CDF{index}" for index in range(1, 16)
        }:
            raise PublicAdapterContractError(
                "CDF identity is not registered in F23"
            )
        if profile not in {"CDF-HARSH", "CDF-MILD"}:
            raise PublicAdapterContractError("unknown CDF profile")
        if not callable(release_metadata):
            raise PublicAdapterContractError(
                "CDF release_metadata must be callable"
            )
        super().__init__(**kwargs)
        if self.decision_dimension != 10:
            raise PublicAdapterContractError(
                "CDF decision dimension differs from the F23 registry"
            )
        if len(self.objective_names) != 2:
            raise PublicAdapterContractError(
                "CDF objective count differs from the F23 registry"
            )
        self.profile = profile
        self._release_metadata = release_metadata

    def identity(self) -> Mapping[str, Any]:
        return {**super().identity(), "profile": self.profile}

    def freeze_information(
        self, event_id: int, feedback: Mapping[str, Any] | None
    ) -> InformationSnapshot:
        if not 0 <= event_id < 60:
            raise PublicAdapterContractError("CDF event must be in 0..59")
        prohibited_feedback = _find_prohibited_key(feedback)
        if prohibited_feedback is not None:
            raise InformationBoundaryError(
                f"prohibited information field: {prohibited_feedback}"
            )
        if feedback is not None:
            if event_id == 0:
                raise PublicAdapterContractError(
                    "first CDF event cannot receive prior feedback"
                )
            release_time = feedback.get("released_at")
            missing_fields = (
                "ell_exec",
                "ell_ref",
                "s_exec",
                "hard_constraint_violation",
            )
            allowed_fields = {
                "available",
                "reason",
                "released_at",
                *missing_fields,
            }
            if (
                feedback.get("available") is not False
                or feedback.get("reason")
                != "MISSING_BY_DESIGN_PUBLIC_BENCHMARK"
                or type(release_time) is not int
                or release_time != event_id
                or any(
                    name not in feedback or feedback[name] is not None
                    for name in missing_fields
                )
                or set(feedback) - allowed_fields
            ):
                raise PublicAdapterContractError(
                    "CDF feedback must remain missing by design and be "
                    "released at the current integer event"
                )
        released = dict(self._release_metadata(event_id))
        prohibited = _find_prohibited_key(released)
        if prohibited is not None:
            raise InformationBoundaryError(
                f"prohibited information field: {prohibited}"
            )
        fields = {
            "current_dynamic_environment": InformationField(
                available_at=event_id,
                value=released,
            ),
            "current_problem_identity": InformationField(
                available_at=event_id,
                value=dict(self.identity()),
            ),
            "frozen_interface_versions_and_hashes": InformationField(
                available_at=0,
                value={
                        "binding": "BIND-CDF-DYNAMIC-01",
                        "evaluator_interface_version": (
                            self.evaluator_interface_version
                        ),
                        "fixture_evaluator_sha256": (
                            self.fixture_evaluator_sha256
                        ),
                        "execution_authority": self.execution_authority,
                    },
                ),
        }
        if feedback is not None:
            fields["prior_missing_execution_feedback"] = InformationField(
                available_at=event_id,
                value=dict(feedback),
            )
        self._information = freeze_information(
            decision_time=event_id,
            fields=fields,
        )
        return self._information
