"""v1.1 protocol identity and fail-closed R2 execution scope.

Port provenance:
    FORMAL_V1/dt_ramde_formal/contracts.py
    SHA-256 3412b62dd3f9331dcda283c575ed813a72421dc6fbc48785c1a7a6a439d9fcfd

The v1.0 run values and rolling-only method registry are intentionally not
ported. This module binds the current v1.1 overlay and permits correctness
fixtures only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite
import re
from typing import Any


class ConfigurationError(ValueError):
    """A requested action cannot be reconciled with the active contracts."""


class ExecutionScope(str, Enum):
    """Scopes understood by v1.1, including scopes R2 must reject."""

    UNIT_TEST_FIXTURE = "unit_test_fixture"
    PUBLIC_CORRECTNESS_FIXTURE = "public_correctness_fixture"
    ENGINEERING_PILOT = "engineering_pilot"
    BENCHMARK_EFFECT = "benchmark_effect"
    WEIGHT_EFFECT = "weight_effect"
    HIDDEN = "hidden"
    CONFIRMATORY = "confirmatory"


@dataclass(frozen=True)
class ContractBindings:
    """Exact identities that authorize the current R2 implementation work."""

    protocol_id: str = "WGT-JOURNAL-2026-01"
    protocol_version: str = "v1.1.8-r2-shade-success-frozen"
    r1_readiness_record_id: str = "WGT-V11-R1-READINESS-20260723-01"
    r2_start_record_id: str = "WGT-V11-R2-START-20260724-01"
    f22_contract_id: str = "WGT-DT-RAMDE-F22-01"
    f23_contract_id: str = "WGT-F23-BENCHMARK-CONTRACT-01"
    f24_contract_id: str = "WGT-F24-STATISTICS-CONTRACT-01"
    f22_f23_shade_amendment_id: str = (
        "AMEND-V11-F22-F23-SHADE-20260724-01"
    )
    f22_shade_success_overlay_id: str = (
        "WGT-DT-RAMDE-F22-SHADE-SUCCESS-OVERLAY-01"
    )
    f23_shade_success_overlay_id: str = (
        "WGT-F23-SHADE-SUCCESS-BINDING-OVERLAY-01"
    )

    def validate(self) -> None:
        if self != ContractBindings():
            raise ConfigurationError(
                "contract identity differs from the active v1.1 R2 authority set"
            )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class R2ExecutionRequest:
    """A fail-closed request envelope for work allowed during R2."""

    scope: ExecutionScope
    contracts: ContractBindings = ContractBindings()
    participant_data_requested: bool = False
    effect_estimation_requested: bool = False
    hidden_generation_requested: bool = False
    results_writing_requested: bool = False
    remote_git_mutation_requested: bool = False
    release_or_distribution_requested: bool = False

    def validate(self) -> None:
        self.contracts.validate()
        allowed_scopes = {
            ExecutionScope.UNIT_TEST_FIXTURE,
            ExecutionScope.PUBLIC_CORRECTNESS_FIXTURE,
        }
        if self.scope not in allowed_scopes:
            raise ConfigurationError(
                f"{self.scope.value!r} is outside the R2 correctness scope"
            )

        prohibited = {
            "participant_data_requested": self.participant_data_requested,
            "effect_estimation_requested": self.effect_estimation_requested,
            "hidden_generation_requested": self.hidden_generation_requested,
            "results_writing_requested": self.results_writing_requested,
            "remote_git_mutation_requested": self.remote_git_mutation_requested,
            "release_or_distribution_requested": self.release_or_distribution_requested,
        }
        requested = sorted(name for name, enabled in prohibited.items() if enabled)
        if requested:
            raise ConfigurationError(
                "R2 prohibited permission requested: " + ", ".join(requested)
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scope"] = self.scope.value
        return payload


@dataclass(frozen=True)
class R6ContractBindings:
    """Exact result-blind authority inherited by the isolated R6 pilot."""

    protocol_id: str = "WGT-JOURNAL-2026-01"
    r5_contract_id: str = (
        "WGT-V11-R5-ENDPOINT-STATISTICS-SAMPLE-SEED-RESOURCE-01"
    )
    r5_contract_sha256: str = (
        "4e2dd0a0f4a97b57d71dd13eb60aa8a3c3eb34f0708aae609d50a31d155f6554"
    )
    r5_base_commit: str = "4526bc1cc3123bdadc331117c94e3a09f42fe2eb"
    author_authorization_text: str = "R6_PILOT_AUTHORIZED"
    execution_authority: str = "development_only_not_effect_evidence"

    def validate(self) -> None:
        if self != R6ContractBindings():
            raise ConfigurationError(
                "R6 binding differs from the author-authorized result-blind contract"
            )


@dataclass(frozen=True)
class R6ExecutionRequest:
    """Fail-closed permissions for an isolated, nonformal engineering pilot."""

    scope: ExecutionScope
    contracts: R6ContractBindings = R6ContractBindings()
    nonformal_development_fixture_acknowledged: bool = True
    participant_data_requested: bool = False
    effect_estimation_requested: bool = False
    method_comparison_requested: bool = False
    formal_subject_generation_requested: bool = False
    hidden_generation_requested: bool = False
    results_writing_requested: bool = False
    remote_git_mutation_requested: bool = False
    release_or_distribution_requested: bool = False

    def validate(self) -> None:
        self.contracts.validate()
        if self.scope is not ExecutionScope.ENGINEERING_PILOT:
            raise ConfigurationError(
                f"{self.scope.value!r} is outside the R6 engineering-pilot scope"
            )
        if not self.nonformal_development_fixture_acknowledged:
            raise ConfigurationError(
                "R6 requires an explicit nonformal development-fixture acknowledgement"
            )
        prohibited = {
            "participant_data_requested": self.participant_data_requested,
            "effect_estimation_requested": self.effect_estimation_requested,
            "method_comparison_requested": self.method_comparison_requested,
            "formal_subject_generation_requested": (
                self.formal_subject_generation_requested
            ),
            "hidden_generation_requested": self.hidden_generation_requested,
            "results_writing_requested": self.results_writing_requested,
            "remote_git_mutation_requested": self.remote_git_mutation_requested,
            "release_or_distribution_requested": (
                self.release_or_distribution_requested
            ),
        }
        requested = sorted(name for name, enabled in prohibited.items() if enabled)
        if requested:
            raise ConfigurationError(
                "R6 prohibited permission requested: " + ", ".join(requested)
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scope"] = self.scope.value
        return payload


@dataclass(frozen=True)
class R8ContractBindings:
    """Frozen upstream identities required by an R8 formal request."""

    protocol_id: str
    r5_contract_id: str
    r5_contract_sha256: str
    r5a_contract_id: str
    r5a_contract_sha256: str
    r7_contract_id: str
    r7_contract_sha256: str
    formal_schedule_id: str
    formal_schedule_sha256: str
    source_git_commit: str
    source_git_tree: str

    def validate(self) -> None:
        expected = {
            "protocol_id": "WGT-JOURNAL-2026-01",
            "r5_contract_id": (
                "WGT-V11-R5-ENDPOINT-STATISTICS-SAMPLE-SEED-RESOURCE-01"
            ),
            "r5_contract_sha256": (
                "4e2dd0a0f4a97b57d71dd13eb60aa8a3c3eb34f0708aae609d50a31d155f6554"
            ),
            "r5a_contract_id": "WGT-V11-R5A-E3-INPUT-CONTRACT-01",
            "r5a_contract_sha256": (
                "a7275dc1624fc2167c0ed5a599f9b5cb3297151037c47c5b85fb27d38e857424"
            ),
            "r7_contract_id": "WGT-V11-R7-FORMAL-EXECUTION-CONTRACT-01",
            "formal_schedule_id": "WGT-V11-R7-FORMAL-SCHEDULE-01",
            "formal_schedule_sha256": (
                "40ea633532a3ba2c461ae47925a91ccae305bafac397e6246ced1951fa6e8969"
            ),
        }
        payload = asdict(self)
        for key, value in expected.items():
            if payload[key] != value:
                raise ConfigurationError(f"R8 {key} differs from R7 freeze")
        if re.fullmatch(r"[0-9a-f]{64}", self.r7_contract_sha256) is None:
            raise ConfigurationError("R8 R7 contract SHA-256 is invalid")
        for label, value in {
            "source_git_commit": self.source_git_commit,
            "source_git_tree": self.source_git_tree,
        }.items():
            if re.fullmatch(r"[0-9a-f]{40}", value) is None:
                raise ConfigurationError(
                    f"R8 {label} is not a full Git object identity"
                )


@dataclass(frozen=True)
class R8ExecutionRequest:
    """One-time, exact-command envelope for formal public effect execution."""

    scope: ExecutionScope
    companion_scope: ExecutionScope
    contracts: R8ContractBindings
    request_id: str
    frozen_exact_command: str
    author_confirmation_text: str
    author_exact_command_confirmed: bool
    formal_effect_execution_requested: bool = True
    participant_data_requested: bool = False
    hidden_generation_requested: bool = False
    results_analysis_requested: bool = False
    results_writing_requested: bool = False
    remote_git_mutation_requested: bool = False
    release_or_distribution_requested: bool = False

    def validate(self) -> None:
        self.contracts.validate()
        if self.scope not in {
            ExecutionScope.BENCHMARK_EFFECT,
            ExecutionScope.WEIGHT_EFFECT,
        } or {
            self.scope,
            self.companion_scope,
        } != {
            ExecutionScope.BENCHMARK_EFFECT,
            ExecutionScope.WEIGHT_EFFECT,
        }:
            raise ConfigurationError(
                "R8 requires exactly benchmark_effect plus weight_effect scope"
            )
        if self.request_id != "WGT-V11-R8-EXECUTION-REQUEST-20260725-01":
            raise ConfigurationError("unexpected R8 request identity")
        if (
            not self.author_exact_command_confirmed
            or not self.frozen_exact_command
            or self.author_confirmation_text != self.frozen_exact_command
        ):
            raise ConfigurationError(
                "R8 requires verbatim author confirmation of the frozen command"
            )
        if not self.formal_effect_execution_requested:
            raise ConfigurationError(
                "R8 request must explicitly request formal effect execution"
            )
        prohibited = {
            "participant_data_requested": self.participant_data_requested,
            "hidden_generation_requested": self.hidden_generation_requested,
            "results_analysis_requested": self.results_analysis_requested,
            "results_writing_requested": self.results_writing_requested,
            "remote_git_mutation_requested": self.remote_git_mutation_requested,
            "release_or_distribution_requested": (
                self.release_or_distribution_requested
            ),
        }
        requested = sorted(name for name, enabled in prohibited.items() if enabled)
        if requested:
            raise ConfigurationError(
                "R8 prohibited permission requested: " + ", ".join(requested)
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scope"] = self.scope.value
        payload["companion_scope"] = self.companion_scope.value
        return payload


@dataclass(frozen=True)
class R8CCorrectiveContractBindings:
    """Corrective 100/100 formal identities, distinct from historical R8."""

    protocol_id: str
    r5_contract_id: str
    r5_contract_sha256: str
    r5a_contract_id: str
    r5a_contract_sha256: str
    corrective_protocol_id: str
    corrective_protocol_sha256: str
    r8c_formal_contract_id: str
    r8c_formal_contract_sha256: str
    formal_schedule_id: str
    formal_schedule_sha256: str
    source_git_commit: str
    source_git_tree: str

    def validate(self) -> None:
        expected_common = {
            "protocol_id": "WGT-JOURNAL-2026-01",
            "r5_contract_id": (
                "WGT-V11-R5-ENDPOINT-STATISTICS-SAMPLE-SEED-RESOURCE-01"
            ),
            "r5_contract_sha256": (
                "4e2dd0a0f4a97b57d71dd13eb60aa8a3c3eb34f0708aae609d50a31d155f6554"
            ),
            "r5a_contract_id": "WGT-V11-R5A-E3-INPUT-CONTRACT-01",
            "r5a_contract_sha256": (
                "a7275dc1624fc2167c0ed5a599f9b5cb3297151037c47c5b85fb27d38e857424"
            ),
            "corrective_protocol_id": (
                "WGT-V11-R8C-RESULT-BLIND-CORRECTIVE-PROTOCOL-01"
            ),
            "corrective_protocol_sha256": (
                "dfe74d041f36b12fd13cb86e1fa2bba5483bbd871a7749b2c98e09160ee39b43"
            ),
        }
        payload = asdict(self)
        for key, value in expected_common.items():
            if payload[key] != value:
                raise ConfigurationError(
                    f"R8C {key} differs from the corrective freeze"
                )
        allowed_phase_identities = {
            (
                "WGT-V11-R8C-FORMAL-EXECUTION-CONTRACT-01",
                "WGT-V11-R8C-FORMAL-SCHEDULE-01",
            ),
            (
                "WGT-V11-R8C-E1E2-TARGET-QUALIFIED-"
                "FORMAL-EXECUTION-CONTRACT-01",
                "WGT-V11-R8C-E1E2-FORMAL-SCHEDULE-01",
            ),
        }
        if (
            self.r8c_formal_contract_id,
            self.formal_schedule_id,
        ) not in allowed_phase_identities:
            raise ConfigurationError(
                "R8C formal contract/schedule phase identity is invalid"
            )
        for label, value in {
            "r8c_formal_contract_sha256": self.r8c_formal_contract_sha256,
            "formal_schedule_sha256": self.formal_schedule_sha256,
        }.items():
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ConfigurationError(f"R8C {label} is invalid")
        for label, value in {
            "source_git_commit": self.source_git_commit,
            "source_git_tree": self.source_git_tree,
        }.items():
            if re.fullmatch(r"[0-9a-f]{40}", value) is None:
                raise ConfigurationError(
                    f"R8C {label} is not a full Git object identity"
                )


@dataclass(frozen=True)
class R8CCorrectiveExecutionRequest:
    """One-time envelope for a future, separately authorized corrective run."""

    scope: ExecutionScope
    companion_scope: ExecutionScope
    contracts: R8CCorrectiveContractBindings
    request_id: str
    frozen_exact_command: str
    author_confirmation_text: str
    author_exact_command_confirmed: bool
    formal_effect_execution_requested: bool = True
    participant_data_requested: bool = False
    hidden_generation_requested: bool = False
    results_analysis_requested: bool = False
    results_writing_requested: bool = False
    remote_git_mutation_requested: bool = False
    release_or_distribution_requested: bool = False

    def validate(self) -> None:
        self.contracts.validate()
        if self.contracts.r8c_formal_contract_id == (
            "WGT-V11-R8C-E1E2-TARGET-QUALIFIED-"
            "FORMAL-EXECUTION-CONTRACT-01"
        ):
            if (
                self.scope is not ExecutionScope.BENCHMARK_EFFECT
                or self.companion_scope is not ExecutionScope.BENCHMARK_EFFECT
            ):
                raise ConfigurationError(
                    "R8C E1+E2 phase permits benchmark_effect scope only"
                )
            expected_request_id = (
                "WGT-V11-R8C-E1E2-TARGET-QUALIFIED-"
                "EXECUTION-REQUEST-20260726-01"
            )
        else:
            if {
                self.scope,
                self.companion_scope,
            } != {
                ExecutionScope.BENCHMARK_EFFECT,
                ExecutionScope.WEIGHT_EFFECT,
            }:
                raise ConfigurationError(
                    "R8C requires benchmark_effect plus weight_effect scope"
                )
            expected_request_id = (
                "WGT-V11-R8C-EXECUTION-REQUEST-20260726-01"
            )
        if self.request_id != expected_request_id:
            raise ConfigurationError("unexpected R8C request identity")
        if (
            not self.author_exact_command_confirmed
            or not self.frozen_exact_command
            or self.author_confirmation_text != self.frozen_exact_command
        ):
            raise ConfigurationError(
                "R8C requires a separate verbatim author confirmation"
            )
        if not self.formal_effect_execution_requested:
            raise ConfigurationError(
                "R8C must explicitly request formal effect execution"
            )
        prohibited = {
            "participant_data_requested": self.participant_data_requested,
            "hidden_generation_requested": self.hidden_generation_requested,
            "results_analysis_requested": self.results_analysis_requested,
            "results_writing_requested": self.results_writing_requested,
            "remote_git_mutation_requested": self.remote_git_mutation_requested,
            "release_or_distribution_requested": (
                self.release_or_distribution_requested
            ),
        }
        requested = sorted(
            name for name, enabled in prohibited.items() if enabled
        )
        if requested:
            raise ConfigurationError(
                "R8C prohibited permission requested: "
                + ", ".join(requested)
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scope"] = self.scope.value
        payload["companion_scope"] = self.companion_scope.value
        return payload


REGISTERED_F22_VARIANTS = (
    "FULL",
    "NO_CROSS_EVENT_MEMORY",
    "NO_EXECUTION_FEEDBACK",
    "NO_REJECTION_CREDIT",
    "NO_MEMORY_RESET_GATE",
    "NO_LINEAGE_CREDIT",
    "CROSS_EVENT_WARM_START_ONLY",
    "CROSS_EVENT_MEMORY_ONLY",
    "SHADE_ONLY",
)

FULL_AUDIT_MATERIALIZATION = "full"
COMPACT_CHECKPOINT_AUDIT_MATERIALIZATION = "compact_checkpoint"


@dataclass(frozen=True)
class AlgorithmConfig:
    """Explicit synthetic/public-fixture configuration for the R2 engine."""

    variant: str
    population_size: int
    cfe_per_event: int
    algorithm_seed: int
    max_events: int
    timing_mode: str
    method_label: str
    adapter_id: str
    adapter_version: str
    selector_id: str
    selector_version: str
    atomic_steps_per_evaluation: int
    event_time_limit_seconds: float
    configuration_evidence_id: str
    execution_request: (
        R2ExecutionRequest
        | R6ExecutionRequest
        | R8ExecutionRequest
        | R8CCorrectiveExecutionRequest
    )
    audit_materialization: str = FULL_AUDIT_MATERIALIZATION

    def validate(self) -> None:
        self.execution_request.validate()
        if self.audit_materialization not in {
            FULL_AUDIT_MATERIALIZATION,
            COMPACT_CHECKPOINT_AUDIT_MATERIALIZATION,
        }:
            raise ConfigurationError("unknown audit materialization mode")
        if self.audit_materialization == COMPACT_CHECKPOINT_AUDIT_MATERIALIZATION:
            permitted_compact_binding = (
                isinstance(self.execution_request, R8CCorrectiveExecutionRequest)
                and self.configuration_evidence_id
                == "WGT_V11_R8C_PUBLIC_E1_E2_FORMAL"
            ) or (
                isinstance(self.execution_request, R6ExecutionRequest)
                and self.configuration_evidence_id
                == "WGT_V11_R8C_E1E2_FULL_PATH_TARGET_QUALIFICATION_PILOT"
            )
            if not permitted_compact_binding:
                raise ConfigurationError(
                    "compact checkpoint audit materialization is restricted "
                    "to corrective E1+E2 formal execution or its qualification"
                )
        if self.variant not in REGISTERED_F22_VARIANTS:
            raise ConfigurationError("unregistered F22 variant")
        if self.population_size < 4:
            raise ConfigurationError("population_size must be at least four")
        if self.cfe_per_event < self.population_size:
            raise ConfigurationError(
                "CFE budget must cover current-event initialization"
            )
        if self.algorithm_seed < 0 or self.max_events < 1:
            raise ConfigurationError(
                "algorithm_seed/max_events must be nonnegative/positive"
            )
        if self.atomic_steps_per_evaluation < 1:
            raise ConfigurationError(
                "atomic_steps_per_evaluation must be positive"
            )
        if (
            not isfinite(self.event_time_limit_seconds)
            or self.event_time_limit_seconds <= 0.0
        ):
            raise ConfigurationError(
                "event_time_limit_seconds must be finite and positive"
            )
        if not all(
            (
                self.adapter_id,
                self.adapter_version,
                self.selector_id,
                self.selector_version,
                self.method_label,
            )
        ):
            raise ConfigurationError(
                "adapter, selector, and method identities must be explicit"
            )
        evidence_suffix = (
            "_FORMAL"
            if isinstance(
                self.execution_request,
                (R8ExecutionRequest, R8CCorrectiveExecutionRequest),
            )
            else (
                "_PILOT"
                if isinstance(self.execution_request, R6ExecutionRequest)
                else "_FIXTURE"
            )
        )
        if not self.configuration_evidence_id.endswith(evidence_suffix):
            raise ConfigurationError(
                "engine configuration evidence suffix differs from its execution scope"
            )
        if self.timing_mode not in {
            "TS2_fixed_periodic_replanning",
            "TS1_single_event",
        }:
            raise ConfigurationError("unknown timing mode")
        if self.timing_mode == "TS1_single_event":
            if self.max_events != 1:
                raise ConfigurationError("TS1 has exactly one event")
            if self.method_label == "DT-RAMDE_TS2_FULL":
                raise ConfigurationError("TS1 cannot use the FULL TS2 method label")
            if self.variant != "NO_CROSS_EVENT_MEMORY":
                raise ConfigurationError(
                    "R2 TS1 fixture must disable all cross-event components"
                )
        expected_method_label = (
            "F22_MG_STATIC"
            if self.timing_mode == "TS1_single_event"
            else (
                "DT-RAMDE_TS2_FULL"
                if self.variant == "FULL"
                else self.variant
            )
        )
        if self.method_label != expected_method_label:
            raise ConfigurationError(
                "method_label does not match the frozen variant/timing identity"
            )

    def to_dict(self) -> dict[str, Any]:
        from .state import COMPONENTS

        payload = asdict(self)
        payload["execution_request"] = self.execution_request.to_dict()
        if self.audit_materialization == FULL_AUDIT_MATERIALIZATION:
            payload.pop("audit_materialization")
        component = COMPONENTS[self.variant]
        payload["variant_components"] = {
            "M_g": True,
            "M_g_mode": component.mg_mode,
            "M_k": component.parameter_memory,
            "warm_start": component.warm_start,
            "execution_q": component.execution_credit,
            "rejection_q": component.rejection_credit,
            "lineage": component.lineage_mode,
            "soft_reset": component.soft_reset,
        }
        return payload
