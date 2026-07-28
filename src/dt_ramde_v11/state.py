"""DT-RAMDE v1.1 persistent state, delayed credit, lineage, and reset gate.

Semantic-port provenance:
    FORMAL_V1/dt_ramde_formal/core.py
    SHA-256 def7b3e8c3c41e088abe1fd50ffc6ab1a2511525151d38742b0eba38ed9f2369

The state types are separated from Pareto primitives so serialized resume
state cannot depend on domain adapters or mutable optimizer candidates.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation.contracts import TerminalCode
from evaluation.ledger import EvaluationLedger


class StateIntegrityError(RuntimeError):
    """A zero-tolerance state, credit, or resume invariant failed."""


@dataclass(frozen=True)
class VariantComponents:
    mg_mode: str
    parameter_memory: bool
    warm_start: bool
    execution_credit: bool
    rejection_credit: bool
    lineage_mode: str
    soft_reset: bool | str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


COMPONENTS: dict[str, VariantComponents] = {
    "FULL": VariantComponents(
        "F22_weighted_survivor", True, True, True, True, "chain", True
    ),
    "NO_CROSS_EVENT_MEMORY": VariantComponents(
        "F22_weighted_survivor", False, False, False, False, "off", False
    ),
    "NO_EXECUTION_FEEDBACK": VariantComponents(
        "F22_weighted_survivor",
        True,
        True,
        False,
        True,
        "chain",
        "rejection_only",
    ),
    "NO_REJECTION_CREDIT": VariantComponents(
        "F22_weighted_survivor",
        True,
        True,
        True,
        False,
        "chain",
        "execution_only",
    ),
    "NO_MEMORY_RESET_GATE": VariantComponents(
        "F22_weighted_survivor", True, True, True, True, "chain", False
    ),
    "NO_LINEAGE_CREDIT": VariantComponents(
        "F22_weighted_survivor",
        True,
        True,
        True,
        True,
        "event_last_5_uniform",
        True,
    ),
    "CROSS_EVENT_WARM_START_ONLY": VariantComponents(
        "F22_weighted_survivor", False, True, False, False, "off", False
    ),
    "CROSS_EVENT_MEMORY_ONLY": VariantComponents(
        "F22_weighted_survivor", True, False, True, True, "chain", True
    ),
    "SHADE_ONLY": VariantComponents(
        "WGT_SHADE_CMO_SUCCESS_01", False, False, False, False, "off", False
    ),
}


REJECTION_CREDIT: dict[TerminalCode, float] = {
    TerminalCode.REJECT_SAFETY_FILTER: -1.0,
    TerminalCode.REJECT_NO_FEASIBLE: -1.0,
    TerminalCode.REJECT_BUDGET_NO_FEASIBLE: -0.75,
    TerminalCode.REJECT_TIMEOUT: -0.75,
    TerminalCode.REJECT_NUMERICAL: -1.0,
}


NORMAL_TRANSITIONS = {
    ("UNINITIALIZED", "EVENT_OPEN"),
    ("EVENT_CLOSED", "EVENT_OPEN"),
    ("EVENT_OPEN", "PRIOR_CREDIT_RESOLVED"),
    ("PRIOR_CREDIT_RESOLVED", "RESET_GATE_APPLIED"),
    ("RESET_GATE_APPLIED", "SEARCH_INITIALIZED"),
    ("SEARCH_INITIALIZED", "SEARCHING"),
    ("SEARCHING", "TERMINAL_SELECTED"),
    ("SEARCHING", "TERMINAL_REJECTED"),
    ("TERMINAL_SELECTED", "ACTION_COMMITTED"),
    ("TERMINAL_REJECTED", "NO_ACTION_COMMITTED"),
    ("ACTION_COMMITTED", "PENDING_CREDIT_WRITTEN"),
    ("NO_ACTION_COMMITTED", "PENDING_CREDIT_WRITTEN"),
    ("PENDING_CREDIT_WRITTEN", "EVENT_CLOSED"),
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass
class MGState:
    memory_f: list[float]
    memory_cr: list[float]
    pointer: int

    @classmethod
    def initialize(cls) -> "MGState":
        return cls(memory_f=[0.5] * 10, memory_cr=[0.5] * 10, pointer=0)


@dataclass(frozen=True)
class ParameterAtom:
    f: float
    cr: float
    signed_credit: float
    source_event: int
    lineage_node_id: str
    age: int = 0

    def validate(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (self.f, self.cr, self.signed_credit)
        ):
            raise StateIntegrityError("parameter atom must be finite")
        if not (
            0.0 < self.f <= 1.0
            and 0.0 <= self.cr <= 1.0
            and -1.0 <= self.signed_credit <= 1.0
        ):
            raise StateIntegrityError("parameter atom is out of range")
        if self.source_event < 0 or self.age < 0 or not self.lineage_node_id:
            raise StateIntegrityError("parameter atom identity/age is invalid")


@dataclass(frozen=True)
class PendingCredit:
    pending_id: str
    source_event: int
    terminal_code: TerminalCode
    lineage_weights: tuple[tuple[str, float], ...]
    parameter_values: Mapping[str, tuple[float, float]]
    information_hash: str
    adapter_version: str

    def validate(self) -> None:
        if not self.pending_id or self.source_event < 0 or not self.adapter_version:
            raise StateIntegrityError("pending credit identity is invalid")
        if len(self.information_hash) != 64 or any(
            char not in "0123456789abcdef" for char in self.information_hash
        ):
            raise StateIntegrityError("pending information hash is invalid")
        if self.terminal_code not in {TerminalCode.ACCEPTED, *REJECTION_CREDIT}:
            raise StateIntegrityError("pending terminal code cannot receive credit")
        if self.lineage_weights:
            if any(
                not node_id or not math.isfinite(weight) or weight <= 0.0
                for node_id, weight in self.lineage_weights
            ):
                raise StateIntegrityError("pending lineage weight is invalid")
            if not math.isclose(
                sum(weight for _node_id, weight in self.lineage_weights),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise StateIntegrityError("pending lineage weights must sum to one")
        weight_ids = {node_id for node_id, _weight in self.lineage_weights}
        if weight_ids != set(self.parameter_values):
            raise StateIntegrityError(
                "pending parameter values must match lineage weights"
            )
        for f_value, cr_value in self.parameter_values.values():
            if not (
                math.isfinite(f_value)
                and math.isfinite(cr_value)
                and 0.0 < f_value <= 1.0
                and 0.0 <= cr_value <= 1.0
            ):
                raise StateIntegrityError("pending F/CR value is invalid")


@dataclass
class MemoryState:
    run_binding_id: str | None = None
    invalidated: bool = False
    invalidation_reason: str | None = None
    checkpoint_checksum: str | None = None
    bank: list[ParameterAtom] = field(default_factory=list)
    atom_audit: list[dict[str, Any]] = field(default_factory=list)
    tau: float = 0.5
    negative_streak: int = 0
    valid_feedback_count: int = 0
    cooldown_remaining: int = 0
    solution_memory: tuple[tuple[float, ...], ...] = ()
    pending_credit: PendingCredit | None = None
    consumed_pending_ids: list[str] = field(default_factory=list)
    reset_count: int = 0
    reset_log: list[dict[str, Any]] = field(default_factory=list)
    event_index: int = -1

    @property
    def transfer_allowed(self) -> bool:
        return self.cooldown_remaining == 0

    def validate(self) -> None:
        if self.checkpoint_checksum is not None and (
            len(self.checkpoint_checksum) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.checkpoint_checksum
            )
        ):
            raise StateIntegrityError("checkpoint checksum format is invalid")
        if self.invalidated != (self.invalidation_reason is not None):
            raise StateIntegrityError(
                "invalidated state must have exactly one invalidation reason"
            )
        if self.invalidation_reason is not None and not self.invalidation_reason:
            raise StateIntegrityError("invalidation reason must be nonempty")
        if self.run_binding_id is not None and (
            len(self.run_binding_id) != 24
            or any(
                character not in "0123456789abcdef"
                for character in self.run_binding_id
            )
        ):
            raise StateIntegrityError("run binding identity is invalid")
        if self.event_index >= 0 and self.run_binding_id is None:
            raise StateIntegrityError(
                "persisted event state requires a run binding identity"
            )
        if not math.isfinite(self.tau) or not 0.0 <= self.tau <= 1.0:
            raise StateIntegrityError("tau is invalid")
        counters = (
            self.negative_streak,
            self.valid_feedback_count,
            self.cooldown_remaining,
            self.reset_count,
        )
        if any(counter < 0 for counter in counters) or self.event_index < -1:
            raise StateIntegrityError("persistent counter is invalid")
        if len(self.bank) > 20:
            raise StateIntegrityError("parameter bank exceeds capacity")
        for sequence, record in enumerate(self.atom_audit, start=1):
            if (
                record.get("sequence") != sequence
                or record.get("action")
                not in {
                    "CREATED",
                    "AGED",
                    "EXPIRED",
                    "CAPACITY_EVICTED",
                    "RESET_REMOVED",
                }
                or not isinstance(record.get("atom"), Mapping)
            ):
                raise StateIntegrityError("parameter atom audit is invalid")
        keys: set[tuple[int, str]] = set()
        for atom in self.bank:
            atom.validate()
            key = (atom.source_event, atom.lineage_node_id)
            if key in keys:
                raise StateIntegrityError("duplicate parameter atom key")
            keys.add(key)
        if len(self.consumed_pending_ids) != len(set(self.consumed_pending_ids)):
            raise StateIntegrityError("consumed pending IDs are not unique")
        for vector in self.solution_memory:
            if not vector or not all(math.isfinite(value) for value in vector):
                raise StateIntegrityError("solution memory vector is invalid")
        if self.pending_credit is not None:
            self.pending_credit.validate()
            if (
                self.run_binding_id is not None
                and not self.pending_credit.pending_id.startswith(
                    f"{self.run_binding_id}:event:"
                )
            ):
                raise StateIntegrityError(
                    "pending credit does not match the run binding"
                )
            if self.pending_credit.pending_id in self.consumed_pending_ids:
                raise StateIntegrityError("pending credit was already consumed")

    def validate_timing(self, timing_mode: str) -> None:
        if timing_mode not in {
            "TS1_single_event",
            "TS2_fixed_periodic_replanning",
        }:
            raise StateIntegrityError("unknown timing mode")
        if timing_mode == "TS1_single_event":
            has_cross_event_state = bool(
                self.bank
                or self.atom_audit
                or self.solution_memory
                or self.pending_credit
                or self.consumed_pending_ids
                or self.negative_streak
                or self.valid_feedback_count
                or self.cooldown_remaining
                or self.reset_count
                or self.reset_log
                or self.tau != 0.5
            )
            if has_cross_event_state:
                raise StateIntegrityError("TS1 cross-event state must be empty")

    def _checkpoint_payload(self) -> dict[str, Any]:
        return {
            "run_binding_id": self.run_binding_id,
            "invalidated": self.invalidated,
            "invalidation_reason": self.invalidation_reason,
            "bank": [asdict(atom) for atom in self.bank],
            "atom_audit": [dict(item) for item in self.atom_audit],
            "tau": self.tau,
            "negative_streak": self.negative_streak,
            "valid_feedback_count": self.valid_feedback_count,
            "cooldown_remaining": self.cooldown_remaining,
            "solution_memory": [list(vector) for vector in self.solution_memory],
            "pending_credit": (
                None
                if self.pending_credit is None
                else {
                    **asdict(self.pending_credit),
                    "terminal_code": self.pending_credit.terminal_code.value,
                    "lineage_weights": [
                        list(item) for item in self.pending_credit.lineage_weights
                    ],
                    "parameter_values": {
                        key: list(value)
                        for key, value in self.pending_credit.parameter_values.items()
                    },
                }
            ),
            "consumed_pending_ids": list(self.consumed_pending_ids),
            "reset_count": self.reset_count,
            "reset_log": [dict(item) for item in self.reset_log],
            "event_index": self.event_index,
        }

    def compute_checkpoint_checksum(self) -> str:
        return _sha256_json(self._checkpoint_payload())

    def seal_checkpoint(self) -> None:
        self.checkpoint_checksum = self.compute_checkpoint_checksum()

    def verify_checkpoint(self) -> None:
        if self.checkpoint_checksum is None:
            raise StateIntegrityError("checkpoint checksum is missing")
        if self.checkpoint_checksum != self.compute_checkpoint_checksum():
            raise StateIntegrityError("checkpoint checksum mismatch")

    def to_dict(self) -> dict[str, Any]:
        payload = self._checkpoint_payload()
        payload["checkpoint_checksum"] = (
            self.checkpoint_checksum or _sha256_json(payload)
        )
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MemoryState":
        payload = dict(value)
        if "checkpoint_checksum" not in payload:
            raise StateIntegrityError("checkpoint checksum is missing")
        payload["bank"] = [
            ParameterAtom(**dict(atom)) for atom in payload.get("bank", [])
        ]
        payload["solution_memory"] = tuple(
            tuple(float(item) for item in vector)
            for vector in payload.get("solution_memory", [])
        )
        pending = payload.get("pending_credit")
        if pending is not None:
            pending_payload = dict(pending)
            pending_payload["terminal_code"] = TerminalCode(
                pending_payload["terminal_code"]
            )
            pending_payload["lineage_weights"] = tuple(
                (str(node_id), float(weight))
                for node_id, weight in pending_payload["lineage_weights"]
            )
            pending_payload["parameter_values"] = {
                str(key): (float(values[0]), float(values[1]))
                for key, values in pending_payload["parameter_values"].items()
            }
            payload["pending_credit"] = PendingCredit(**pending_payload)
        state = cls(**payload)
        state.validate()
        state.verify_checkpoint()
        return state


@dataclass(frozen=True)
class LineageNode:
    node_id: str
    event_id: int
    generation: int
    target_predecessor: str | None
    f: float | None
    cr: float | None
    survival: bool
    target_id: str | None = None
    pbest_id: str | None = None
    r1_id: str | None = None
    r2_id: str | None = None
    parameter_source: str | None = None
    j_rand: int | None = None
    pre_repair_hash: str = ""
    post_repair_hash: str = ""
    repaired: bool = False
    objectives: tuple[float, ...] = ()
    constraints: tuple[float, ...] = ()
    feasible: bool | None = None
    normalized_cv: float | None = None
    archive_admission: bool = False
    evaluation_status: str = "completed"


class LineageDAG:
    def __init__(self) -> None:
        self.nodes: dict[str, LineageNode] = {}
        self.accepted_order: list[str] = []

    def add(self, node: LineageNode) -> None:
        if not node.node_id or node.node_id in self.nodes:
            raise StateIntegrityError("duplicate or empty lineage node")
        if node.event_id < 0 or node.generation < -1:
            raise StateIntegrityError("lineage event/generation is invalid")
        if node.target_predecessor is not None and node.target_predecessor not in self.nodes:
            raise StateIntegrityError("lineage predecessor must already exist")
        if (node.f is None) != (node.cr is None):
            raise StateIntegrityError("lineage F/CR must both be present or absent")
        if node.f is not None and (
            not math.isfinite(node.f)
            or not math.isfinite(node.cr)
            or not 0.0 < node.f <= 1.0
            or not 0.0 <= node.cr <= 1.0
        ):
            raise StateIntegrityError("lineage F/CR values are invalid")
        if not all(
            math.isfinite(value)
            for value in (*node.objectives, *node.constraints)
        ):
            raise StateIntegrityError("lineage evaluation values must be finite")
        if node.normalized_cv is not None and (
            not math.isfinite(node.normalized_cv)
            or node.normalized_cv < 0.0
        ):
            raise StateIntegrityError("lineage normalized CV is invalid")
        if node.evaluation_status not in {
            "completed",
            "numerical_failure",
        }:
            raise StateIntegrityError("lineage evaluation status is invalid")
        self.nodes[node.node_id] = node
        if node.survival and node.f is not None:
            self.accepted_order.append(node.node_id)

    def mark_survival(self, node_id: str, survival: bool) -> None:
        if node_id not in self.nodes:
            raise StateIntegrityError("unknown lineage node")
        node = self.nodes[node_id]
        if node.survival == survival:
            return
        updated = replace(node, survival=survival)
        self.nodes[node_id] = updated
        if survival and updated.f is not None and node_id not in self.accepted_order:
            self.accepted_order.append(node_id)

    def mark_archive_admission(self, node_id: str) -> None:
        if node_id not in self.nodes:
            raise StateIntegrityError("unknown lineage node")
        node = self.nodes[node_id]
        if not node.archive_admission:
            self.nodes[node_id] = replace(node, archive_admission=True)

    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            dict(vars(self.nodes[node_id]))
            for node_id in sorted(self.nodes)
        )

    def credit_chain(
        self, terminal_node_id: str, *, mode: str
    ) -> tuple[tuple[str, float], ...]:
        if terminal_node_id not in self.nodes:
            raise StateIntegrityError("unknown terminal lineage node")
        if mode == "off":
            return ()
        if mode == "event_last_5_uniform":
            node_ids = self.accepted_order[-5:]
            if not node_ids:
                return ()
            return tuple((node_id, 1.0 / len(node_ids)) for node_id in node_ids)
        if mode != "chain":
            raise StateIntegrityError("unknown lineage credit mode")

        node_ids: list[str] = []
        current: str | None = terminal_node_id
        while current is not None and len(node_ids) < 5:
            node = self.nodes[current]
            if node.survival and node.f is not None:
                node_ids.append(current)
            current = node.target_predecessor
        raw = [2.0**-depth for depth in range(len(node_ids))]
        total = sum(raw)
        return tuple(
            (node_id, weight / total)
            for node_id, weight in zip(node_ids, raw, strict=True)
        )

    def parameter_values(
        self, weights: Sequence[tuple[str, float]]
    ) -> dict[str, tuple[float, float]]:
        result: dict[str, tuple[float, float]] = {}
        for node_id, _weight in weights:
            node = self.nodes[node_id]
            if node.f is None or node.cr is None:
                raise StateIntegrityError("lineage root cannot receive parameter credit")
            result[node_id] = (node.f, node.cr)
        return result


def _component(variant: str) -> VariantComponents:
    try:
        return COMPONENTS[variant]
    except KeyError as error:
        raise StateIntegrityError(f"unknown variant: {variant}") from error


def apply_valid_q(memory: MemoryState, q_value: float) -> None:
    if not math.isfinite(q_value) or not -1.0 <= q_value <= 1.0:
        raise StateIntegrityError("credit q must be finite and within [-1,1]")
    memory.tau = 0.8 * memory.tau + 0.2 * ((q_value + 1.0) / 2.0)
    memory.negative_streak = (
        memory.negative_streak + 1 if q_value < 0.0 else 0
    )
    memory.valid_feedback_count += 1


def resolve_pending(
    memory: MemoryState,
    *,
    variant: str,
    feedback: Mapping[str, Any] | None,
) -> tuple[list[ParameterAtom], float | None, str]:
    pending = memory.pending_credit
    if pending is None:
        return [], None, "NO_PENDING"
    pending.validate()
    if pending.pending_id in memory.consumed_pending_ids:
        raise StateIntegrityError("pending credit consumed more than once")
    component = _component(variant)

    q_value: float | None = None
    update_trust = False
    if pending.terminal_code is TerminalCode.ACCEPTED:
        if not component.execution_credit:
            total_credit = 0.25
            status = "EXECUTION_Q_DISABLED"
        elif feedback is None or not bool(feedback.get("available", False)):
            total_credit = 0.25
            status = "MISSING_EXPIRED"
        else:
            ell_exec = float(feedback["ell_exec"])
            ell_ref = float(feedback["ell_ref"])
            scale = float(feedback["s_exec"])
            if not (
                math.isfinite(ell_exec)
                and math.isfinite(ell_ref)
                and math.isfinite(scale)
                and scale > 0.0
            ):
                raise StateIntegrityError("execution feedback is invalid")
            q_value = float(np.clip((ell_ref - ell_exec) / scale, -1.0, 1.0))
            if bool(feedback.get("hard_constraint_violation", False)):
                q_value = -1.0
            total_credit = 0.25 + 0.75 * q_value
            status = "EXECUTION_Q_RESOLVED"
            update_trust = True
    else:
        if pending.terminal_code not in REJECTION_CREDIT:
            raise StateIntegrityError("unknown rejection terminal code")
        if not component.rejection_credit:
            memory.consumed_pending_ids.append(pending.pending_id)
            memory.pending_credit = None
            return [], None, "REJECTION_Q_DISABLED"
        q_value = REJECTION_CREDIT[pending.terminal_code]
        total_credit = q_value
        status = "REJECTION_Q_RESOLVED"
        update_trust = True

    atoms = (
        [
            ParameterAtom(
                f=pending.parameter_values[node_id][0],
                cr=pending.parameter_values[node_id][1],
                signed_credit=float(total_credit * weight),
                source_event=pending.source_event,
                lineage_node_id=node_id,
            )
            for node_id, weight in pending.lineage_weights
        ]
        if component.parameter_memory
        else []
    )
    for atom in atoms:
        atom.validate()

    memory.consumed_pending_ids.append(pending.pending_id)
    memory.pending_credit = None
    if update_trust and q_value is not None:
        apply_valid_q(memory, q_value)
    return atoms, q_value, status


def append_atoms(
    memory: MemoryState, atoms: Sequence[ParameterAtom]
) -> set[tuple[int, str]]:
    existing = {(atom.source_event, atom.lineage_node_id) for atom in memory.bank}
    added: set[tuple[int, str]] = set()
    for atom in atoms:
        atom.validate()
        key = (atom.source_event, atom.lineage_node_id)
        if key in existing or key in added:
            raise StateIntegrityError("duplicate parameter atom key")
        memory.bank.append(atom)
        memory.atom_audit.append(
            {
                "sequence": len(memory.atom_audit) + 1,
                "action": "CREATED",
                "atom": asdict(atom),
            }
        )
        added.add(key)
    return added


def age_prune_bank(
    memory: MemoryState, *, newly_added_keys: set[tuple[int, str]]
) -> None:
    aged: list[ParameterAtom] = []
    for atom in memory.bank:
        key = (atom.source_event, atom.lineage_node_id)
        age = atom.age if key in newly_added_keys else atom.age + 1
        updated = replace(atom, age=age)
        if age > 5:
            memory.atom_audit.append(
                {
                    "sequence": len(memory.atom_audit) + 1,
                    "action": "EXPIRED",
                    "atom": asdict(updated),
                }
            )
            continue
        if age != atom.age:
            memory.atom_audit.append(
                {
                    "sequence": len(memory.atom_audit) + 1,
                    "action": "AGED",
                    "atom": asdict(updated),
                }
            )
        aged.append(updated)
    aged.sort(key=lambda atom: (atom.source_event, atom.lineage_node_id))
    evicted = aged[:-20] if len(aged) > 20 else []
    for atom in evicted:
        memory.atom_audit.append(
            {
                "sequence": len(memory.atom_audit) + 1,
                "action": "CAPACITY_EVICTED",
                "atom": asdict(atom),
            }
        )
    memory.bank = aged[-20:]


def sample_atom(
    memory: MemoryState, rng: np.random.Generator
) -> ParameterAtom | None:
    eligible = [
        atom
        for atom in memory.bank
        if atom.signed_credit > 0.0 and atom.age <= 5
    ]
    if not eligible:
        return None
    weights = np.asarray(
        [atom.signed_credit * (0.9**atom.age) for atom in eligible],
        dtype=float,
    )
    weights /= weights.sum()
    return eligible[int(rng.choice(len(eligible), p=weights))]


def reset_memory(memory: MemoryState, *, event_id: int, reason: str) -> None:
    for atom in memory.bank:
        memory.atom_audit.append(
            {
                "sequence": len(memory.atom_audit) + 1,
                "action": "RESET_REMOVED",
                "atom": asdict(atom),
                "event_id": int(event_id),
                "reason": reason,
            }
        )
    memory.bank = []
    memory.tau = 0.5
    memory.negative_streak = 0
    memory.valid_feedback_count = 0
    memory.cooldown_remaining = 2
    memory.reset_count += 1
    memory.reset_log.append({"event_id": int(event_id), "reason": reason})


def apply_reset_gate(
    memory: MemoryState,
    *,
    event_id: int,
    variant: str,
    hard_reason: str | None = None,
) -> str | None:
    component = _component(variant)
    if hard_reason is not None:
        reset_memory(memory, event_id=event_id, reason=hard_reason)
        return hard_reason
    if component.soft_reset is False:
        return None
    if memory.negative_streak >= 3:
        reason = "three_consecutive_valid_negative_q"
        reset_memory(memory, event_id=event_id, reason=reason)
        return reason
    if memory.valid_feedback_count >= 5 and memory.tau < 0.25:
        reason = "tau_below_threshold"
        reset_memory(memory, event_id=event_id, reason=reason)
        return reason
    return None


def close_event_cooldown(memory: MemoryState) -> None:
    if memory.cooldown_remaining > 0:
        memory.cooldown_remaining -= 1


class StateMachine:
    def __init__(self, *, run_id: str, memory: MemoryState) -> None:
        if not run_id:
            raise StateIntegrityError("run_id must be nonempty")
        memory.validate()
        self.run_id = run_id
        self.memory = memory
        self.state = (
            "UNINITIALIZED" if memory.event_index < 0 else "EVENT_CLOSED"
        )
        self.event_id = memory.event_index
        self.logs: list[dict[str, Any]] = []
        self._sequence = 0

    def checksum(self) -> str:
        return _sha256_json(
            {
                "state": self.state,
                "event_id": self.event_id,
                "memory": self.memory.to_dict(),
            }
        )

    def transition(
        self,
        new_state: str,
        *,
        information_hash: str,
        ledger: EvaluationLedger,
    ) -> None:
        if (self.state, new_state) not in NORMAL_TRANSITIONS:
            self.state = "INVALID_STATE_INTEGRITY"
            raise StateIntegrityError(f"illegal state transition to {new_state}")
        if len(information_hash) != 64:
            self.state = "INVALID_STATE_INTEGRITY"
            raise StateIntegrityError("information hash is invalid")
        previous = self.state
        self.state = new_state
        self._sequence += 1
        self.logs.append(
            {
                "run_id": self.run_id,
                "event_id": self.event_id,
                "state_from": previous,
                "state_to": new_state,
                "timestamp": f"logical:{self.event_id}:{self._sequence}",
                "information_hash": information_hash,
                "budget_snapshot": ledger.snapshot(),
                "state_checksum": self.checksum(),
            }
        )

    def verify_last_checksum(self) -> None:
        if self.logs and self.logs[-1]["state_checksum"] != self.checksum():
            self.state = "INVALID_STATE_INTEGRITY"
            raise StateIntegrityError("state checksum mismatch")
