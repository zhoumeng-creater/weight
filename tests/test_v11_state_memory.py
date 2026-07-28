from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
import pytest

from dt_ramde_v11.state import (
    COMPONENTS,
    LineageDAG,
    LineageNode,
    MGState,
    MemoryState,
    ParameterAtom,
    PendingCredit,
    StateIntegrityError,
    StateMachine,
    age_prune_bank,
    append_atoms,
    apply_reset_gate,
    close_event_cooldown,
    resolve_pending,
    sample_atom,
)
from evaluation.contracts import TerminalCode
from evaluation.ledger import EvaluationLedger


def _node(
    node_id: str,
    predecessor: str | None = None,
    *,
    survival: bool = True,
    f_value: float | None = 0.5,
    cr_value: float | None = 0.5,
) -> LineageNode:
    return LineageNode(
        node_id=node_id,
        event_id=0,
        generation=0,
        target_predecessor=predecessor,
        f=f_value,
        cr=cr_value,
        survival=survival,
    )


def _accepted_pending(pending_id: str = "p0") -> PendingCredit:
    return PendingCredit(
        pending_id=pending_id,
        source_event=0,
        terminal_code=TerminalCode.ACCEPTED,
        lineage_weights=(("n", 1.0),),
        parameter_values={"n": (0.4, 0.6)},
        information_hash="a" * 64,
        adapter_version="fixture-1",
    )


def test_mg_is_recreated_with_ten_half_slots() -> None:
    state = MGState.initialize()
    assert state.memory_f == [0.5] * 10
    assert state.memory_cr == [0.5] * 10
    assert state.pointer == 0


def test_pending_credit_is_consumed_once_at_the_next_event() -> None:
    pending = _accepted_pending()
    memory = MemoryState(pending_credit=pending)
    atoms, q_value, status = resolve_pending(
        memory,
        variant="FULL",
        feedback={
            "available": True,
            "ell_exec": 0.2,
            "ell_ref": 0.4,
            "s_exec": 0.5,
            "hard_constraint_violation": False,
        },
    )
    assert status == "EXECUTION_Q_RESOLVED"
    assert q_value == pytest.approx(0.4)
    assert len(atoms) == 1
    assert atoms[0].signed_credit == pytest.approx(0.55)
    assert memory.tau == pytest.approx(0.54)
    assert memory.pending_credit is None

    memory.pending_credit = pending
    with pytest.raises(StateIntegrityError, match="more than once"):
        resolve_pending(memory, variant="FULL", feedback=None)


def test_missing_or_disabled_execution_feedback_keeps_search_base_only() -> None:
    missing = MemoryState(pending_credit=_accepted_pending("missing"))
    atoms, q_value, status = resolve_pending(
        missing,
        variant="FULL",
        feedback=None,
    )
    assert status == "MISSING_EXPIRED"
    assert q_value is None
    assert atoms[0].signed_credit == pytest.approx(0.25)
    assert (missing.tau, missing.valid_feedback_count) == (0.5, 0)

    disabled = MemoryState(pending_credit=_accepted_pending("disabled"))
    atoms, q_value, status = resolve_pending(
        disabled,
        variant="NO_EXECUTION_FEEDBACK",
        feedback={
            "available": True,
            "ell_exec": -999.0,
            "ell_ref": 999.0,
            "s_exec": 1.0,
        },
    )
    assert status == "EXECUTION_Q_DISABLED"
    assert q_value is None
    assert atoms[0].signed_credit == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("code", "score"),
    [
        (TerminalCode.REJECT_SAFETY_FILTER, -1.0),
        (TerminalCode.REJECT_NO_FEASIBLE, -1.0),
        (TerminalCode.REJECT_BUDGET_NO_FEASIBLE, -0.75),
        (TerminalCode.REJECT_TIMEOUT, -0.75),
        (TerminalCode.REJECT_NUMERICAL, -1.0),
    ],
)
def test_rejection_credit_is_typed_and_ablatable(
    code: TerminalCode, score: float
) -> None:
    pending = replace(_accepted_pending(code.value), terminal_code=code)
    memory = MemoryState(pending_credit=pending)
    atoms, q_value, status = resolve_pending(
        memory, variant="FULL", feedback=None
    )
    assert (len(atoms), q_value, status) == (
        1,
        score,
        "REJECTION_Q_RESOLVED",
    )

    ablated = MemoryState(
        pending_credit=replace(_accepted_pending("ablated"), terminal_code=code)
    )
    assert resolve_pending(
        ablated, variant="NO_REJECTION_CREDIT", feedback=None
    ) == ([], None, "REJECTION_Q_DISABLED")
    assert (ablated.tau, ablated.negative_streak) == (0.5, 0)


def test_lineage_credit_uses_last_five_target_predecessors() -> None:
    dag = LineageDAG()
    predecessor = None
    for index in range(7):
        node_id = f"n{index}"
        dag.add(_node(node_id, predecessor))
        predecessor = node_id

    chain = dag.credit_chain("n6", mode="chain")
    assert [node_id for node_id, _weight in chain] == [
        "n6",
        "n5",
        "n4",
        "n3",
        "n2",
    ]
    assert sum(weight for _node_id, weight in chain) == pytest.approx(1.0)
    expected = np.asarray([1.0, 0.5, 0.25, 0.125, 0.0625])
    expected /= expected.sum()
    np.testing.assert_allclose([weight for _node_id, weight in chain], expected)

    uniform = dag.credit_chain("n6", mode="event_last_5_uniform")
    assert all(weight == pytest.approx(0.2) for _node_id, weight in uniform)


def test_lineage_records_match_recursive_dataclass_serialization() -> None:
    node = replace(
        _node("n0"),
        target_id="target",
        pbest_id="pbest",
        r1_id="r1",
        r2_id="r2",
        parameter_source="M_g:0",
        j_rand=2,
        pre_repair_hash="a" * 64,
        post_repair_hash="b" * 64,
        repaired=True,
        objectives=(1.0, 2.0),
        constraints=(-1.0,),
        feasible=True,
        normalized_cv=0.0,
        archive_admission=True,
    )
    dag = LineageDAG()
    dag.add(node)

    assert dag.records() == (asdict(node),)


def test_reset_gate_and_two_closed_event_cooldown() -> None:
    memory = MemoryState(
        bank=[ParameterAtom(0.5, 0.5, 0.5, 0, "n")],
        negative_streak=3,
    )
    assert apply_reset_gate(memory, event_id=1, variant="FULL") == (
        "three_consecutive_valid_negative_q"
    )
    assert memory.bank == []
    assert (memory.tau, memory.cooldown_remaining) == (0.5, 2)
    assert memory.transfer_allowed is False

    close_event_cooldown(memory)
    assert memory.cooldown_remaining == 1
    close_event_cooldown(memory)
    assert memory.cooldown_remaining == 0
    assert memory.transfer_allowed is True


def test_reset_gate_covers_tau_boundary_hard_reset_and_ablation() -> None:
    threshold = MemoryState(
        bank=[ParameterAtom(0.5, 0.5, 0.5, 0, "threshold")],
        tau=0.249999,
        valid_feedback_count=5,
    )
    assert apply_reset_gate(
        threshold,
        event_id=5,
        variant="FULL",
    ) == "tau_below_threshold"
    assert threshold.bank == []
    assert (
        threshold.tau,
        threshold.negative_streak,
        threshold.valid_feedback_count,
        threshold.cooldown_remaining,
        threshold.reset_count,
    ) == (0.5, 0, 0, 2, 1)

    exact_boundary = MemoryState(tau=0.25, valid_feedback_count=5)
    assert apply_reset_gate(
        exact_boundary,
        event_id=5,
        variant="FULL",
    ) is None

    ablated_soft = MemoryState(negative_streak=3)
    assert apply_reset_gate(
        ablated_soft,
        event_id=2,
        variant="NO_MEMORY_RESET_GATE",
    ) is None

    hard = MemoryState(
        bank=[ParameterAtom(0.5, 0.5, 0.5, 0, "hard")]
    )
    assert apply_reset_gate(
        hard,
        event_id=2,
        variant="NO_MEMORY_RESET_GATE",
        hard_reason="prior_numerical_failure",
    ) == "prior_numerical_failure"
    assert hard.bank == []
    assert hard.cooldown_remaining == 2


def test_bank_age_capacity_and_positive_sampling() -> None:
    memory = MemoryState()
    atoms = [
        ParameterAtom(
            f=0.5,
            cr=0.5,
            signed_credit=(1 if index % 2 == 0 else -1) * 0.1,
            source_event=index,
            lineage_node_id=f"n{index}",
        )
        for index in range(25)
    ]
    new_keys = append_atoms(memory, atoms)
    age_prune_bank(memory, newly_added_keys=new_keys)
    assert len(memory.bank) == 20
    assert memory.bank[0].source_event == 5
    rng = np.random.default_rng(5)
    assert all(sample_atom(memory, rng).signed_credit > 0 for _ in range(25))
    assert sum(
        item["action"] == "CREATED" for item in memory.atom_audit
    ) == 25
    assert sum(
        item["action"] == "CAPACITY_EVICTED" for item in memory.atom_audit
    ) == 5

    for _event in range(6):
        age_prune_bank(memory, newly_added_keys=set())
    assert memory.bank == []
    assert sum(
        item["action"] == "EXPIRED" for item in memory.atom_audit
    ) == 20


def test_bank_sampling_uses_positive_credit_times_age_decay() -> None:
    memory = MemoryState(
        bank=[
            ParameterAtom(0.4, 0.4, 1.0, 0, "fresh", age=0),
            ParameterAtom(0.6, 0.6, 2.0, 0, "older", age=2),
            ParameterAtom(0.8, 0.8, -1.0, 0, "negative", age=0),
        ]
    )

    class RecordingRng:
        probabilities: np.ndarray | None = None

        def choice(
            self,
            _count: int,
            *,
            p: np.ndarray,
        ) -> int:
            self.probabilities = p.copy()
            return 1

    rng = RecordingRng()
    selected = sample_atom(memory, rng)  # type: ignore[arg-type]
    expected = np.asarray([1.0, 2.0 * (0.9**2)])
    expected /= expected.sum()
    np.testing.assert_allclose(rng.probabilities, expected)
    assert selected is memory.bank[1]


def test_state_serialization_and_checksum_detect_resume_drift() -> None:
    memory = MemoryState(
        bank=[ParameterAtom(0.4, 0.6, 0.25, 0, "n")],
        solution_memory=((0.1, 0.2),),
        pending_credit=_accepted_pending(),
    )
    restored = MemoryState.from_dict(memory.to_dict())
    assert restored.to_dict() == memory.to_dict()
    tampered = memory.to_dict()
    tampered["tau"] = 0.6
    with pytest.raises(StateIntegrityError, match="checkpoint checksum"):
        MemoryState.from_dict(tampered)

    ledger = EvaluationLedger(max_cfe=1)
    machine = StateMachine(run_id="run-1", memory=restored)
    machine.event_id = 0
    machine.transition(
        "EVENT_OPEN", information_hash="b" * 64, ledger=ledger
    )
    restored.tau = 0.6
    with pytest.raises(StateIntegrityError, match="checksum"):
        machine.verify_last_checksum()


def test_resume_validation_rejects_a_pending_id_already_marked_consumed() -> None:
    memory = MemoryState(
        pending_credit=_accepted_pending("already-consumed"),
        consumed_pending_ids=["already-consumed"],
    )
    with pytest.raises(StateIntegrityError, match="already consumed"):
        memory.validate()


def test_lineage_rejects_invalid_parameter_values_transactionally() -> None:
    dag = LineageDAG()
    with pytest.raises(StateIntegrityError, match="F/CR"):
        dag.add(_node("bad", f_value=0.0, cr_value=1.5))
    assert dag.nodes == {}


def test_invalid_feedback_does_not_consume_pending_credit() -> None:
    pending = _accepted_pending("invalid-feedback")
    memory = MemoryState(pending_credit=pending)
    with pytest.raises(StateIntegrityError, match="feedback"):
        resolve_pending(
            memory,
            variant="FULL",
            feedback={
                "available": True,
                "ell_exec": 1.0,
                "ell_ref": 2.0,
                "s_exec": 0.0,
            },
        )
    assert memory.pending_credit is pending
    assert memory.consumed_pending_ids == []


def test_illegal_transition_invalidates_state_and_ts1_rejects_memory() -> None:
    machine = StateMachine(run_id="run-1", memory=MemoryState())
    with pytest.raises(StateIntegrityError, match="illegal"):
        machine.transition(
            "SEARCHING",
            information_hash="b" * 64,
            ledger=EvaluationLedger(max_cfe=1),
        )
    assert machine.state == "INVALID_STATE_INTEGRITY"

    with pytest.raises(StateIntegrityError, match="TS1"):
        MemoryState(bank=[ParameterAtom(0.5, 0.5, 0.1, 0, "n")]).validate_timing(
            "TS1_single_event"
        )
    MemoryState().validate_timing("TS1_single_event")


def test_registered_component_matrix_is_exact_and_single_factor_named() -> None:
    expected = {
        "FULL": {
            "mg_mode": "F22_weighted_survivor",
            "parameter_memory": True,
            "warm_start": True,
            "execution_credit": True,
            "rejection_credit": True,
            "lineage_mode": "chain",
            "soft_reset": True,
        },
        "NO_CROSS_EVENT_MEMORY": {
            "mg_mode": "F22_weighted_survivor",
            "parameter_memory": False,
            "warm_start": False,
            "execution_credit": False,
            "rejection_credit": False,
            "lineage_mode": "off",
            "soft_reset": False,
        },
        "NO_EXECUTION_FEEDBACK": {
            "mg_mode": "F22_weighted_survivor",
            "parameter_memory": True,
            "warm_start": True,
            "execution_credit": False,
            "rejection_credit": True,
            "lineage_mode": "chain",
            "soft_reset": "rejection_only",
        },
        "NO_REJECTION_CREDIT": {
            "mg_mode": "F22_weighted_survivor",
            "parameter_memory": True,
            "warm_start": True,
            "execution_credit": True,
            "rejection_credit": False,
            "lineage_mode": "chain",
            "soft_reset": "execution_only",
        },
        "NO_MEMORY_RESET_GATE": {
            "mg_mode": "F22_weighted_survivor",
            "parameter_memory": True,
            "warm_start": True,
            "execution_credit": True,
            "rejection_credit": True,
            "lineage_mode": "chain",
            "soft_reset": False,
        },
        "NO_LINEAGE_CREDIT": {
            "mg_mode": "F22_weighted_survivor",
            "parameter_memory": True,
            "warm_start": True,
            "execution_credit": True,
            "rejection_credit": True,
            "lineage_mode": "event_last_5_uniform",
            "soft_reset": True,
        },
        "CROSS_EVENT_WARM_START_ONLY": {
            "mg_mode": "F22_weighted_survivor",
            "parameter_memory": False,
            "warm_start": True,
            "execution_credit": False,
            "rejection_credit": False,
            "lineage_mode": "off",
            "soft_reset": False,
        },
        "CROSS_EVENT_MEMORY_ONLY": {
            "mg_mode": "F22_weighted_survivor",
            "parameter_memory": True,
            "warm_start": False,
            "execution_credit": True,
            "rejection_credit": True,
            "lineage_mode": "chain",
            "soft_reset": True,
        },
        "SHADE_ONLY": {
            "mg_mode": "WGT_SHADE_CMO_SUCCESS_01",
            "parameter_memory": False,
            "warm_start": False,
            "execution_credit": False,
            "rejection_credit": False,
            "lineage_mode": "off",
            "soft_reset": False,
        },
    }
    actual = {
        name: component.to_dict()
        for name, component in COMPONENTS.items()
    }
    assert actual == expected
