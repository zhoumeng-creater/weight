from __future__ import annotations

import ast
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACEABILITY = PROJECT_ROOT / "tests" / "f22_test_traceability.json"
REQUIRED_IDS = tuple(f"F22-T{index:02d}" for index in range(1, 16))


def test_f22_required_ids_map_to_existing_pytest_functions() -> None:
    payload = json.loads(TRACEABILITY.read_text(encoding="utf-8"))
    assert payload["contract_id"] == "F22"
    assert payload["overall_status"] == "PASS"
    assert payload["completion_claim"] == "F22_15_OF_15_VERIFIED"
    assert payload["contract_source"].endswith(
        "#162-f22-单元属性测试清单"
    )
    assert payload["required_test_ids"] == list(REQUIRED_IDS)
    assert tuple(payload["mappings"]) == REQUIRED_IDS
    t13 = payload["mappings"]["F22-T13"]
    assert t13["component_matrix_registration_status"] == "PASS"
    assert (
        t13[
            "shade_only_multiobjective_constrained_success_metric_binding"
        ]
        == "FROZEN_AND_IMPLEMENTED_WGT_SHADE_CMO_SUCCESS_01"
    )
    assert t13["status"] == "PASS"
    assert {
        "tests/test_v11_weight_adapter.py::"
        "test_zero_action_uses_prefrozen_positive_scale_at_next_event",
        "tests/test_v11_weight_adapter.py::"
        "test_tiny_action_keeps_prefrozen_scale_after_pending_restore",
    }.issubset(set(payload["mappings"]["F22-T04"]["tests"]))

    for test_id, entry in payload["mappings"].items():
        assert entry["contract_assertion"]
        assert entry["tests"], f"{test_id} has no bound pytest node"
        for node_id in entry["tests"]:
            relative_path, function_name = node_id.split("::", maxsplit=1)
            source_path = PROJECT_ROOT / relative_path
            assert source_path.is_file(), node_id
            tree = ast.parse(
                source_path.read_text(encoding="utf-8"),
                filename=str(source_path),
            )
            functions = {
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            assert function_name in functions, node_id
