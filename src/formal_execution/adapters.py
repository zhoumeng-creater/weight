"""R8-only wrappers for the R4 public benchmark implementations.

The R4 classes remain historically truthful (`NO_EFFECT`).  These wrappers add
the narrow R7/R8 execution authority while preserving their exact evaluators.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any

from benchmark_adapters.cdf_operational import (
    CDFOperationalEvaluator,
    CDF_OPERATIONAL_SUITE_ID,
)
from benchmark_adapters.lircmop_paper import (
    LIRCMOP_PAPER_SUITE_ID,
    LIRCMOPPaperEvaluator,
)
from benchmark_adapters.r4_evaluators import CDFEvaluator, LIRCMOPEvaluator
from benchmark_adapters.r4_public import (
    R4CDFPublicAdapter,
    R4StaticPublicAdapter,
)
from benchmark_adapters.r4_wgt_rr import (
    WGTRRBindingError,
    WGTRRPublicAdapter,
)
from weight_application.formal_e3_adapter import FormalHallE3Adapter

from .public_rolling import generate_public_instance


R8_EXECUTION_AUTHORITY = (
    "R7_CONTRACT_PLUS_VERBATIM_CONFIRMED_R8_REQUEST_ONLY"
)
R8C_EXECUTION_AUTHORITY = (
    "R8C_CORRECTIVE_CONTRACT_PLUS_SEPARATE_VERBATIM_REQUEST_ONLY"
)


class FormalR8StaticAdapter(R4StaticPublicAdapter):
    adapter_id = "BIND-STATIC-CMOP-01/R8-FORMAL-PUBLIC"
    adapter_version = "1.2.0-r8-formal"
    bridge_role = "r8_frozen_formal_public_benchmark"
    bridge_stage = "R8"
    execution_authority = R8_EXECUTION_AUTHORITY
    registered_effect_instance = True
    formal_effect_execution_allowed = True

    def identity(self) -> Mapping[str, Any]:
        return {
            **super().identity(),
            "split": "public_fixed_formal",
            "registered_effect_instance": True,
            "formal_effect_execution_allowed": True,
            "execution_authority": self.execution_authority,
        }


class FormalR8CDFAdapter(R4CDFPublicAdapter):
    adapter_id = "BIND-CDF-DYNAMIC-01/R8-FORMAL-PUBLIC"
    adapter_version = "1.2.0-r8-formal"
    bridge_role = "r8_frozen_formal_public_benchmark"
    bridge_stage = "R8"
    execution_authority = R8_EXECUTION_AUTHORITY
    registered_effect_instance = True
    formal_effect_execution_allowed = True

    def identity(self) -> Mapping[str, Any]:
        return {
            **super().identity(),
            "split": "public_fixed_formal",
            "registered_effect_instance": True,
            "formal_effect_execution_allowed": True,
            "execution_authority": self.execution_authority,
        }


class FormalR8WGTRRAdapter(WGTRRPublicAdapter):
    adapter_id = "BIND-ROLLING-01/R8-FORMAL-PUBLIC"
    adapter_version = "1.2.0-r8-formal"

    @staticmethod
    def _validate_instance(instance: Mapping[str, Any]) -> None:
        template = instance.get("template")
        index = instance.get("index")
        if not isinstance(template, str) or type(index) is not int:
            raise WGTRRBindingError(
                "formal rolling instance lacks template/index"
            )
        expected = generate_public_instance(template, index)
        def canonical(value: Any) -> str:
            return json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        if canonical(instance) != canonical(expected):
            raise WGTRRBindingError(
                "formal rolling instance differs from frozen public generator"
            )

    def identity(self) -> Mapping[str, Any]:
        return {
            **super().identity(),
            "split": "public_development_formal",
            "registered_effect_instance": True,
            "formal_effect_execution_allowed": True,
            "execution_authority": R8_EXECUTION_AUTHORITY,
        }


class FormalR8CStaticAdapter(FormalR8StaticAdapter):
    adapter_id = "BIND-STATIC-CMOP-01/R8C-CORRECTIVE-FORMAL-PUBLIC"
    adapter_version = "1.3.0-r8c-corrective"
    bridge_role = "r8c_corrective_formal_public_benchmark"
    bridge_stage = "R8C"
    execution_authority = R8C_EXECUTION_AUTHORITY


class FormalR8CCDFAdapter(FormalR8CDFAdapter):
    adapter_id = "BIND-CDF-DYNAMIC-01/R8C-CORRECTIVE-FORMAL-PUBLIC"
    adapter_version = "1.3.0-r8c-corrective"
    bridge_role = "r8c_corrective_formal_public_benchmark"
    bridge_stage = "R8C"
    execution_authority = R8C_EXECUTION_AUTHORITY


class FormalR8CWGTRRAdapter(FormalR8WGTRRAdapter):
    adapter_id = "BIND-ROLLING-01/R8C-CORRECTIVE-FORMAL-PUBLIC"
    adapter_version = "1.3.0-r8c-corrective"

    def identity(self) -> Mapping[str, Any]:
        return {
            **super().identity(),
            "split": "public_development_r8c_corrective_formal",
            "execution_authority": R8C_EXECUTION_AUTHORITY,
        }


class FormalR8CHallE3Adapter(FormalHallE3Adapter):
    adapter_id = "WGT-V11-R8C-CORRECTIVE-FORMAL-PUBLIC-HALL-E3"
    adapter_version = "1.1.0-r8c-corrective"

    def identity(self) -> Mapping[str, Any]:
        return {
            **super().identity(),
            "role": "r8c_corrective_formal_public_synthetic_e3_benchmark",
            "execution_authority": R8C_EXECUTION_AUTHORITY,
        }


def make_formal_lircmop_adapter(
    problem_index: int,
) -> FormalR8StaticAdapter:
    evaluator = LIRCMOPEvaluator(problem_index)
    return FormalR8StaticAdapter(
        suite_id="LIR-CMOP-JMETALPY-1.7.0",
        problem_id=evaluator.problem_id,
        evaluator_version="STATIC-CMOP-EVAL-1.0.0",
        fixture_evaluator_sha256=evaluator.binding_sha256,
        lower=evaluator.lower_bounds,
        upper=evaluator.upper_bounds,
        objective_names=evaluator.objective_names,
        constraint_names=evaluator.constraint_names,
        evaluator=evaluator,
    )


def make_formal_cdf_adapter(
    problem_index: int,
    *,
    profile: str,
    environment_seed: int,
) -> FormalR8CDFAdapter:
    evaluator = CDFEvaluator(
        problem_index=problem_index,
        profile=profile,
        environment_seed=environment_seed,
    )
    return FormalR8CDFAdapter(
        suite_id="CDF-1-15",
        problem_id=evaluator.problem_id,
        profile=profile,
        evaluator_version="CDF-EVAL-1.0.0",
        fixture_evaluator_sha256=evaluator.binding_sha256,
        lower=evaluator.lower_bounds,
        upper=evaluator.upper_bounds,
        objective_names=evaluator.objective_names,
        constraint_names=evaluator.constraint_names,
        evaluator=evaluator,
        release_metadata=evaluator.release_metadata,
    )


def make_formal_wgt_rr_adapter(
    template: str,
    index: int,
) -> FormalR8WGTRRAdapter:
    return FormalR8WGTRRAdapter(generate_public_instance(template, index))


def make_corrective_lircmop_adapter(
    problem_index: int,
) -> FormalR8CStaticAdapter:
    evaluator = LIRCMOPPaperEvaluator(problem_index)
    return FormalR8CStaticAdapter(
        suite_id=LIRCMOP_PAPER_SUITE_ID,
        problem_id=evaluator.problem_id,
        evaluator_version="STATIC-CMOP-EVAL-1.0.0",
        fixture_evaluator_sha256=evaluator.binding_sha256,
        lower=evaluator.lower_bounds,
        upper=evaluator.upper_bounds,
        objective_names=evaluator.objective_names,
        constraint_names=evaluator.constraint_names,
        evaluator=evaluator,
    )


def make_corrective_cdf_adapter(
    problem_index: int,
    *,
    profile: str,
    environment_seed: int,
) -> FormalR8CCDFAdapter:
    evaluator = CDFOperationalEvaluator(
        problem_index=problem_index,
        profile=profile,
        environment_seed=environment_seed,
    )
    return FormalR8CCDFAdapter(
        suite_id=CDF_OPERATIONAL_SUITE_ID,
        problem_id=evaluator.problem_id,
        profile=profile,
        evaluator_version="CDF-EVAL-1.0.0",
        fixture_evaluator_sha256=evaluator.binding_sha256,
        lower=evaluator.lower_bounds,
        upper=evaluator.upper_bounds,
        objective_names=evaluator.objective_names,
        constraint_names=evaluator.constraint_names,
        evaluator=evaluator,
        release_metadata=evaluator.release_metadata,
    )


def make_corrective_wgt_rr_adapter(
    template: str,
    index: int,
) -> FormalR8CWGTRRAdapter:
    return FormalR8CWGTRRAdapter(
        generate_public_instance(template, index)
    )
