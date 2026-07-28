"""R4 executable public-benchmark bindings without effect authority.

These classes upgrade the R2 bridge identity from a caller-supplied fixture to
an exact, source-bound evaluator.  They deliberately keep formal effect
execution closed: R4 proves that the bridge is executable and fair; R5/R7 must
still freeze numerical execution contracts and authorize a run.
"""

from __future__ import annotations

from .public_cmop import CDFPublicAdapter, StaticCMOPPublicAdapter
from .r4_evaluators import CDFEvaluator, LIRCMOPEvaluator


class R4StaticPublicAdapter(StaticCMOPPublicAdapter):
    """Executable static constrained-MO binding frozen at R4."""

    adapter_id = "BIND-STATIC-CMOP-01/R4-EXECUTABLE"
    adapter_version = "1.1.0-r4-binding"
    bridge_role = "r4_exact_public_evaluator_binding"
    bridge_stage = "R4"
    execution_authority = "R4_BINDING_ONLY_NO_EFFECT"
    registered_benchmark_evaluator = True

    def identity(self):
        return {
            **super().identity(),
            "bridge_stage": self.bridge_stage,
            "registered_benchmark_evaluator": (
                self.registered_benchmark_evaluator
            ),
            "execution_authority": self.execution_authority,
        }


class R4CDFPublicAdapter(CDFPublicAdapter):
    """Executable CDF1--15 binding frozen at R4."""

    adapter_id = "BIND-CDF-DYNAMIC-01/R4-EXECUTABLE"
    adapter_version = "1.1.0-r4-binding"
    bridge_role = "r4_exact_public_evaluator_binding"
    bridge_stage = "R4"
    execution_authority = "R4_BINDING_ONLY_NO_EFFECT"
    registered_benchmark_evaluator = True

    def identity(self):
        return {
            **super().identity(),
            "bridge_stage": self.bridge_stage,
            "registered_benchmark_evaluator": (
                self.registered_benchmark_evaluator
            ),
            "execution_authority": self.execution_authority,
        }


def make_r4_lircmop_adapter(problem_index: int) -> R4StaticPublicAdapter:
    """Construct an exact LIR-CMOP1--14 R4 adapter."""

    evaluator = LIRCMOPEvaluator(problem_index)
    return R4StaticPublicAdapter(
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


def make_r4_cdf_adapter(
    problem_index: int,
    *,
    profile: str,
    environment_seed: int = 0,
) -> R4CDFPublicAdapter:
    """Construct an exact CDF1--15 R4 adapter."""

    evaluator = CDFEvaluator(
        problem_index=problem_index,
        profile=profile,
        environment_seed=environment_seed,
    )
    return R4CDFPublicAdapter(
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
