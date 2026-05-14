"""Async acceptance verifier orchestrating domain proof checks."""

from __future__ import annotations

from typing import Any, Callable

from app.execution_models import ExecutionPlan, PlanSlice, WorkerAction
from app.services.direct_execution.acceptance.backtests import verify_backtests
from app.services.direct_execution.acceptance.builder import build_acceptance_contract
from app.services.direct_execution.acceptance.contracts import FAIL, PASS, AcceptanceResult, PredicateResult, fail_result, result_from_predicates
from app.services.direct_execution.acceptance.contradictions import contradiction_predicates
from app.services.direct_execution.acceptance.features import verify_features
from app.services.direct_execution.acceptance.models import verify_models
from app.services.direct_execution.acceptance.proof_client import AcceptanceProofInfraError, ProofClient, build_proof_client_config
from app.services.direct_execution.acceptance.research import verify_research
from app.services.direct_execution.acceptance.subjects import successful_tool_names

ProofClientFactory = Callable[[], Any]


class AcceptanceVerifier:
    def __init__(
        self,
        *,
        direct_config: Any,
        incident_store: Any | None = None,
        proof_client_factory: ProofClientFactory | None = None,
    ) -> None:
        self.direct_config = direct_config
        self.incident_store = incident_store
        self.proof_client_factory = proof_client_factory

    async def verify(
        self,
        *,
        plan: ExecutionPlan,
        slice_obj: PlanSlice,
        action: WorkerAction,
        transcript: list[dict[str, Any]],
        known_facts: dict[str, Any] | None = None,
        required_output_facts: list[str] | None = None,
    ) -> AcceptanceResult:
        del plan, known_facts, required_output_facts
        contract = build_acceptance_contract(slice_obj)
        slice_obj.acceptance_contract = contract.to_dict()
        contradiction_results = contradiction_predicates(action, contract.forbid_contradictions)
        hard_contradictions = [item for item in contradiction_results if item.status == FAIL]
        if hard_contradictions:
            return result_from_predicates(
                contract=contract,
                predicates=hard_contradictions,
                route="repair_only",
            )
        if contract.kind == "generic_read_advisory":
            return result_from_predicates(
                contract=contract,
                predicates=[PredicateResult(id="advisory_contract", status=PASS, required=False), *contradiction_results],
                route="accepted",
            )
        if contract.kind == "generic_read_strict":
            return result_from_predicates(
                contract=contract,
                predicates=[
                    PredicateResult(
                        id="domain_tool_evidence_present",
                        status=PASS if successful_tool_names(action) else FAIL,
                        details={"successful_tool_names": sorted(successful_tool_names(action))},
                    ),
                    *contradiction_results,
                ],
            )
        try:
            async with self._proof_client() as proof_client:
                if contract.kind in {"standalone_backtests", "integration_backtest", "condition_stability_analysis"}:
                    result = await verify_backtests(contract=contract, slice_obj=slice_obj, action=action, transcript=transcript, proof_client=proof_client)
                elif contract.kind in {"write_result", "research_shortlist_write", "research_setup"}:
                    result = await verify_research(contract=contract, slice_obj=slice_obj, action=action, transcript=transcript, proof_client=proof_client)
                elif contract.kind in {"feature_contract_exploration", "feature_contract_construction", "feature_profitability_filter"}:
                    result = await verify_features(contract=contract, slice_obj=slice_obj, action=action, transcript=transcript, proof_client=proof_client)
                elif contract.kind == "model_training":
                    result = await verify_models(contract=contract, slice_obj=slice_obj, action=action, transcript=transcript, proof_client=proof_client)
                else:
                    result = fail_result(contract=contract, reason="acceptance_contract_unknown", route="repair_only", details={"kind": contract.kind})
        except AcceptanceProofInfraError as exc:
            self._record_incident(slice_obj=slice_obj, reason="acceptance_mcp_infra_unavailable", details={"error": str(exc)})
            return fail_result(
                contract=contract,
                reason="acceptance_mcp_infra_unavailable",
                route="hard_block_infra",
                details={"error": str(exc)},
            )
        except Exception as exc:
            self._record_incident(slice_obj=slice_obj, reason="acceptance_verifier_exception", details={"error": str(exc)})
            return fail_result(
                contract=contract,
                reason="acceptance_verifier_exception",
                route="repair_only",
                details={"error": str(exc)},
            )
        if contradiction_results:
            predicates = list(result.predicates) + [item for item in contradiction_results if item.id != "forbidden_contradictions_absent"]
            if predicates != result.predicates:
                result = result_from_predicates(
                    contract=contract,
                    predicates=predicates,
                    warnings=result.warnings,
                    evidence_refs=result.evidence_refs,
                    proof_calls=result.proof_calls,
                    incidents=result.incidents,
                    route=result.route,
                )
        if result.route == "hard_block_infra":
            infra_predicates = [p for p in result.predicates if p.status == FAIL and "infra_error" in p.details]
            if infra_predicates:
                self._record_incident(
                    slice_obj=slice_obj,
                    reason="acceptance_inner_proof_infra_error",
                    details={
                        "infra_errors": [p.details.get("infra_error", "") for p in infra_predicates[:3]],
                        "failed_predicates": [p.id for p in infra_predicates[:3]],
                    },
                )
        return result

    def _proof_client(self) -> Any:
        if self.proof_client_factory is not None:
            return self.proof_client_factory()
        return ProofClient(build_proof_client_config(self.direct_config))

    def _record_incident(self, *, slice_obj: PlanSlice, reason: str, details: dict[str, Any]) -> None:
        if self.incident_store is None:
            return
        self.incident_store.record(
            summary="Acceptance verifier incident",
            metadata={
                "slice_id": slice_obj.slice_id,
                "reason": reason,
                **details,
            },
            source="direct_acceptance",
            severity="medium",
        )


__all__ = ["AcceptanceVerifier"]
