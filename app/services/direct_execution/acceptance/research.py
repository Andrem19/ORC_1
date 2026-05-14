"""Research-memory acceptance verification."""

from __future__ import annotations

import logging
from typing import Any

from app.execution_models import PlanSlice, WorkerAction
from app.services.direct_execution.acceptance.contracts import FAIL, PASS, AcceptanceContract, PredicateResult, check_infra_route, extract_proof, is_infra_error, proof_refs, result_from_predicates
from app.services.direct_execution.acceptance.subjects import node_ids_from_action_and_transcript, successful_mutating_tool_count

logger = logging.getLogger(__name__)


async def verify_research(
    *,
    contract: AcceptanceContract,
    slice_obj: PlanSlice,
    action: WorkerAction,
    transcript: list[dict[str, Any]],
    proof_client: Any,
) -> Any:
    del slice_obj
    predicates: list[PredicateResult] = []
    proof_calls: list[dict[str, Any]] = []
    evidence_refs: list[str] = []
    if contract.kind in {"write_result", "research_shortlist_write"}:
        predicates.append(
            PredicateResult(
                id="mutating_tool_call_present",
                status=PASS if successful_mutating_tool_count(action) > 0 else FAIL,
                details={"mutating_tool_count": successful_mutating_tool_count(action)},
            )
        )
    if contract.kind == "research_setup":
        predicates.extend(_research_setup_fact_predicates(action))

    node_ids = node_ids_from_action_and_transcript(action, transcript)
    requires_node = contract.kind in {"write_result", "research_shortlist_write", "research_setup"}
    predicates.append(
        PredicateResult(
            id="research_node_subject_present",
            status=PASS if node_ids or not requires_node else FAIL,
            required=requires_node,
            details={"node_ids": node_ids},
        )
    )

    # For research_setup, node proofs are advisory — the critical gate is
    # research_project_present + research_setup_facts_present.  Individual
    # node proofs are required for write_result / shortlist writes where the
    # data quality of specific nodes matters.
    node_proof_required = contract.kind in {"write_result", "research_shortlist_write"}

    project_id = _project_id(action)
    logger.debug(
        "research acceptance: kind=%s node_ids=%s project_id=%s node_proof_required=%s",
        contract.kind,
        node_ids[:5],
        project_id,
        node_proof_required,
    )

    consecutive_infra = 0
    for node_id in node_ids[:10]:
        arguments = {
            "action": "prove",
            "node_id": node_id,
            "acceptance_requirements": dict(contract.acceptance_requirements),
        }
        if project_id:
            arguments["project_id"] = project_id
        try:
            payload = await proof_client.call_proof("research_memory", arguments)
        except Exception as exc:
            logger.warning(
                "research acceptance: prove call FAILED for node_id=%s project_id=%s: %s",
                node_id,
                project_id,
                exc,
            )
            predicates.append(
                PredicateResult(
                    id="research_node_proof_pass",
                    status=FAIL,
                    required=node_proof_required,
                    evidence_ref=f"mcp://research_memory/{node_id}#prove",
                    details={"node_id": node_id, "proof_status": "infra_error", "error": str(exc)},
                )
            )
            consecutive_infra += 1
            if consecutive_infra >= 2:
                break
            continue
        proof_calls.append({"tool": "research_memory", "arguments": arguments, "payload": payload})
        proof = extract_proof(payload)
        evidence_refs.extend(proof_refs(proof))
        infra = is_infra_error(proof)
        proof_status = proof["status"]
        logger.debug(
            "research acceptance: prove node_id=%s status=%s infra=%s error=%s",
            node_id,
            proof_status,
            infra,
            proof.get("error", "")[:200],
        )
        predicates.append(
            PredicateResult(
                id="research_node_proof_pass",
                status=PASS if proof_status == PASS else FAIL,
                required=node_proof_required,
                evidence_ref=f"mcp://research_memory/{node_id}#prove",
                details={
                    "node_id": node_id,
                    "proof_status": proof_status,
                    **({"infra_error": proof["error"]} if infra else {}),
                },
            )
        )
        if infra:
            route_hint, consecutive_infra = check_infra_route(predicates, consecutive_infra)
            if route_hint == "hard_block_infra":
                break
        else:
            consecutive_infra = 0
    infra_errors = [p for p in predicates if p.status == FAIL and "infra_error" in p.details]
    route = "hard_block_infra" if infra_errors else "fallback_allowed"
    return result_from_predicates(
        contract=contract,
        predicates=predicates,
        evidence_refs=evidence_refs,
        proof_calls=proof_calls,
        route=route,
    )


def _research_setup_fact_predicates(action: WorkerAction) -> list[PredicateResult]:
    facts = getattr(action, "facts", {}) or {}
    required = {
        "research_project_present": ("research.project_id", "project_id"),
        "research_setup_facts_present": (
            "research.baseline_configured",
            "research.atlas_defined",
            "research.invariants_recorded",
            "research.naming_recorded",
        ),
    }
    predicates: list[PredicateResult] = []
    for predicate_id, keys in required.items():
        if predicate_id == "research_setup_facts_present":
            missing = [key for key in keys if not _truthy(facts.get(key) if isinstance(facts, dict) else None)]
            predicates.append(PredicateResult(id=predicate_id, status=PASS if not missing else FAIL, details={"missing": missing}))
        else:
            present = any(_truthy(facts.get(key) if isinstance(facts, dict) else None) for key in keys)
            predicates.append(PredicateResult(id=predicate_id, status=PASS if present else FAIL))
    return predicates


def _project_id(action: WorkerAction) -> str:
    facts = getattr(action, "facts", {}) or {}
    if not isinstance(facts, dict):
        return ""
    return str(facts.get("research.project_id") or facts.get("project_id") or "").strip()


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


__all__ = ["verify_research"]
