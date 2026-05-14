"""Feature-domain acceptance verification."""

from __future__ import annotations

from typing import Any

from app.execution_models import PlanSlice, WorkerAction
from app.services.direct_execution.acceptance.contracts import FAIL, PASS, AcceptanceContract, PredicateResult, check_infra_route, extract_proof, is_infra_error, proof_refs, result_from_predicates
from app.services.direct_execution.acceptance.subjects import feature_names_from_action_and_transcript, successful_tool_names


async def verify_features(
    *,
    contract: AcceptanceContract,
    slice_obj: PlanSlice,
    action: WorkerAction,
    transcript: list[dict[str, Any]],
    proof_client: Any,
) -> Any:
    feature_names = feature_names_from_action_and_transcript(action, transcript)
    predicates: list[PredicateResult] = []
    proof_calls: list[dict[str, Any]] = []
    evidence_refs: list[str] = []
    if contract.kind == "feature_profitability_filter":
        facts = getattr(action, "facts", {}) or {}
        has_decisions = bool(
            isinstance(facts, dict)
            and (facts.get("selected_candidates") or facts.get("rejected_candidates") or facts.get("research.shortlist_families"))
        )
        predicates.append(PredicateResult(id="candidate_decision_set_present", status=PASS if has_decisions else FAIL))
        predicates.append(
            PredicateResult(
                id="domain_tool_evidence_present",
                status=PASS if successful_tool_names(action) & {"features_custom", "features_dataset", "features_catalog", "features_analytics", "events", "datasets"} else FAIL,
            )
        )
        return result_from_predicates(contract=contract, predicates=predicates)

    predicates.append(
        PredicateResult(
            id="feature_subject_present",
            status=PASS if feature_names or contract.kind == "feature_contract_exploration" else FAIL,
            details={"feature_names": feature_names},
        )
    )
    if contract.kind == "feature_contract_exploration":
        predicates.append(
            PredicateResult(
                id="domain_tool_evidence_present",
                status=PASS if successful_tool_names(action) & {"features_custom", "features_dataset", "features_catalog", "events", "datasets"} else FAIL,
            )
        )
        return result_from_predicates(contract=contract, predicates=predicates)

    consecutive_infra = 0
    for name in feature_names[:10]:
        arguments = {
            "action": "inspect",
            "view": "acceptance_proof",
            "name": name,
            "acceptance_requirements": dict(contract.acceptance_requirements),
        }
        payload = await proof_client.call_proof("features_custom", arguments)
        proof_calls.append({"tool": "features_custom", "arguments": arguments, "payload": payload})
        proof = extract_proof(payload)
        evidence_refs.extend(proof_refs(proof))
        infra = is_infra_error(proof)
        predicates.append(
            PredicateResult(
                id="feature_proof_pass",
                status=PASS if proof["status"] == PASS else FAIL,
                evidence_ref=f"mcp://features_custom/{name}#acceptance_proof",
                details={
                    "feature_name": name,
                    "proof_status": proof["status"],
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
    if bool(contract.acceptance_requirements.get("require_dataset_activation")):
        dataset_arguments = _dataset_arguments(action)
        if dataset_arguments is None:
            predicates.append(PredicateResult(id="dataset_activation_subject_present", status=FAIL))
        else:
            dataset_arguments["acceptance_requirements"] = dict(contract.acceptance_requirements)
            payload = await proof_client.call_proof("features_dataset", dataset_arguments)
            proof_calls.append({"tool": "features_dataset", "arguments": dataset_arguments, "payload": payload})
            proof = extract_proof(payload)
            infra = is_infra_error(proof)
            predicates.append(
                PredicateResult(
                    id="dataset_activation_proof_pass",
                    status=PASS if proof["status"] == PASS else FAIL,
                    details={
                        "proof_status": proof["status"],
                        **({"infra_error": proof["error"]} if infra else {}),
                    },
                )
            )
            evidence_refs.extend(proof_refs(proof))
    infra_errors = [p for p in predicates if p.status == FAIL and "infra_error" in p.details]
    route = "hard_block_infra" if infra_errors else "fallback_allowed"
    return result_from_predicates(
        contract=contract,
        predicates=predicates,
        evidence_refs=evidence_refs,
        proof_calls=proof_calls,
        route=route,
    )


def _dataset_arguments(action: WorkerAction) -> dict[str, Any] | None:
    facts = getattr(action, "facts", {}) or {}
    if not isinstance(facts, dict):
        return None
    dataset_id = str(facts.get("dataset_id") or "").strip()
    if dataset_id:
        return {"action": "inspect", "view": "acceptance_proof", "dataset_id": dataset_id}
    symbol = str(facts.get("symbol") or facts.get("research.symbol") or "").strip()
    timeframe = str(facts.get("timeframe") or facts.get("anchor_timeframe") or "").strip()
    if symbol and timeframe:
        return {"action": "inspect", "view": "acceptance_proof", "symbol": symbol, "timeframe": timeframe}
    return None


__all__ = ["verify_features"]
