"""Model-domain acceptance verification."""

from __future__ import annotations

from typing import Any

from app.execution_models import PlanSlice, WorkerAction
from app.services.direct_execution.acceptance.contracts import FAIL, PASS, AcceptanceContract, PredicateResult, extract_proof, is_infra_error, proof_refs, result_from_predicates
from app.services.direct_execution.acceptance.subjects import model_refs_from_action_and_transcript


async def verify_models(
    *,
    contract: AcceptanceContract,
    slice_obj: PlanSlice,
    action: WorkerAction,
    transcript: list[dict[str, Any]],
    proof_client: Any,
) -> Any:
    del slice_obj
    dataset_ids, versions = model_refs_from_action_and_transcript(action, transcript)
    predicates: list[PredicateResult] = [
        PredicateResult(id="model_dataset_subject_present", status=PASS if dataset_ids else FAIL, details={"dataset_ids": dataset_ids}),
        PredicateResult(id="model_version_subject_present", status=PASS if versions else FAIL, details={"versions": versions}),
    ]
    proof_calls: list[dict[str, Any]] = []
    evidence_refs: list[str] = []
    for dataset_id in dataset_ids[:5]:
        arguments = {
            "action": "inspect",
            "view": "acceptance_proof",
            "dataset_id": dataset_id,
            "acceptance_requirements": dict(contract.acceptance_requirements),
        }
        payload = await proof_client.call_proof("models_dataset", arguments)
        proof_calls.append({"tool": "models_dataset", "arguments": arguments, "payload": payload})
        proof = extract_proof(payload)
        infra = is_infra_error(proof)
        predicates.append(
            PredicateResult(
                id="model_dataset_proof_pass",
                status=PASS if proof["status"] == PASS else FAIL,
                details={
                    "dataset_id": dataset_id,
                    "proof_status": proof["status"],
                    **({"infra_error": proof["error"]} if infra else {}),
                },
            )
        )
        evidence_refs.extend(proof_refs(proof))
    for model_id, version in versions[:5]:
        arguments = {
            "action": "inspect",
            "view": "acceptance_proof",
            "model_id": model_id,
            "version": version,
            "acceptance_requirements": dict(contract.acceptance_requirements),
        }
        payload = await proof_client.call_proof("models_registry", arguments)
        proof_calls.append({"tool": "models_registry", "arguments": arguments, "payload": payload})
        proof = extract_proof(payload)
        infra = is_infra_error(proof)
        predicates.append(
            PredicateResult(
                id="model_registry_proof_pass",
                status=PASS if proof["status"] == PASS else FAIL,
                details={
                    "model_id": model_id,
                    "version": version,
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


__all__ = ["verify_models"]
