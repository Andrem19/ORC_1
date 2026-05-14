"""Backtest acceptance verification."""

from __future__ import annotations

from typing import Any

from app.execution_models import PlanSlice, WorkerAction
from app.services.direct_execution.acceptance.contracts import FAIL, PASS, AcceptanceContract, PredicateResult, check_infra_route, extract_proof, is_infra_error, proof_refs, result_from_predicates
from app.services.direct_execution.acceptance.subjects import run_ids_from_action_and_transcript


async def verify_backtests(
    *,
    contract: AcceptanceContract,
    slice_obj: PlanSlice,
    action: WorkerAction,
    transcript: list[dict[str, Any]],
    proof_client: Any,
) -> Any:
    del slice_obj
    run_ids = run_ids_from_action_and_transcript(action, transcript)
    predicates: list[PredicateResult] = [
        PredicateResult(
            id="run_set_non_empty" if contract.kind != "standalone_backtests" else "candidate_set_non_empty",
            status=PASS if run_ids else FAIL,
            details={"run_ids": run_ids},
        )
    ]
    proof_calls: list[dict[str, Any]] = []
    evidence_refs: list[str] = []
    consecutive_infra = 0
    required_ids = frozenset(contract.required_predicates)
    for run_id in run_ids:
        predicates.append(PredicateResult(id="each_candidate_has_run_id", status=PASS, evidence_ref=run_id))
        arguments = {
            "action": "inspect",
            "view": "acceptance_proof",
            "run_id": run_id,
            "acceptance_requirements": dict(contract.acceptance_requirements),
        }
        payload = await proof_client.call_proof("backtests_runs", arguments)
        proof_calls.append({"tool": "backtests_runs", "arguments": arguments, "payload": payload})
        proof = extract_proof(payload)
        evidence_refs.extend(proof_refs(proof))
        infra = is_infra_error(proof)
        predicates.append(
            PredicateResult(
                id="each_run_proof_pass",
                status=PASS if proof["status"] == PASS else FAIL,
                evidence_ref=f"mcp://backtests_runs/{run_id}#acceptance_proof",
                details={
                    "run_id": run_id,
                    "proof_status": proof["status"],
                    "blocking_reasons": proof["blocking_reasons"],
                    **({"infra_error": proof["error"]} if infra else {}),
                },
            )
        )
        _append_named_predicates(predicates, proof, run_id=run_id, required_ids=required_ids)
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


def _append_named_predicates(
    predicates: list[PredicateResult],
    proof: dict[str, Any],
    *,
    run_id: str,
    required_ids: frozenset[str] = frozenset(),
) -> None:
    """Map server-side predicate aliases to canonical names.

    Predicates are marked required=True only when their canonical name
    appears in the contract's required_predicates (passed via required_ids).
    This prevents informational predicates from blocking acceptance when
    the contract does not require them (e.g. snapshot/timeframe matching
    for condition_stability_analysis contracts).
    """
    raw = proof.get("predicates")
    if not isinstance(raw, list):
        return
    required_names = {
        "each_run_exists": {"run_resolves", "run_exists"},
        "each_run_completed": {"compute_terminal_completed", "run_completed", "status_terminal_completed"},
        "each_run_has_metrics": {"summary_metrics_present", "metrics_present"},
        "each_run_matches_snapshot_symbol_timeframes": {"snapshot_matches_expected", "symbol_matches_expected", "timeframes_match_expected"},
    }
    by_name: dict[str, str] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        by_name[str(item.get("id") or "")] = str(item.get("status") or "").strip().lower()
    for canonical_id, aliases in required_names.items():
        matched = [by_name[name] for name in aliases if name in by_name]
        if not matched:
            continue
        predicates.append(
            PredicateResult(
                id=canonical_id,
                status=PASS if all(status == PASS for status in matched) else FAIL,
                required=canonical_id in required_ids,
                evidence_ref=f"mcp://backtests_runs/{run_id}#{canonical_id}",
                details={"run_id": run_id, "matched_statuses": matched},
            )
        )


__all__ = ["verify_backtests"]
