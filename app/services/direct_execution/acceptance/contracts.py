"""Typed acceptance contract/result models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


PASS = "pass"
FAIL = "fail"
PARTIAL = "partial"


@dataclass(frozen=True)
class AcceptanceContract:
    kind: str = "generic_read_advisory"
    mode: str = "strict"
    required_subjects: str = "none"
    required_predicates: list[str] = field(default_factory=list)
    mcp_proof_calls: list[dict[str, Any]] = field(default_factory=list)
    forbid_contradictions: list[str] = field(default_factory=list)
    allow_advisory_unblock: bool = False
    acceptance_requirements: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PredicateResult:
    id: str
    status: str
    required: bool = True
    evidence_ref: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AcceptanceResult:
    status: str
    contract: dict[str, Any]
    predicates: list[PredicateResult] = field(default_factory=list)
    blocking_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    proof_calls: list[dict[str, Any]] = field(default_factory=list)
    incidents: list[dict[str, Any]] = field(default_factory=list)
    route: str = "fallback_allowed"

    @property
    def passed(self) -> bool:
        return self.status == PASS

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["passed"] = self.passed
        return data


def normalize_contract(value: dict[str, Any] | AcceptanceContract | None) -> AcceptanceContract:
    if isinstance(value, AcceptanceContract):
        return value
    if not isinstance(value, dict):
        return AcceptanceContract()
    return AcceptanceContract(
        kind=str(value.get("kind") or "generic_read_advisory"),
        mode=str(value.get("mode") or "strict"),
        required_subjects=str(value.get("required_subjects") or "none"),
        required_predicates=_string_list(value.get("required_predicates")),
        mcp_proof_calls=_dict_list(value.get("mcp_proof_calls")),
        forbid_contradictions=_string_list(value.get("forbid_contradictions")),
        allow_advisory_unblock=bool(value.get("allow_advisory_unblock", False)),
        acceptance_requirements=dict(value.get("acceptance_requirements") or {}),
    )


def result_from_predicates(
    *,
    contract: AcceptanceContract,
    predicates: list[PredicateResult],
    warnings: list[str] | None = None,
    evidence_refs: list[str] | None = None,
    proof_calls: list[dict[str, Any]] | None = None,
    incidents: list[dict[str, Any]] | None = None,
    route: str = "fallback_allowed",
) -> AcceptanceResult:
    blockers = [
        predicate.id
        for predicate in predicates
        if predicate.required and predicate.status not in {PASS, "not_applicable"}
    ]
    status = PASS if not blockers else FAIL
    return AcceptanceResult(
        status=status,
        contract=contract.to_dict(),
        predicates=list(predicates),
        blocking_reasons=blockers,
        warnings=list(warnings or []),
        evidence_refs=list(evidence_refs or []),
        proof_calls=list(proof_calls or []),
        incidents=list(incidents or []),
        route=route,
    )


def fail_result(
    *,
    contract: AcceptanceContract,
    reason: str,
    route: str = "fallback_allowed",
    details: dict[str, Any] | None = None,
) -> AcceptanceResult:
    return result_from_predicates(
        contract=contract,
        predicates=[PredicateResult(id=reason, status=FAIL, details=dict(details or {}))],
        route=route,
    )


def pass_result(*, contract: AcceptanceContract, predicates: list[PredicateResult] | None = None) -> AcceptanceResult:
    return result_from_predicates(contract=contract, predicates=list(predicates or []))


def extract_proof(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract the inner proof dict from a possibly-enveloped MCP response.

    Envelope forms handled (in priority order):
    1. payload["data"]["proof"]  -- normalized envelope (status=ok, data={proof={...}})
    2. payload["proof"]          -- direct envelope ({proof: {status: ...}})
    3. payload itself            -- flat envelope (already the proof dict)
    """
    if not isinstance(payload, dict):
        return {"status": "", "error": "", "error_type": "", "predicates": [], "evidence_refs": [], "blocking_reasons": []}
    data = payload.get("data")
    if isinstance(data, dict) and isinstance(data.get("proof"), dict):
        proof = data["proof"]
    elif isinstance(payload.get("proof"), dict):
        proof = payload["proof"]
    else:
        proof = payload
    return {
        "status": str(proof.get("status") or "").strip().lower(),
        "error": str(proof.get("error") or "").strip(),
        "error_type": str(proof.get("error_type") or "").strip(),
        "predicates": proof.get("predicates") if isinstance(proof.get("predicates"), list) else [],
        "evidence_refs": proof.get("evidence_refs") if isinstance(proof.get("evidence_refs"), list) else [],
        "blocking_reasons": proof.get("blocking_reasons") if isinstance(proof.get("blocking_reasons"), list) else [],
    }


_INFRA_ERROR_SIGNALS = (
    "AttributeError", "TypeError", "KeyError", "ImportError",
    "ModuleNotFoundError", "RuntimeError", "ConnectionError",
    "TimeoutError", "OSError", "IOError", "BrokenPipeError",
    "ConnectionResetError", "InternalError", "OperationalError",
)

_INFRA_FAIL_FAST_THRESHOLD = 2


def is_infra_error(proof: dict[str, Any]) -> bool:
    """Detect whether a proof response indicates an infrastructure/backend error."""
    if proof.get("status") == "pass":
        return False
    error = str(proof.get("error") or "")
    error_type = str(proof.get("error_type") or "")
    for signal in _INFRA_ERROR_SIGNALS:
        if signal in error or signal in error_type:
            return True
    return False


def proof_refs(proof: Any) -> list[str]:
    """Extract evidence_refs from a proof dict (raw or normalized)."""
    if not isinstance(proof, dict):
        return []
    raw = proof.get("evidence_refs")
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def check_infra_route(predicates: list[PredicateResult], consecutive_count: int = 0) -> tuple[str, int]:
    """Check if recent predicates indicate an infra-error pattern requiring fast-fail.

    Returns (route_override, updated_consecutive_count).
    route_override is "hard_block_infra" when fast-fail threshold is reached, else "".
    """
    if predicates and predicates[-1].status == FAIL and "infra_error" in predicates[-1].details:
        consecutive_count += 1
    else:
        consecutive_count = 0
    if consecutive_count >= _INFRA_FAIL_FAST_THRESHOLD:
        return "hard_block_infra", consecutive_count
    return "", consecutive_count


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


__all__ = [
    "AcceptanceContract",
    "AcceptanceResult",
    "FAIL",
    "PARTIAL",
    "PASS",
    "PredicateResult",
    "check_infra_route",
    "extract_proof",
    "fail_result",
    "is_infra_error",
    "normalize_contract",
    "pass_result",
    "proof_refs",
    "result_from_predicates",
]
