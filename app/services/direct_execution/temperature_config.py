"""
Adaptive temperature configuration for LM providers.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("orchestrator.direct.temperature")


def get_adaptive_temperature(provider: str, base_temperature: float = 0.7) -> float:
    """
    Get adaptive temperature based on provider capabilities.

    Weak providers (like LMStudio with 35B models) use lower temperature
    to reduce hallucination and improve consistency.
    """
    provider_name = str(provider or "").strip().lower()
    if provider_name in {"lmstudio"}:
        # For weak providers, use more conservative temperature
        # Base 0.7 → adaptive 0.4 for lmstudio
        adaptive = max(0.1, min(base_temperature, 0.55))
        if adaptive != base_temperature:
            logger.info(
                f"Temperature adjusted for {provider_name}: {base_temperature} -> {adaptive}",
            )
        return adaptive
    return base_temperature


def log_temperature_adjustment(provider: str, from_temp: float, to_temp: float) -> None:
    """Log temperature adjustment for debugging."""
    if from_temp != to_temp:
        logger.debug(
            f"Provider {provider}: temperature adjusted {from_temp} -> {to_temp}",
        )


__all__ = ["get_adaptive_temperature", "log_temperature_adjustment"]
