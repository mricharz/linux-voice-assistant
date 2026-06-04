"""Downlink-transport resolution helpers (JR4-166 M2).

A leaf module (no heavy audio/ESPHome imports) so the gating decision is
unit-testable in isolation. ``__main__`` re-exports + uses ``resolve_downlink_mode``.
"""

import logging

_LOGGER = logging.getLogger(__name__)


def resolve_downlink_mode(downlink: str, bff_url: str) -> str:
    """Resolve the effective TTS downlink transport.

    "wss" (multiplexed-over-the-BFF-link) requires ``--bff-url`` — without it
    there is no link socket to carry TTS back, so fall back to "lan"
    (Direct-TCP :9090 + RH dial-back) which is also the rollback default. Pure
    so the gating is unit-testable without spinning up ``main()``.
    """
    if downlink == "wss" and not bff_url:
        _LOGGER.warning(
            "--downlink wss requires --bff-url; falling back to lan (Direct-TCP)"
        )
        return "lan"
    return downlink
