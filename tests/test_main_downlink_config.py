"""Tests for the --downlink config gating in __main__ (JR4-166 M2).

The TtsPcmServer (:9090) start and the LAN-callback advertisement are gated on
the resolved downlink mode. The decision is pure (``resolve_downlink_mode``) so
it is tested directly without spinning up the full async ``main()``.
"""

from linux_voice_assistant.downlink_config import resolve_downlink_mode


def test_lan_stays_lan_with_bff_url():
    assert resolve_downlink_mode("lan", "wss://host/v1/wyoming") == "lan"


def test_lan_stays_lan_without_bff_url():
    assert resolve_downlink_mode("lan", "") == "lan"


def test_wss_with_bff_url_is_wss():
    """--downlink wss + a BFF url → wss (the :9090 server is NOT started)."""
    assert resolve_downlink_mode("wss", "wss://host/v1/wyoming") == "wss"


def test_wss_without_bff_url_falls_back_to_lan():
    """--downlink wss but no --bff-url → lan (so the :9090 server still starts)."""
    assert resolve_downlink_mode("wss", "") == "lan"
