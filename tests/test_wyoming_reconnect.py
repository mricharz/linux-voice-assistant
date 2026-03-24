"""Tests for Wyoming client reconnect resilience."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from linux_voice_assistant.wyoming_client import WyomingClient, WyomingClientConfig


@pytest.fixture
def config():
    return WyomingClientConfig(
        host="127.0.0.1",
        port=19999,
        reconnect_min_delay=0.1,
        reconnect_max_delay=0.5,
        reconnect_multiplier=2.0,
        reconnect_jitter=0.0,  # no jitter for deterministic tests
        read_timeout=1.0,
    )


def test_config_defaults():
    """Max backoff should be 300s per CLAUDE.md requirements."""
    cfg = WyomingClientConfig()
    assert cfg.reconnect_max_delay == 300.0
    assert cfg.read_timeout == 60.0


def test_config_read_timeout_default():
    """Read timeout should default to 60s."""
    cfg = WyomingClientConfig()
    assert cfg.read_timeout == 60.0


def test_connection_state_callback_called(config):
    """on_connection_state should be called on connect and disconnect."""
    states = []
    client = WyomingClient(
        config,
        on_connection_state=lambda connected: states.append(connected),
    )
    client._notify_connection_state(True)
    client._notify_connection_state(False)
    assert states == [True, False]


def test_connection_state_callback_none(config):
    """_notify_connection_state should not crash when callback is None."""
    client = WyomingClient(config)
    # Should be a no-op, not raise
    client._notify_connection_state(True)
    client._notify_connection_state(False)


def test_connection_state_callback_exception(config):
    """_notify_connection_state should not propagate callback exceptions."""
    def bad_callback(connected: bool) -> None:
        raise RuntimeError("callback error")

    client = WyomingClient(config, on_connection_state=bad_callback)
    # Should log but not raise
    client._notify_connection_state(True)


def test_close_connection_cleans_up():
    """_close_connection should reset state without crashing."""
    cfg = WyomingClientConfig(host="127.0.0.1", port=19999)
    client = WyomingClient(cfg)
    # Not connected — closing should be a no-op
    asyncio.run(client._close_connection())
    assert not client._connected
    assert not client._utterance_active


def test_close_connection_with_writer():
    """_close_connection should close writer and reset references."""
    cfg = WyomingClientConfig(host="127.0.0.1", port=19999)
    client = WyomingClient(cfg)

    # Mock a writer
    mock_writer = MagicMock()
    mock_writer.close = MagicMock()
    mock_writer.wait_closed = AsyncMock()
    client._writer = mock_writer
    client._reader = MagicMock()
    client._connected = True
    client._utterance_active = True

    asyncio.run(client._close_connection())
    assert not client._connected
    assert not client._utterance_active
    assert client._writer is None
    assert client._reader is None
    mock_writer.close.assert_called_once()


def test_initial_state(config):
    """Client should start in disconnected state."""
    client = WyomingClient(config)
    assert not client.connected
    assert not client._running
    assert not client._utterance_active
    assert client._writer is None
    assert client._reader is None
