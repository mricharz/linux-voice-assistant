"""Utility methods."""

import logging
import socket
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from .models import ServerState

_LOGGER = logging.getLogger(__name__)


def get_mac() -> str:
    mac = uuid.getnode()
    mac_str = ":".join(f"{(mac >> i) & 0xff:02x}" for i in range(40, -1, -8))
    return mac_str


def call_all(*callables: Optional[Callable[[], None]]) -> None:
    for item in filter(None, callables):
        item()


def create_event_sockets(paths: List[str]) -> List[Tuple[socket.socket, str]]:
    """Create non-blocking Unix datagram sockets for event emission."""
    sockets = []
    for path in paths:
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            sock.setblocking(False)
            sockets.append((sock, path))
            _LOGGER.debug("Created event socket for: %s", path)
        except Exception:
            _LOGGER.exception("Failed to create event socket for: %s", path)
    return sockets


def emit_event(state: "ServerState", event: str) -> None:
    """Emit an event to all configured event sockets (non-blocking)."""
    if not state.event_sockets:
        return
    data = event.encode()
    for sock, path in state.event_sockets:
        try:
            sock.sendto(data, path)
        except BlockingIOError:
            pass  # Socket buffer full, skip
        except OSError:
            pass  # Socket not available, skip
