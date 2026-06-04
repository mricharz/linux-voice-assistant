"""Wyoming protocol wire helpers, transport-neutral.

Event-type constants and the wire-format serializer shared by the Wyoming
transport clients. SmartSpot has a single uplink transport — the multiplexed
WSS client (``wyoming_ws_client.WyomingWsClient``) to the BFF — but the wire
format it speaks (and the BFF byte-passes to ``parakeet:10300``) is the plain
Wyoming event framing defined here.

Wire format (per event):
  - JSON header line (terminated by \\n)
  - optional data bytes (length from header["data_length"])
  - optional binary payload (length from header["payload_length"])
"""

import json
from typing import Any, Dict, Optional

# Wyoming event type constants
_EVENT_INFO = "info"
_EVENT_AUDIO_START = "audio-start"
_EVENT_AUDIO_CHUNK = "audio-chunk"
_EVENT_AUDIO_STOP = "audio-stop"
_EVENT_TRANSCRIPT = "transcript"


def _build_event(
    event_type: str,
    data: Optional[Dict[str, Any]] = None,
    payload: bytes = b"",
) -> bytes:
    """Serialize a Wyoming event into wire format bytes."""
    data_bytes = json.dumps(data).encode("utf-8") if data else b""
    header = {
        "type": event_type,
        "data_length": len(data_bytes),
        "payload_length": len(payload),
    }
    return json.dumps(header).encode("utf-8") + b"\n" + data_bytes + payload
