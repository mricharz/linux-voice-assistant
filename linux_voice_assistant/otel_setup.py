"""OpenTelemetry tracing setup for the voice assistant.

Initializes distributed tracing with OTLP/HTTP export.
Designed to be zero-overhead when disabled (OTEL_ENABLED != "true"):
heavy SDK modules are only imported when tracing is actually enabled.

Environment variables:
  OTEL_ENABLED              - Set to "true" to enable tracing (default: disabled)
  OTEL_SERVICE_NAME         - Service name for spans (default: jarvis-smartspot-livingroom)
  OTEL_EXPORTER_OTLP_ENDPOINT - Collector endpoint (default: http://172.16.5.51:4318)
"""

import logging
import os
from contextlib import contextmanager
from typing import Any, Generator, Optional

_LOGGER = logging.getLogger(__name__)

# Sentinel: set to True after successful init
_tracing_enabled = False

# Holds the TracerProvider once initialised (avoids repeated lookups)
_tracer_provider: Any = None


def _is_otel_requested() -> bool:
    """Check if the user opted in to tracing via env var."""
    return os.environ.get("OTEL_ENABLED", "").lower() == "true"


def init_tracing() -> bool:
    """Initialise OTel tracing if OTEL_ENABLED=true.

    Returns True if tracing was successfully enabled, False otherwise.
    Safe to call multiple times (idempotent).
    """
    global _tracing_enabled, _tracer_provider

    if _tracing_enabled:
        return True

    if not _is_otel_requested():
        _LOGGER.debug("OTel tracing disabled (OTEL_ENABLED != 'true')")
        return False

    try:
        # Late-import heavy SDK modules only when actually needed
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.propagate import set_global_textmap
        from opentelemetry.propagators.composite import CompositeHTTPPropagator
        from opentelemetry.trace.propagation.tracecontext import (
            TraceContextTextMapPropagator,
        )

        service_name = os.environ.get(
            "OTEL_SERVICE_NAME", "jarvis-smartspot-livingroom"
        )
        endpoint = os.environ.get(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://172.16.5.51:4318"
        )

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")
        # BatchSpanProcessor exports asynchronously — no blocking on the hot path
        provider.add_span_processor(BatchSpanProcessor(exporter))

        trace.set_tracer_provider(provider)
        _tracer_provider = provider

        # W3C TraceContext propagation (standard for downstream services)
        set_global_textmap(
            CompositeHTTPPropagator([TraceContextTextMapPropagator()])
        )

        _tracing_enabled = True
        _LOGGER.info(
            "OTel tracing enabled (service=%s, endpoint=%s)",
            service_name,
            endpoint,
        )
        return True

    except ImportError:
        _LOGGER.warning(
            "OTel tracing requested but opentelemetry packages not installed"
        )
        return False
    except Exception:
        _LOGGER.exception("Failed to initialise OTel tracing")
        return False


def shutdown_tracing() -> None:
    """Flush pending spans and shut down the tracer provider."""
    global _tracing_enabled, _tracer_provider

    if not _tracing_enabled or _tracer_provider is None:
        return

    try:
        _tracer_provider.shutdown()
        _LOGGER.info("OTel tracing shut down")
    except Exception:
        _LOGGER.exception("Error shutting down OTel tracing")
    finally:
        _tracing_enabled = False
        _tracer_provider = None


def get_tracer(name: str = "linux_voice_assistant") -> Any:
    """Return an OTel Tracer if tracing is enabled, otherwise a no-op stub.

    The returned object always supports ``start_as_current_span()``,
    so callers never need to check whether tracing is active.
    """
    if _tracing_enabled:
        from opentelemetry import trace

        return trace.get_tracer(name)
    return _NoOpTracer()


def get_current_trace_context() -> Optional[dict]:
    """Extract the current W3C traceparent as a dict (for propagation).

    Returns None if tracing is disabled or no active span.
    """
    if not _tracing_enabled:
        return None

    try:
        from opentelemetry import context as otel_context
        from opentelemetry.propagate import inject

        carrier: dict = {}
        inject(carrier, context=otel_context.get_current())
        return carrier if carrier else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# No-op fallback (zero overhead when tracing is disabled)
# ---------------------------------------------------------------------------

class _NoOpSpan:
    """Minimal no-op span that satisfies the context-manager protocol."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        pass

    def record_exception(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _NoOpTracer:
    """Minimal no-op tracer returned when OTel is disabled."""

    @contextmanager
    def start_as_current_span(
        self, name: str, **kwargs: Any
    ) -> Generator[_NoOpSpan, None, None]:
        yield _NoOpSpan()
