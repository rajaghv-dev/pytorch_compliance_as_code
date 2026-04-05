"""
otel.py — OpenTelemetry span emission for compliance observability.

Emits structured traces for key compliance events (inference, fairness checks,
consent lookups) so they can be visualised in Jaeger / any OTLP-compatible backend.

Why OpenTelemetry for compliance?
----------------------------------
EU AI Act Article 14 requires that high-risk AI systems allow human oversight,
including the ability to monitor the system in operation. OpenTelemetry (OTel)
is the CNCF standard for distributed tracing and provides:

  1. Real-time visibility — compliance events appear in production dashboards
     alongside latency, errors, and throughput metrics.
  2. Correlation — a single trace ID ties together the model forward pass,
     fairness check, consent lookup, and audit chain entry for one request.
  3. Backend agnosticism — the same code ships traces to Jaeger (open source),
     Grafana Tempo, AWS X-Ray, Google Cloud Trace, or Datadog without changes.

Span Attributes
---------------
OTel span attributes are key-value pairs attached to a span. We use string
values for all compliance attributes to ensure compatibility with all backends.
Attribute names follow the OTel semantic conventions where possible:
  https://opentelemetry.io/docs/reference/specification/trace/semantic_conventions/

Regulatory references:
  EU AI Act Art. 14 — Human oversight measures
    https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689  (Article 14)
  EU AI Act Art. 61 — Post-market monitoring (observability during deployment)

OpenTelemetry documentation:
  Python SDK: https://opentelemetry-python.readthedocs.io/
  Specification: https://opentelemetry.io/docs/reference/specification/
  OTLP protocol: https://opentelemetry.io/docs/reference/specification/protocol/otlp/

Compatible backends:
  Jaeger: https://www.jaegertracing.io/
  Grafana Tempo: https://grafana.com/oss/tempo/
  Zipkin: https://zipkin.io/
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Optional

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False


class OtelComplianceLogger:
    """
    Emits OpenTelemetry spans for compliance-relevant events.

    Args:
        service_name: Service name tag attached to every span.
        exporter: Optional OTLP/Jaeger exporter. When None, spans are printed locally.
    """

    def __init__(self, service_name: str = "torchcomply", exporter=None) -> None:
        if not _OTEL_AVAILABLE:
            raise ImportError(
                "opentelemetry is not installed. "
                "Run: pip install opentelemetry-api opentelemetry-sdk"
            )
        provider = TracerProvider()
        if exporter is None:
            self._mem_exporter = InMemorySpanExporter()
            provider.add_span_processor(SimpleSpanProcessor(self._mem_exporter))
        else:
            self._mem_exporter = None
            provider.add_span_processor(SimpleSpanProcessor(exporter))

        trace.set_tracer_provider(provider)
        self._tracer = trace.get_tracer(service_name)

    @contextmanager
    def span(self, name: str, attributes: Optional[Dict[str, str]] = None):
        """Context manager that wraps a block in an OTel span."""
        with self._tracer.start_as_current_span(name) as s:
            if attributes:
                for k, v in attributes.items():
                    s.set_attribute(k, v)
            yield s

    def get_finished_spans(self) -> list:
        """Return all finished spans (only when using the in-memory exporter)."""
        if self._mem_exporter:
            return self._mem_exporter.get_finished_spans()
        return []
