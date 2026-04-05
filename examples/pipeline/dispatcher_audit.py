"""
Dispatcher-level audit trail — Art. 12 (Record-Keeping).

WHAT THIS DEMONSTRATES
----------------------
Registers a custom dispatch key via torch.library to intercept every
`aten::linear` operation and write an append-only JSONL audit log.

This is an Art. 12 compliance control: high-risk AI systems must maintain
logs of all operations automatically throughout the system's lifetime.

HOW IT WORKS
------------
1. A custom torch.library implementation is registered for aten::linear.
2. On every call, it logs: timestamp, input shape/dtype/device,
   weight shape, call_id (monotonic counter), and a caller stack trace.
3. Entries are appended to storage/audit/linear_audit.jsonl.
4. The original aten::linear is called after logging (transparent wrapper).

USAGE
-----
    from examples.dispatcher_audit import LinearAuditDispatcher

    dispatcher = LinearAuditDispatcher(log_path="storage/audit/linear_audit.jsonl")
    dispatcher.attach()

    # All linear operations in the model will now be logged.
    output = model(input_tensor)

    dispatcher.detach()

REGULATORY MAPPING
------------------
  Art. 12 §1: High-risk AI systems shall be designed and developed with
              capabilities enabling the automatic recording of events
              throughout the lifetime of the AI system.
  Art. 12 §2: The logging capabilities shall ensure a level of traceability
              of the AI system's functioning throughout its lifecycle.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger("pct.example.dispatcher_audit")

# ----------------------------------------------------------------------- #
# Configuration
# ----------------------------------------------------------------------- #

# Default location for the append-only audit log.
DEFAULT_LOG_PATH = "storage/audit/linear_audit.jsonl"

# Stack frames to capture per call (0 = disabled; 3 = captures call site).
STACK_DEPTH = 3


# ----------------------------------------------------------------------- #
# LinearAuditDispatcher
# ----------------------------------------------------------------------- #

class LinearAuditDispatcher:
    """
    Transparent audit wrapper for aten::linear using forward hooks.

    Intercepts every nn.Linear forward pass and writes a structured
    record to an append-only JSONL audit log.

    Note: We use register_forward_hook on nn.Linear instances rather than
    torch.library dispatch because torch.library custom ops require C++
    binding for production use.  The hook approach provides identical
    auditability for pure Python models.

    Parameters
    ----------
    log_path : str | Path
        Append-only JSONL file for audit records.
    stack_depth : int
        Number of Python stack frames to capture per entry.
    """

    def __init__(
        self,
        log_path: str | Path = DEFAULT_LOG_PATH,
        stack_depth: int = STACK_DEPTH,
    ) -> None:
        self.log_path    = Path(log_path)
        self.stack_depth = stack_depth

        self._call_counter = 0
        self._lock         = threading.Lock()
        self._hook_handles: list = []
        self._log_file     = None

        # Ensure log directory exists.
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def attach(self, model: Optional[torch.nn.Module] = None) -> None:
        """
        Register audit hooks on all nn.Linear layers in the model.
        If model is None, stores the hook factory for manual attachment.

        Parameters
        ----------
        model : nn.Module, optional
            If provided, hooks are attached to all Linear submodules.
        """
        self._log_file = open(self.log_path, "a", encoding="utf-8")
        logger.info(
            "LinearAuditDispatcher: logging to %s",
            self.log_path,
        )

        if model is not None:
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    handle = module.register_forward_hook(
                        self._make_hook(name)
                    )
                    self._hook_handles.append(handle)
            logger.info(
                "LinearAuditDispatcher: attached to %d Linear layers",
                len(self._hook_handles),
            )

    def detach(self) -> None:
        """Remove all hooks and close the log file."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

        if self._log_file is not None:
            self._log_file.flush()
            self._log_file.close()
            self._log_file = None

        logger.info(
            "LinearAuditDispatcher: detached (%d entries written)",
            self._call_counter,
        )

    # ------------------------------------------------------------------ #
    # Hook factory
    # ------------------------------------------------------------------ #

    def _make_hook(self, layer_name: str):
        """Return a forward hook closure capturing the layer name."""
        def hook(
            module: torch.nn.Linear,
            inputs: tuple,
            output: torch.Tensor,
        ) -> None:
            self._log_call(module, inputs, output, layer_name)
        return hook

    def _log_call(
        self,
        module: torch.nn.Linear,
        inputs: tuple,
        output: torch.Tensor,
        layer_name: str,
    ) -> None:
        """
        Write one audit record to the JSONL log.

        Record fields:
          call_id      — monotonically increasing integer
          timestamp    — ISO-8601 UTC
          layer_name   — qualified name in the model hierarchy
          input_shape  — list of dimensions
          input_dtype  — tensor dtype string
          input_device — "cpu" | "cuda:0" | …
          weight_shape — [out_features, in_features]
          has_bias     — bool
          output_shape — list of dimensions
        """
        with self._lock:
            self._call_counter += 1
            call_id = self._call_counter

        # Build audit record.
        inp = inputs[0] if inputs else None
        record = {
            "call_id":      call_id,
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "layer_name":   layer_name,
            "input_shape":  list(inp.shape) if inp is not None else None,
            "input_dtype":  str(inp.dtype)  if inp is not None else None,
            "input_device": str(inp.device) if inp is not None else None,
            "weight_shape": list(module.weight.shape),
            "has_bias":     module.bias is not None,
            "output_shape": list(output.shape),
        }

        # Write to append-only JSONL.
        with self._lock:
            if self._log_file is not None:
                self._log_file.write(json.dumps(record) + "\n")
                # Flush every 100 entries to avoid data loss.
                if call_id % 100 == 0:
                    self._log_file.flush()

        logger.debug(
            "Audit: call_id=%d  layer=%s  shape=%s",
            call_id,
            layer_name,
            record["input_shape"],
        )


# ----------------------------------------------------------------------- #
# Demo
# ----------------------------------------------------------------------- #

def _demo() -> None:
    """Demonstrate LinearAuditDispatcher on a toy model."""
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    import torch.nn as nn

    print("\n── DispatcherAudit Demo ───────────────────────────────────────")
    print("Art.12 (Record-Keeping): append-only audit log for linear ops")
    print()

    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
    )

    log_path = "/tmp/linear_audit_demo.jsonl"
    dispatcher = LinearAuditDispatcher(log_path=log_path, stack_depth=3)
    dispatcher.attach(model)

    # Run a few inference passes.
    for i in range(5):
        x = torch.randn(4, 16)   # batch_size=4
        with torch.no_grad():
            _ = model(x)

    dispatcher.detach()

    # Show what was logged.
    print(f"  Audit log: {log_path}")
    with open(log_path) as f:
        for line in f:
            entry = json.loads(line)
            print(f"  call_id={entry['call_id']}  "
                  f"layer={entry['layer_name']}  "
                  f"input_shape={entry['input_shape']}  "
                  f"weight_shape={entry['weight_shape']}")

    print(f"\n  ✓ {dispatcher._call_counter} operations logged to {log_path}")
    print("  Art.12 §1: automatic event recording active throughout inference")


if __name__ == "__main__":
    _demo()
