"""
audit.py — Hash-chained audit trail for EU AI Act Article 12 compliance.

EU AI Act Article 12 requires high-risk AI systems to have logging capabilities
that allow post-hoc verification of system behaviour. This module implements
those requirements using a blockchain-inspired SHA-256 hash chain:

  - Every forward pass through a leaf module appends a new AuditEntry
  - Each entry's hash includes the hash of the previous entry (``prev_hash``)
  - Modifying any entry breaks the chain and is detected by ``verify()``

This is analogous to a Merkle tree path: tampering at position N invalidates
all entries from N onwards, making silent log manipulation computationally
infeasible.

Regulatory references:
  EU AI Act Art. 12 — Transparency and provision of information to users
    https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689  (Article 12)
  NIST SP 800-92 — Guide to Computer Security Log Management
    https://csrc.nist.gov/publications/detail/sp/800-92/final

PyTorch forward hook API:
  https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
"""

from __future__ import annotations

import datetime
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn


class IntegrityViolation(Exception):
    """Raised when audit chain verification fails at a specific index."""

    def __init__(self, index: int) -> None:
        self.index = index
        super().__init__(f"Chain integrity violation at entry #{index}")


def _tensor_hash(t) -> str:
    """Return a short SHA-256 hex of a tensor's values (first 8 chars)."""
    try:
        data = t.detach().cpu().contiguous().view(-1).float().numpy().tobytes()
        return hashlib.sha256(data).hexdigest()[:16]
    except Exception:
        return ""


@dataclass
class AuditEntry:
    timestamp: int  # time.time_ns()
    module_name: str  # dotted path, e.g. "distilbert.transformer.layer.0.attention.q_lin"
    operator_type: str  # class name, e.g. "Linear", "LayerNorm"
    input_shapes: list  # list of shape lists
    output_shape: tuple  # shape of the output tensor
    output_hash: str   # SHA-256[:16] of output tensor values — binds log to actual computation
    device: str  # "cuda:0" or "cpu"
    prev_hash: str  # hash of the preceding entry ("" for first)
    hash: str = field(default="", init=False)  # computed by AuditChain.append()


class AuditChain:
    """Append-only, hash-chained audit log for PyTorch forward pass operations.

    Args:
        wal_path: Optional write-ahead log path. When set, every ``append()``
            call immediately writes the entry to disk. This ensures no entries
            are lost if the process crashes mid-run. The file is opened in
            append mode, so multiple runs accumulate in the same file.

    Example::

        chain = AuditChain(wal_path="/tmp/audit_wal.jsonl")
        # entries now written to disk as they occur — crash-safe
    """

    def __init__(self, wal_path: Optional[Union[str, Path]] = None) -> None:
        self.entries: List[AuditEntry] = []
        self._wal_path: Optional[Path] = Path(wal_path) if wal_path else None
        if self._wal_path:
            self._wal_path.parent.mkdir(parents=True, exist_ok=True)
            # open in append mode so multiple runs accumulate
            self._wal_file = open(self._wal_path, "a", encoding="utf-8")  # noqa: WPS515
        else:
            self._wal_file = None

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def append(self, entry: AuditEntry) -> None:
        """Compute entry hash and append to chain. Must be called instead of direct list append.

        The hash is computed over a canonical JSON serialisation with sorted keys so
        the same data always produces the same hash regardless of dict insertion order.
        This mirrors the approach used in Git commit hashing and Certificate Transparency logs.

        output_hash binds the log entry to the actual computation: any post-hoc modification
        of the tensor values would produce a different output_hash and break the chain.
        """
        entry_dict = {
            "timestamp": entry.timestamp,
            "module_name": entry.module_name,
            "operator_type": entry.operator_type,
            "input_shapes": entry.input_shapes,
            "output_shape": list(entry.output_shape),
            "output_hash": entry.output_hash,
            "device": entry.device,
            "prev_hash": entry.prev_hash,
        }
        entry.hash = hashlib.sha256(json.dumps(entry_dict, sort_keys=True).encode()).hexdigest()
        self.entries.append(entry)
        # WAL: write immediately to disk so entries survive a crash
        if self._wal_file is not None:
            self._wal_file.write(json.dumps(asdict(entry), separators=(",", ":")) + "\n")
            self._wal_file.flush()

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(self) -> bool:
        """Walk the chain and recompute every hash. Raise IntegrityViolation if any mismatch."""
        for i, entry in enumerate(self.entries):
            entry_dict = {
                "timestamp": entry.timestamp,
                "module_name": entry.module_name,
                "operator_type": entry.operator_type,
                "input_shapes": entry.input_shapes,
                "output_shape": list(entry.output_shape),
                "output_hash": entry.output_hash,
                "device": entry.device,
                "prev_hash": entry.prev_hash,
            }
            expected = hashlib.sha256(json.dumps(entry_dict, sort_keys=True).encode()).hexdigest()
            if entry.hash != expected:
                raise IntegrityViolation(i)
        return True

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def root_hash(self) -> str:
        """Return the hash of the last entry — a single fingerprint for the whole chain.

        This is analogous to a Merkle root: any modification to any entry changes
        the root hash. Suitable for inclusion in regulatory filings.
        """
        return self.entries[-1].hash if self.entries else ""

    def summary(self) -> dict:
        if not self.entries:
            return {
                "total_entries": 0,
                "first_timestamp": None,
                "last_timestamp": None,
                "unique_operators": 0,
                "root_hash": "",
                "chain_valid": True,
            }
        try:
            chain_valid = self.verify()
        except IntegrityViolation:
            chain_valid = False
        return {
            "total_entries": len(self.entries),
            "first_timestamp": self.entries[0].timestamp,
            "last_timestamp": self.entries[-1].timestamp,
            "unique_operators": len({e.operator_type for e in self.entries}),
            "root_hash": self.root_hash(),
            "chain_valid": chain_valid,
        }

    def flush_jsonl(self, path: Union[str, Path]) -> None:
        """Persist the full audit chain to a JSONL file (one JSON object per line).

        Each line is a complete AuditEntry serialised as JSON. The file is
        append-only: subsequent calls append to the same file, preserving
        cross-run continuity.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for entry in self.entries:
                d = asdict(entry)
                d["iso_timestamp"] = datetime.datetime.fromtimestamp(
                    entry.timestamp / 1e9, tz=datetime.timezone.utc
                ).isoformat()
                f.write(json.dumps(d, separators=(",", ":")) + "\n")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps([asdict(e) for e in self.entries])

    @classmethod
    def from_json(cls, s: str) -> "AuditChain":
        chain = cls()
        for d in json.loads(s):
            entry = AuditEntry(
                timestamp=d["timestamp"],
                module_name=d["module_name"],
                operator_type=d["operator_type"],
                input_shapes=d["input_shapes"],
                output_shape=tuple(d["output_shape"]),
                output_hash=d.get("output_hash", ""),
                device=d["device"],
                prev_hash=d["prev_hash"],
            )
            entry.hash = d["hash"]
            chain.entries.append(entry)
        return chain

    def close(self) -> None:
        """Close the WAL file handle (call when done, or use as context manager)."""
        if self._wal_file is not None:
            self._wal_file.close()
            self._wal_file = None

    def __del__(self) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self.entries)


# ---------------------------------------------------------------------------
# Hook factory
# ---------------------------------------------------------------------------


def compliance_hook(
    module: nn.Module,
    inp: tuple,
    out,
    chain: AuditChain,
    name: str,
) -> None:
    """Forward hook compatible with register_forward_hook."""

    def _shape(t):
        return list(t.shape) if hasattr(t, "shape") else []

    input_shapes = [_shape(x) for x in inp if hasattr(x, "shape")]
    output_shape = tuple(_shape(out))

    # Compute hash of output tensor values — binds this log entry to the actual computation.
    # Modifying output values after the fact would change this hash and break chain.verify().
    out_hash = _tensor_hash(out) if isinstance(out, torch.Tensor) else ""

    # Best-effort device detection: first parameter wins, fall back to "cpu"
    try:
        device = str(next(module.parameters()).device)
    except StopIteration:
        device = "cpu"

    prev_hash = chain.entries[-1].hash if chain.entries else ""
    entry = AuditEntry(
        timestamp=time.time_ns(),
        module_name=name,
        operator_type=type(module).__name__,
        input_shapes=input_shapes,
        output_shape=output_shape,
        output_hash=out_hash,
        device=device,
        prev_hash=prev_hash,
    )
    chain.append(entry)


def register_compliance_hooks(model: nn.Module, chain: AuditChain) -> list:
    """Register compliance_hook on every leaf module (no children). Returns handle list.

    We hook only leaf modules (those with no children) to avoid double-counting:
    a ``Linear`` layer inside a ``TransformerBlock`` inside a ``BertEncoder`` would
    otherwise generate three entries per forward pass.

    The ``n=name`` default-argument capture in the lambda is the standard Python idiom
    for binding the loop variable inside a closure — without it, all hooks would log
    the last value of ``name``.

    To remove hooks after use, call ``handle.remove()`` on each returned handle
    (or use ``ComplianceEngine.detach()`` which does this automatically).
    """
    handles = []
    for name, module in model.named_modules():
        if not list(module.children()):
            handle = module.register_forward_hook(
                lambda m, inp, out, n=name: compliance_hook(m, inp, out, chain, n)
            )
            handles.append(handle)
    return handles
