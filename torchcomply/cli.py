"""
torchcomply CLI — command-line compliance tool.

Commands:
  torchcomply validate <audit.jsonl>   Verify audit chain integrity and print summary
  torchcomply diff <before.json> <after.json>  Compare two compliance snapshots
  torchcomply version                   Print version

Usage examples::

  torchcomply validate examples/01_audit_trail/sample_output/audit_chain.jsonl
  torchcomply diff snapshot_run1.json snapshot_run2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_validate(args) -> int:
    """Verify a JSONL audit chain file and print a summary."""
    from torchcomply.core.audit import AuditChain, IntegrityViolation

    path = Path(args.file)
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 1

    chain = AuditChain()
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            from torchcomply.core.audit import AuditEntry
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

    print(f"Loaded {len(chain)} entries from {path}")
    try:
        chain.verify()
        s = chain.summary()
        print("✅ Chain integrity: VALID")
        print(f"   Root hash:       {s['root_hash']}")
        print(f"   Entries:         {s['total_entries']}")
        print(f"   Unique ops:      {s['unique_operators']}")
        return 0
    except IntegrityViolation as exc:
        print(f"❌ INTEGRITY VIOLATION at entry #{exc.index}")
        print("   The audit chain has been tampered with or corrupted.")
        return 2


def _cmd_diff(args) -> int:
    """Compare two compliance snapshot JSON files."""
    from torchcomply.core.diff import ComplianceDiff, ComplianceSnapshot

    for p in [args.before, args.after]:
        if not Path(p).exists():
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            return 1

    before_data = json.loads(Path(args.before).read_text())
    after_data = json.loads(Path(args.after).read_text())

    before = ComplianceSnapshot(**{
        k: v for k, v in before_data.items()
        if k in ComplianceSnapshot.__dataclass_fields__
    })
    after = ComplianceSnapshot(**{
        k: v for k, v in after_data.items()
        if k in ComplianceSnapshot.__dataclass_fields__
    })

    diff = ComplianceDiff(before, after)
    print(diff.report())
    return 1 if diff.has_regressions else 0


def _cmd_version(_args) -> int:
    import torchcomply
    print(f"torchcomply {torchcomply.__version__}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="torchcomply",
        description="Compliance-as-Code CLI for PyTorch — EU AI Act / GDPR",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # validate
    p_val = sub.add_parser("validate", help="Verify audit chain integrity from a JSONL file")
    p_val.add_argument("file", help="Path to audit_chain.jsonl")
    p_val.set_defaults(func=_cmd_validate)

    # diff
    p_diff = sub.add_parser("diff", help="Compare two compliance snapshots (JSON files)")
    p_diff.add_argument("before", help="Path to before snapshot JSON")
    p_diff.add_argument("after", help="Path to after snapshot JSON")
    p_diff.set_defaults(func=_cmd_diff)

    # version
    p_ver = sub.add_parser("version", help="Print torchcomply version")
    p_ver.set_defaults(func=_cmd_version)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
