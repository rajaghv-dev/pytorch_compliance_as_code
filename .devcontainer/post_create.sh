#!/usr/bin/env bash
# Codespaces post-create setup — installs all dependencies for the free 2-core / 8GB tier.
# Runs once after the container is created. Takes ~5 min on first launch.
set -e

echo "=== pytorch-compliance-as-code setup ==="
echo "Python: $(python --version)"
echo "RAM:    $(free -h | awk '/^Mem:/{print $2}')"
echo "Disk:   $(df -h / | awk 'NR==2{print $4}') free"
echo ""

# Core package (all torchcomply + pipeline deps)
pip install --quiet --upgrade pip
pip install --quiet -e ".[dev]"

# CrypTen — separate install due to deprecated sklearn dependency
echo "Installing CrypTen..."
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True \
  pip install --quiet crypten --no-build-isolation
pip install --quiet onnxscript  # required by CrypTen + PyTorch 2.x ONNX exporter

# Apply CrypTen / PyTorch 2.x compatibility patch
python - <<'PATCH'
import site, pathlib, re

for sp in site.getsitepackages():
    f = pathlib.Path(sp) / "crypten/nn/onnx_converter.py"
    if f.exists():
        src = f.read_text()
        old = (
            "try:\n"
            "    import torch.onnx.symbolic_registry as sym_registry  # noqa\n"
            "\n"
            "    SYM_REGISTRY = True\n"
            "except ImportError:\n"
            "    from torch.onnx._internal.registration import registry  # noqa\n"
            "\n"
            "    SYM_REGISTRY = False"
        )
        new = (
            "try:\n"
            "    import torch.onnx.symbolic_registry as sym_registry  # noqa\n"
            "\n"
            "    SYM_REGISTRY = True\n"
            "except ImportError:\n"
            "    try:\n"
            "        from torch.onnx._internal.registration import registry  # noqa\n"
            "    except ImportError:\n"
            "        pass  # PyTorch 2.x removed these ONNX internals\n"
            "\n"
            "    SYM_REGISTRY = False"
        )
        if old in src:
            f.write_text(src.replace(old, new))
            print(f"Patched {f}")
        else:
            print(f"Patch not needed or already applied: {f}")
        break
PATCH

echo ""
echo "=== Smoke test ==="
python -c "
import torch, transformers, opacus, captum, peft, mlflow, crypten, reportlab
print('All imports OK')
print(f'  torch       {torch.__version__}')
print(f'  transformers {transformers.__version__}')
print(f'  opacus      {opacus.__version__}')
print(f'  captum      {captum.__version__}')
print(f'  peft        {peft.__version__}')
print(f'  mlflow      {mlflow.__version__}')
print(f'  crypten     {crypten.__version__}')
"

echo ""
echo "Setup complete. Run examples:"
echo "  python examples/01_audit_trail/run.py"
echo "  for i in examples/[0-9]*/run.py; do python \$i; done"
echo ""
echo "Or open notebooks/run_examples.ipynb in Jupyter."
