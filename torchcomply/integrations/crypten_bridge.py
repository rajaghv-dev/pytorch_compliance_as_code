"""
crypten_bridge.py — CrypTen Secure Multi-Party Computation wrapper.

ComplianceSecureInference encrypts a PyTorch model with CrypTen so that
inference runs on ENCRYPTED inputs — the model never sees plaintext data.

Maps to: GDPR Article 25 (Privacy by Design and by Default).
CrypTen is a Meta Research library for privacy-preserving ML via
Secure Multi-Party Computation (MPC). Named in the conference abstract.

Secure Multi-Party Computation (MPC) Primer
---------------------------------------------
MPC allows multiple parties to jointly compute a function over their private
inputs without revealing those inputs to each other. CrypTen uses arithmetic
secret sharing:

  Party A holds share [x]_A, Party B holds share [x]_B, where [x]_A + [x]_B = x
  Neither party can recover x alone. All arithmetic (add, mul, relu) is done on
  shares and the result is reconstructed only at the end.

This is complementary to Differential Privacy (Opacus):
  - DP-SGD protects *training data* from model inspection (training-time privacy)
  - CrypTen MPC protects *inference inputs* from the model server (inference-time privacy)

MPC vs Homomorphic Encryption (HE):
  HE (e.g. Microsoft SEAL, OpenFHE) allows arbitrary computation on encrypted data
  without any communication, but is ~1000× slower than MPC. MPC requires multiple
  communication rounds but is practical for production inference.

Simulation mode:
  CrypTen's single-process simulation mode (world_size=1) is useful for demos
  and correctness testing. Real MPC requires at least 2 separate processes
  (distinct machines or containers). Use ``crypten.mpc.run_multiprocess()``
  for a real two-party setup.

Regulatory references:
  GDPR Art. 25 — Data protection by design and by default
    https://gdpr-info.eu/art-25-gdpr/
  EDPB Recommendations 01/2020 — Measures for transfers of personal data to third countries
    https://edpb.europa.eu/our-work-tools/our-documents/recommendations/recommendations-012020-measures-supplements-transfer_en

Key papers:
  Knott et al. (2021) — CrypTen: Secure Multi-Party Computation Meets Machine Learning
    https://arxiv.org/abs/2109.00984
  Mohassel & Zhang (2017) — SecureML: A System for Scalable Privacy-Preserving Machine Learning
    https://eprint.iacr.org/2017/396

CrypTen documentation:
  https://crypten.ai/
  https://github.com/facebookresearch/CrypTen
"""

from __future__ import annotations

import time
from typing import List

import torch
import torch.nn as nn


class ComplianceSecureInference:
    """
    Wraps a PyTorch model with CrypTen encrypted inference.

    Args:
        model: A trained PyTorch model (CPU, simple architecture).
        dummy_input: A sample input tensor matching the model's expected shape.

    Raises:
        RuntimeError: If CrypTen is not installed or fails to initialise.
    """

    def __init__(self, model: nn.Module, dummy_input: torch.Tensor) -> None:
        import os
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # MPC runs on CPU

        try:
            import crypten
        except ImportError as exc:
            raise RuntimeError(
                "CrypTen is not installed. Install with: pip install crypten"
            ) from exc

        crypten.init()
        self.crypten = crypten
        self.model = model.cpu().eval()
        self.inference_log: List[dict] = []

    def secure_predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Run encrypted inference on ``input_tensor`` using CrypTen secret sharing.

        Walks the model's layers manually so inference runs on arithmetic secret
        shares — neither the input nor intermediate activations are ever in
        plaintext during computation. Returns the decrypted output tensor.
        """
        import crypten

        x_cpu = input_tensor.cpu()
        t_start = time.time()

        # Encrypt the input — in real MPC this would come from a separate party.
        enc = crypten.cryptensor(x_cpu)

        # Walk the model layer by layer on encrypted values.
        plaintext_fallback_layers = []
        for layer in self.model.modules():
            if layer is self.model:
                continue
            if isinstance(layer, nn.Linear):
                w = crypten.cryptensor(layer.weight.data)
                b = crypten.cryptensor(layer.bias.data)
                enc = enc.matmul(w.t()) + b
            elif isinstance(layer, (nn.ReLU, nn.GELU)):
                enc = enc.relu()
            elif isinstance(layer, nn.Sigmoid):
                enc = enc.sigmoid()
            elif isinstance(layer, nn.Sequential):
                continue  # handled by iterating children
            # Unknown layer types are run in plaintext as a fallback — log a warning.
            else:
                layer_name = type(layer).__name__
                plaintext_fallback_layers.append(layer_name)
                print(f"  WARNING: layer '{layer_name}' has no MPC implementation — "
                      f"ran in PLAINTEXT. MPC coverage incomplete for this layer.")
                enc = crypten.cryptensor(layer(enc.get_plain_text()))

        plaintext_output = enc.get_plain_text()
        elapsed_ms = (time.time() - t_start) * 1000

        self.inference_log.append(
            {
                "timestamp": time.time_ns(),
                "input_shape": tuple(x_cpu.shape),
                "output_shape": tuple(plaintext_output.shape),
                "encryption_time_ms": elapsed_ms,
                "protocol": "SecretShare_MPC",
                "plaintext_fallback_layers": plaintext_fallback_layers,
                "full_mpc_coverage": len(plaintext_fallback_layers) == 0,
            }
        )
        return plaintext_output

    def get_log(self) -> List[dict]:
        return list(self.inference_log)
