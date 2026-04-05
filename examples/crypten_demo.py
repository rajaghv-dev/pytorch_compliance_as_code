"""
CrypTen Minimal Demo — GDPR Art.25 Privacy by Design
=====================================================
Demonstrates secret-shared inference: two parties compute a model
forward pass together without either party seeing the other's plaintext.

Party 0 owns the model weights.
Party 1 owns the input data.
Neither party sees the other's secret.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # CrypTen MPC runs on CPU (multiprocessing + CUDA = init error)

import torch
import crypten
import crypten.mpc as mpc
import crypten.communicator as comm

crypten.init()


# ── 1. Define a small model ──────────────────────────────────────────────────
class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 8)
        self.fc2 = torch.nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


torch.manual_seed(42)
model   = TinyClassifier()
x_plain = torch.randn(3, 4)            # 3 samples, 4 features


# ── 2. Plaintext baseline ────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    y_plain = model(x_plain)
print("Plaintext output:\n", y_plain)


# ── 3. Encrypted inference via secret sharing ─────────────────────────────────
@mpc.run_multiprocess(world_size=2)
def encrypted_inference():
    rank = comm.get().get_rank()

    # Each party encrypts their secret:
    #   Party 0 holds model weights (src=0)
    #   Party 1 holds input data    (src=1)
    x_enc = crypten.cryptensor(x_plain, src=1)   # input from party 1

    # Encrypt each layer's weight and bias from party 0
    w1 = crypten.cryptensor(model.fc1.weight.data, src=0)
    b1 = crypten.cryptensor(model.fc1.bias.data,   src=0)
    w2 = crypten.cryptensor(model.fc2.weight.data, src=0)
    b2 = crypten.cryptensor(model.fc2.bias.data,   src=0)

    # Encrypted forward pass — all arithmetic on secret shares
    h = x_enc.matmul(w1.t()) + b1   # (3,4)·(4,8) → (3,8)
    h = h.relu()
    y_enc = h.matmul(w2.t()) + b2   # (3,8)·(8,2) → (3,2)

    # Reveal result — only the final output is decrypted
    y_dec = y_enc.get_plain_text()

    if rank == 0:
        print("\nEncrypted inference output:\n", y_dec)
        diff = (y_dec - y_plain).abs().max().item()
        print(f"\nMax absolute error vs plaintext: {diff:.2e}")
        print("Inputs and weights never shared in plaintext between parties.")
        print("GDPR Art.25: data protection by design ✓")


encrypted_inference()
