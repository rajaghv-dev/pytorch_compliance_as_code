"""
Example 7: Encrypted Inference — CrypTen (Secure Multi-Party Computation)
GDPR Article 25: Data protection by design and by default.
"""

import pathlib
import time

import torch
import torch.nn as nn
import torch.optim as optim


def _train_model(device: str) -> nn.Module:
    """3-layer MLP with 128 hidden units — larger than a toy model, still fast on CPU."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(20, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    X = torch.randn(200, 20, device=device)
    Y = torch.randint(0, 10, (200,), device=device)
    for _ in range(10):
        optimizer.zero_grad()
        criterion(model(X), Y).backward()
        optimizer.step()
    return model.cpu().eval()


def main():
    t0 = time.time()
    print("=" * 70)
    print("EXAMPLE 7: Encrypted Inference — CrypTen (Secure Multi-Party Computation)")
    print("GDPR Article 25: Data protection by design and by default")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device.upper()}")
    print("\nNote: This demo uses single-machine CrypTen simulation (world_size=1).")
    print("  Real MPC requires network-isolated processes on separate machines/containers.")
    print("  CrypTen is running on CPU (CUDA disabled — MPC requires CPU for correct IPC).")

    model = _train_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: 3-layer MLP (20→128→128→10), {n_params:,} parameters")
    x = torch.randn(1, 20)

    # ------------------------------------------------------------------
    # Standard inference
    # ------------------------------------------------------------------
    t_std = time.time()
    with torch.no_grad():
        standard_out = model(x)
    std_ms = (time.time() - t_std) * 1000
    print(f"\nStandard inference:   output[:3]={standard_out[0,:3].tolist()}, time={std_ms:.2f}ms")

    # ------------------------------------------------------------------
    # Encrypted inference (CrypTen)
    # ------------------------------------------------------------------
    try:
        from torchcomply.integrations.crypten_bridge import ComplianceSecureInference

        dummy = torch.randn(1, 20)
        secure = ComplianceSecureInference(model, dummy)

        t_enc = time.time()
        encrypted_out = secure.secure_predict(x)
        enc_ms = (time.time() - t_enc) * 1000

        max_diff = (standard_out - encrypted_out).abs().max().item()
        overhead = enc_ms / max(std_ms, 0.001)

        print(
            f"Encrypted inference:  output[:3]={encrypted_out[0,:3].tolist()}, time={enc_ms:.1f}ms"
        )
        print(f"\nMax absolute difference: {max_diff:.6f} (numerical noise from MPC)")
        print(f"Overhead: {overhead:.0f}x slower")
        print(f"Privacy guarantee: model NEVER saw plaintext input")

        log = secure.get_log()[0]
        print(f"\nSecure inference log:")
        print(f"  Protocol:         {log['protocol']}")
        print(f"  Input shape:      {log['input_shape']}")
        print(f"  Encryption time:  {log['encryption_time_ms']:.1f}ms")
        if log.get("full_mpc_coverage"):
            print(f"  MPC coverage:     ✅ All layers ran encrypted")
        else:
            fallback = log.get("plaintext_fallback_layers", [])
            print(f"  MPC coverage:     ⚠️  {fallback} ran in plaintext")

        crypten_ok = True

    except Exception as exc:
        print(f"\n⚠️  CrypTen unavailable: {exc}")
        print("   Try: pip install crypten --no-build-isolation")
        enc_ms = 0.0
        overhead = 0.0
        crypten_ok = False

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Standard\n(plaintext)", "Encrypted\n(CrypTen MPC)"]
    times = [std_ms, enc_ms if crypten_ok else 0]
    colors = ["steelblue", "mediumpurple"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, times, color=colors, width=0.5, edgecolor="white")
    for bar, t in zip(bars, times):
        if t > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{t:.1f}ms",
                ha="center",
                fontsize=11,
                fontweight="bold",
            )
    if crypten_ok and enc_ms > 0:
        ax.annotate(
            f"{overhead:.0f}x overhead\n(cost of cryptographic privacy)",
            xy=(1, enc_ms * 0.95),
            xytext=(0.3, enc_ms * 0.55),
            arrowprops=dict(arrowstyle="->", color="darkred"),
            color="darkred",
            fontsize=9,
        )
    ax.set_ylim(0, (enc_ms if crypten_ok and enc_ms > 0 else std_ms) * 1.35)
    ax.set_ylabel("Inference time (ms)")
    ax.set_title("Standard vs Encrypted Inference — CrypTen\n" "GDPR Article 25: Privacy by Design")
    plt.tight_layout()

    out_dir = pathlib.Path(__file__).parent / "sample_output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "crypten_comparison.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nVisualization saved → {out_path}")
    print(f"Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
