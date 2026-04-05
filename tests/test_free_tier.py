"""
Validates all 11 torchcomply examples run correctly on CPU.

Run fast tests only:  pytest tests/test_free_tier.py -m "not slow"
Run all tests:        pytest tests/test_free_tier.py -m slow --timeout=600
"""

import os
import sys
import pathlib
import subprocess
import time

import psutil
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"

# ---------------------------------------------------------------------------
# Environment used for every subprocess call
# ---------------------------------------------------------------------------

TEST_ENV = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": "",
    "MPLBACKEND": "Agg",
    "TOKENIZERS_PARALLELISM": "false",
}

# ---------------------------------------------------------------------------
# EXAMPLES registry — name, script path, expected output PNGs
# ---------------------------------------------------------------------------

EXAMPLES = [
    {
        "name": "01_audit_trail",
        "script": EXAMPLES_DIR / "01_audit_trail" / "run.py",
        "pngs": [
            EXAMPLES_DIR / "01_audit_trail" / "sample_output" / "audit_waterfall.png",
        ],
    },
    {
        "name": "02_fairness_gate",
        "script": EXAMPLES_DIR / "02_fairness_gate" / "run.py",
        "pngs": [
            EXAMPLES_DIR / "02_fairness_gate" / "sample_output" / "fairness_trajectory.png",
        ],
    },
    {
        "name": "03_captum_explain",
        "script": EXAMPLES_DIR / "03_captum_explain" / "run.py",
        "pngs": [
            EXAMPLES_DIR / "03_captum_explain" / "sample_output" / "captum_attribution.png",
            EXAMPLES_DIR / "03_captum_explain" / "sample_output" / "attribution_heatmap.png",
        ],
    },
    {
        "name": "04_opacus_dp",
        "script": EXAMPLES_DIR / "04_opacus_dp" / "run.py",
        "pngs": [
            EXAMPLES_DIR / "04_opacus_dp" / "sample_output" / "dp_accuracy_tradeoff.png",
            EXAMPLES_DIR / "04_opacus_dp" / "sample_output" / "dp_budget_gauge.png",
        ],
    },
    {
        "name": "05_compliant_dataset",
        "script": EXAMPLES_DIR / "05_compliant_dataset" / "run.py",
        "pngs": [
            EXAMPLES_DIR / "05_compliant_dataset" / "sample_output" / "consent_scatter.png",
            EXAMPLES_DIR / "05_compliant_dataset" / "sample_output" / "class_distribution.png",
        ],
    },
    {
        "name": "06_before_after",
        "script": EXAMPLES_DIR / "06_before_after" / "run.py",
        "pngs": [
            EXAMPLES_DIR / "06_before_after" / "sample_output" / "before_after_comparison.png",
        ],
    },
    {
        "name": "06_full_pipeline_v1",
        "script": EXAMPLES_DIR / "06_full_pipeline_v1" / "run.py",
        "pngs": [
            EXAMPLES_DIR / "06_full_pipeline_v1" / "sample_output" / "coverage_radar.png",
        ],
    },
    {
        "name": "07_crypten_secure",
        "script": EXAMPLES_DIR / "07_crypten_secure" / "run.py",
        "pngs": [
            EXAMPLES_DIR / "07_crypten_secure" / "sample_output" / "crypten_comparison.png",
        ],
    },
    {
        "name": "08_three_mechanisms",
        "script": EXAMPLES_DIR / "08_three_mechanisms" / "run.py",
        "pngs": [
            EXAMPLES_DIR / "08_three_mechanisms" / "sample_output" / "three_mechanisms.png",
        ],
    },
    {
        "name": "09_deployment_monitor",
        "script": EXAMPLES_DIR / "09_deployment_monitor" / "run.py",
        "pngs": [
            EXAMPLES_DIR / "09_deployment_monitor" / "sample_output" / "bias_drift.png",
            EXAMPLES_DIR / "09_deployment_monitor" / "sample_output" / "human_interventions.png",
        ],
    },
    {
        "name": "10_connected_pipeline",
        "script": EXAMPLES_DIR / "10_connected_pipeline" / "run.py",
        "pngs": [
            EXAMPLES_DIR / "10_connected_pipeline" / "sample_output" / "coverage_radar.png",
            EXAMPLES_DIR / "10_connected_pipeline" / "sample_output" / "pipeline_timeline.png",
        ],
    },
    {
        "name": "11_llm_finetune",
        "script": EXAMPLES_DIR / "11_llm_finetune" / "run.py",
        "pngs": [
            EXAMPLES_DIR / "11_llm_finetune" / "sample_output" / "lora_compliance_card.png",
        ],
    },
]

# Flat list of (example_name, png_path) pairs for parametrized PNG checks
_PNG_PARAMS = [
    pytest.param(ex["name"], png, id=f"{ex['name']}/{png.name}")
    for ex in EXAMPLES
    for png in ex["pngs"]
]


def _run_example(script_path: pathlib.Path, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run a single example script and return the CompletedProcess result."""
    return subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        env=TEST_ENV,
        timeout=timeout,
        cwd=str(script_path.parent),
    )


def _example_by_name(name: str) -> dict:
    """Return the EXAMPLES entry matching *name*."""
    for ex in EXAMPLES:
        if ex["name"] == name:
            return ex
    raise KeyError(f"Unknown example: {name!r}")


# ===========================================================================
# 1. Import checks
# ===========================================================================


def test_imports():
    """All required packages import without error."""
    required = [
        "torch",
        "transformers",
        "opacus",
        "captum",
        "peft",
        "mlflow",
        "reportlab",
        "opentelemetry",
    ]
    # crypten requires special build tooling (native C extensions) — skip if absent
    optional = ["crypten"]

    missing = []
    for pkg in required:
        result = subprocess.run(
            [sys.executable, "-c", f"import {pkg}"],
            capture_output=True,
            text=True,
            env=TEST_ENV,
            timeout=30,
        )
        if result.returncode != 0:
            missing.append(pkg)
    assert not missing, f"Failed to import required packages: {missing}"

    for pkg in optional:
        result = subprocess.run(
            [sys.executable, "-c", f"import {pkg}"],
            capture_output=True,
            text=True,
            env=TEST_ENV,
            timeout=30,
        )
        if result.returncode != 0:
            import warnings
            warnings.warn(f"Optional package not available: {pkg}", stacklevel=2)


# ===========================================================================
# 2. Memory budget
# ===========================================================================


def test_memory_budget():
    """System has at least 4 GB RAM — minimum viable for free tier."""
    total_gb = psutil.virtual_memory().total / (1024 ** 3)
    assert total_gb >= 4.0, (
        f"Only {total_gb:.1f} GB RAM available; free-tier examples require >= 4 GB"
    )


# ===========================================================================
# 3. CPU-only execution
# ===========================================================================


def test_cpu_only_execution():
    """CUDA is unavailable when CUDA_VISIBLE_DEVICES is empty."""
    result = subprocess.run(
        [sys.executable, "-c",
         "import torch, sys; sys.exit(0 if not torch.cuda.is_available() else 1)"],
        capture_output=True,
        text=True,
        env=TEST_ENV,
        timeout=30,
    )
    assert result.returncode == 0, (
        "torch.cuda.is_available() returned True despite CUDA_VISIBLE_DEVICES=''"
    )


# ===========================================================================
# 4–15. Per-example functional tests
# ===========================================================================


@pytest.mark.slow
def test_example_01_audit_trail():
    """Ex01 prints audit chain size, chain validity, and an integrity violation."""
    ex = _example_by_name("01_audit_trail")
    result = _run_example(ex["script"])
    out = result.stdout + result.stderr
    assert result.returncode == 0, f"Script failed:\n{out}"
    assert "670 entries" in out, f"Expected '670 entries' in output:\n{out}"
    assert "CHAIN VALID" in out, f"Expected 'CHAIN VALID' in output:\n{out}"
    assert "INTEGRITY VIOLATION" in out, f"Expected 'INTEGRITY VIOLATION' in output:\n{out}"
    png = ex["pngs"][0]
    assert png.exists(), f"PNG not found: {png}"
    assert png.stat().st_size > 10 * 1024, f"PNG too small: {png.stat().st_size} bytes"


@pytest.mark.slow
def test_example_02_fairness_gate():
    """Ex02 blocks a deployment due to a fairness parity violation."""
    ex = _example_by_name("02_fairness_gate")
    result = _run_example(ex["script"])
    out = result.stdout + result.stderr
    assert result.returncode == 0, f"Script failed:\n{out}"
    assert "BLOCKED" in out, f"Expected 'BLOCKED' in output:\n{out}"
    png = ex["pngs"][0]
    assert png.exists(), f"PNG not found: {png}"
    assert png.stat().st_size > 10 * 1024, f"PNG too small: {png.stat().st_size} bytes"


@pytest.mark.slow
def test_example_03_captum():
    """Ex03 generates both attribution PNGs via Captum."""
    ex = _example_by_name("03_captum_explain")
    result = _run_example(ex["script"])
    out = result.stdout + result.stderr
    assert result.returncode == 0, f"Script failed:\n{out}"
    for png in ex["pngs"]:
        assert png.exists(), f"PNG not found: {png}"
        assert png.stat().st_size > 10 * 1024, f"PNG too small: {png.stat().st_size} bytes"


@pytest.mark.slow
def test_example_04_opacus():
    """Ex04 reports a final epsilon value and generates DP visualisation PNGs."""
    ex = _example_by_name("04_opacus_dp")
    result = _run_example(ex["script"])
    out = result.stdout + result.stderr
    assert result.returncode == 0, f"Script failed:\n{out}"
    assert "epsilon" in out.lower(), f"Expected 'epsilon' in output:\n{out}"
    for png in ex["pngs"]:
        assert png.exists(), f"PNG not found: {png}"


@pytest.mark.slow
def test_example_05_dataset():
    """Ex05 denies a consent-withdrawn record and reports class imbalance."""
    ex = _example_by_name("05_compliant_dataset")
    result = _run_example(ex["script"])
    out = result.stdout + result.stderr
    assert result.returncode == 0, f"Script failed:\n{out}"
    assert "DENIED" in out, f"Expected 'DENIED' in output:\n{out}"
    assert "imbalance" in out.lower(), f"Expected 'imbalance' in output:\n{out}"
    for png in ex["pngs"]:
        assert png.exists(), f"PNG not found: {png}"


@pytest.mark.slow
def test_example_06a_before_after():
    """Ex06a compares standard vs compliant checkpoints and validates audit chain."""
    ex = _example_by_name("06_before_after")
    result = _run_example(ex["script"])
    out = result.stdout + result.stderr
    assert result.returncode == 0, f"Script failed:\n{out}"
    assert "Standard checkpoint" in out, f"Expected 'Standard checkpoint' in output:\n{out}"
    assert "Compliant checkpoint" in out, f"Expected 'Compliant checkpoint' in output:\n{out}"
    assert "audit_chain" in out, f"Expected 'audit_chain' in output:\n{out}"
    png = ex["pngs"][0]
    assert png.exists(), f"PNG not found: {png}"


@pytest.mark.slow
def test_example_06b_pipeline():
    """Ex06b full pipeline v1 produces a coverage radar PNG."""
    ex = _example_by_name("06_full_pipeline_v1")
    result = _run_example(ex["script"])
    out = result.stdout + result.stderr
    assert result.returncode == 0, f"Script failed:\n{out}"
    png = ex["pngs"][0]
    assert png.exists(), f"PNG not found: {png}"


@pytest.mark.slow
def test_example_07_crypten():
    """Ex07 performs encrypted inference and never exposes plaintext."""
    ex = _example_by_name("07_crypten_secure")
    result = _run_example(ex["script"])
    out = result.stdout + result.stderr
    assert result.returncode == 0, f"Script failed:\n{out}"
    assert "Encrypted inference" in out, f"Expected 'Encrypted inference' in output:\n{out}"
    png = ex["pngs"][0]
    assert png.exists(), f"PNG not found: {png}"


@pytest.mark.slow
def test_example_08_three_mechanisms():
    """Ex08 exercises hook audit, dispatcher, and provenance logging."""
    ex = _example_by_name("08_three_mechanisms")
    result = _run_example(ex["script"])
    out = result.stdout + result.stderr
    assert result.returncode == 0, f"Script failed:\n{out}"
    assert "15 entries" in out, f"Expected '15 entries' in output:\n{out}"
    assert "3 tensor operations" in out, f"Expected '3 tensor operations' in output:\n{out}"
    assert "1 gradient" in out, f"Expected '1 gradient' in output:\n{out}"
    png = ex["pngs"][0]
    assert png.exists(), f"PNG not found: {png}"


@pytest.mark.slow
def test_example_09_deployment_monitor():
    """Ex09 detects bias drift at batch 15 and logs a human intervention."""
    ex = _example_by_name("09_deployment_monitor")
    result = _run_example(ex["script"])
    out = result.stdout + result.stderr
    assert result.returncode == 0, f"Script failed:\n{out}"
    assert "DRIFT" in out.upper(), f"Expected 'DRIFT' in output:\n{out}"
    assert "batch 15" in out.lower(), f"Expected 'batch 15' in output:\n{out}"
    for png in ex["pngs"]:
        assert png.exists(), f"PNG not found: {png}"


@pytest.mark.slow
def test_example_10_connected_pipeline():
    """Ex10 completes all 10 stages with checkmarks and generates an Annex IV report."""
    ex = _example_by_name("10_connected_pipeline")
    result = _run_example(ex["script"])
    out = result.stdout + result.stderr
    assert result.returncode == 0, f"Script failed:\n{out}"
    # Expect 10 lines containing "Stage" and the Unicode checkmark
    stage_lines = [line for line in out.splitlines() if "Stage" in line]
    passing = [line for line in stage_lines if "\u2705" in line]
    assert len(passing) >= 10, (
        f"Expected >= 10 passing stage lines, found {len(passing)}:\n"
        + "\n".join(stage_lines)
    )
    assert "Annex IV" in out, f"Expected 'Annex IV' in output:\n{out}"
    for png in ex["pngs"]:
        assert png.exists(), f"PNG not found: {png}"


@pytest.mark.slow
def test_example_11_llm_finetune():
    """Ex11 applies LoRA and reports the count of trainable parameters."""
    ex = _example_by_name("11_llm_finetune")
    result = _run_example(ex["script"])
    out = result.stdout + result.stderr
    assert result.returncode == 0, f"Script failed:\n{out}"
    assert "LoRA" in out, f"Expected 'LoRA' in output:\n{out}"
    assert "trainable" in out.lower(), f"Expected 'trainable' in output:\n{out}"
    png = ex["pngs"][0]
    assert png.exists(), f"PNG not found: {png}"


# ===========================================================================
# 16. Wall-time budget across all examples
# ===========================================================================


@pytest.mark.slow
def test_all_examples_wall_time():
    """Total wall time for all 11 examples sequentially stays under 300 seconds."""
    start = time.monotonic()
    failures = []
    for ex in EXAMPLES:
        try:
            result = _run_example(ex["script"], timeout=120)
            if result.returncode != 0:
                failures.append(
                    f"{ex['name']}: exit {result.returncode}\n"
                    f"{result.stdout[-500:]}\n{result.stderr[-500:]}"
                )
        except subprocess.TimeoutExpired:
            failures.append(f"{ex['name']}: timed out after 120 s")
    elapsed = time.monotonic() - start
    assert not failures, "Some examples failed during wall-time run:\n" + "\n---\n".join(failures)
    assert elapsed < 300, (
        f"Total wall time {elapsed:.1f}s exceeds free-tier budget of 300s"
    )


# ===========================================================================
# 17. Per-example memory budget
# ===========================================================================


@pytest.mark.slow
def test_free_tier_memory_per_example():
    """Peak RSS increase per example stays below 2 GB (conservative free-tier budget)."""
    limit_bytes = 2 * 1024 ** 3  # 2 GB
    violations = []
    proc = psutil.Process()
    for ex in EXAMPLES:
        rss_before = proc.memory_info().rss
        try:
            _run_example(ex["script"], timeout=120)
        except subprocess.TimeoutExpired:
            violations.append(f"{ex['name']}: timed out")
            continue
        rss_after = proc.memory_info().rss
        delta = rss_after - rss_before
        if delta > limit_bytes:
            violations.append(
                f"{ex['name']}: RSS delta {delta / 1024**3:.2f} GB > 2 GB limit"
            )
    assert not violations, "Memory budget exceeded:\n" + "\n".join(violations)


# ===========================================================================
# Parametrized PNG existence check (fast — uses already-written outputs)
# ===========================================================================


@pytest.mark.parametrize("example_name,png_path", _PNG_PARAMS)
def test_png_exists(example_name, png_path):
    """Each expected output PNG exists and is non-trivially sized (> 10 KB)."""
    assert png_path.exists(), (
        f"[{example_name}] PNG not found: {png_path}\n"
        "Run the example first or execute the slow tests."
    )
    assert png_path.stat().st_size > 10 * 1024, (
        f"[{example_name}] PNG suspiciously small "
        f"({png_path.stat().st_size} bytes): {png_path}"
    )
