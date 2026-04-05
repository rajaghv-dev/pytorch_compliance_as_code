"""
Tests for src/gpu_monitor.py — GPU detection and thermal management.

WHAT IS TESTED
--------------
- GpuMonitor() initialises without crashing on any machine.
- temperature() returns a non-negative number.
- wait_if_hot() returns True when the GPU is cool.
- get_torch_device() returns "cuda" or "cpu".
- get_faiss_device() returns "gpu" or "cpu".
- gpu_task() context manager executes the wrapped block.
- Threshold constants exist and form a valid cooldown chain:
    cool < warn < pause

HOW TO RUN
----------
    pytest tests/test_gpu_monitor.py -v
"""

from __future__ import annotations

import pytest

from src.gpu_monitor import GpuMonitor


class TestGpuMonitorInit:

    def test_initialises_without_crash(self):
        """GpuMonitor() should never raise, even on a CPU-only machine."""
        monitor = GpuMonitor()
        # has_gpu is either True or False; both are valid.
        assert isinstance(monitor.has_gpu, bool)

    def test_temperature_is_non_negative(self):
        """temperature() must return a number ≥ 0."""
        monitor = GpuMonitor()
        temp = monitor.temperature()
        assert temp >= 0

    def test_temperature_is_plausible(self):
        """temperature() should not report absurd values (> 150°C)."""
        monitor = GpuMonitor()
        temp = monitor.temperature()
        assert temp < 150, f"Suspicious temperature: {temp}°C"


class TestGpuMonitorThresholds:

    def test_thresholds_form_valid_chain(self):
        """cool < warn < pause must always hold."""
        t = GpuMonitor.THRESHOLDS
        assert t["cool"] < t["warn"], "cool threshold must be below warn"
        assert t["warn"] < t["pause"], "warn threshold must be below pause"

    def test_thresholds_are_positive(self):
        """All threshold values must be positive integers."""
        for name, value in GpuMonitor.THRESHOLDS.items():
            assert value > 0, f"Threshold '{name}' is not positive: {value}"


class TestGpuMonitorWaitIfHot:

    def test_wait_if_hot_returns_true_when_cool(self):
        """On a cool (or absent) GPU, wait_if_hot() should return True immediately."""
        monitor = GpuMonitor()
        # At room temperature / idle, the GPU should be well below 85°C.
        result = monitor.wait_if_hot()
        assert result is True

    def test_wait_if_hot_returns_true_without_gpu(self):
        """Without a GPU, wait_if_hot() should always return True."""
        monitor = GpuMonitor()
        monitor.has_gpu = False   # Force CPU-only mode.
        assert monitor.wait_if_hot() is True


class TestGpuMonitorDeviceHelpers:

    def test_get_torch_device_returns_valid_string(self):
        """get_torch_device() must return 'cuda' or 'cpu'."""
        monitor = GpuMonitor()
        device = monitor.get_torch_device()
        assert device in ("cuda", "cpu"), f"Unexpected device: {device!r}"

    def test_get_faiss_device_returns_valid_string(self):
        """get_faiss_device() must return 'gpu' or 'cpu'."""
        monitor = GpuMonitor()
        device = monitor.get_faiss_device()
        assert device in ("gpu", "cpu"), f"Unexpected device: {device!r}"

    def test_no_gpu_forces_cpu(self):
        """When has_gpu=False, both helpers must return the CPU string."""
        monitor = GpuMonitor()
        monitor.has_gpu = False
        assert monitor.get_torch_device() == "cpu"
        assert monitor.get_faiss_device() == "cpu"


class TestGpuTaskContextManager:

    def test_gpu_task_executes_block(self):
        """The code inside a gpu_task() block must be executed."""
        monitor = GpuMonitor()
        executed = []

        with monitor.gpu_task("unit test block"):
            executed.append(True)

        assert executed == [True]

    def test_gpu_task_propagates_exceptions(self):
        """Exceptions inside gpu_task() should propagate normally."""
        monitor = GpuMonitor()
        with pytest.raises(ValueError, match="test error"):
            with monitor.gpu_task("failing block"):
                raise ValueError("test error")
