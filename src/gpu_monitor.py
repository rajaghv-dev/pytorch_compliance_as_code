"""
GPU detection and thermal management for the PyTorch Compliance Toolkit.

WHY THIS FILE EXISTS
--------------------
Several phases of the pipeline can use the GPU for faster processing:
  - Phase 2 (extract): REBEL-large relation extraction runs on GPU.
  - Phase 6 (LLM-enrich): Ollama uses the GPU for all LLM inference.
  - FAISS semantic search uses GPU-accelerated indexing if available.

However, laptop and workstation GPUs can overheat under sustained load.
This module monitors the GPU temperature and automatically pauses
GPU-intensive work when the temperature approaches the safety limit,
then resumes when the GPU has cooled down.

TEMPERATURE THRESHOLDS (adjustable in GpuMonitor.THRESHOLDS)
-------------------------------------------------------------
    WARN  = 80°C   → log a warning; slow down request rate
    PAUSE = 85°C   → stop submitting GPU work; wait for cooldown
    COOL  = 78°C   → resume GPU work after a pause

USAGE
-----
    from src.gpu_monitor import GpuMonitor

    # At startup: detect GPU and start temperature monitoring.
    monitor = GpuMonitor()
    monitor.log_device_info()

    # Before any GPU-intensive call:
    monitor.wait_if_hot()   # blocks until temperature is safe

    # As a context manager (preferred for LLM calls):
    with monitor.gpu_task("phi4 mapping validation"):
        result = client.generate(...)

    # Disable GPU for a block (e.g. when running on CPU-only machine):
    if monitor.has_gpu:
        run_gpu_inference()
    else:
        run_cpu_inference()
"""

from __future__ import annotations

import logging
import subprocess
import time
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger("pct.gpu_monitor")


class GpuMonitor:
    """
    Detects GPU availability and monitors temperature.

    All methods are safe to call on CPU-only machines — they simply
    return sensible defaults (has_gpu=False, temperature=0).
    """

    # Temperature thresholds in degrees Celsius.
    # These are conservative defaults suitable for laptop GPUs.
    # Desktop GPUs with better cooling can handle higher values.
    THRESHOLDS = {
        "warn":    80,   # log a warning at this temperature
        "pause":   85,   # stop submitting GPU work above this temperature
        "cool":    78,   # resume GPU work after a mid-run pause
        "pre_llm": 53,   # required starting temperature before LLM phase begins
    }

    # How long to wait between temperature checks during a cooldown (seconds).
    POLL_INTERVAL_SECONDS = 10

    # Maximum time to wait for cooldown before giving up and using CPU (seconds).
    MAX_WAIT_SECONDS = 300  # 5 minutes

    # Maximum time to wait for the pre-LLM cool-down (seconds).
    # 60°C from 76°C on an RTX 3080 Ti can take 15–20 minutes idle.
    PRE_LLM_MAX_WAIT_SECONDS = 1800  # 30 minutes

    def __init__(self) -> None:
        """
        Initialise the GPU monitor.

        Tries to contact nvidia-smi at startup to detect the GPU.
        Falls back gracefully to CPU mode if nvidia-smi is not available.
        """
        self._gpu_info: dict = {}
        self.has_gpu: bool = False

        self._detect_gpu()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def temperature(self) -> float:
        """
        Return the current GPU temperature in degrees Celsius.

        Returns 0.0 if no GPU is available or if nvidia-smi fails.
        """
        if not self.has_gpu:
            return 0.0
        return self._query_temperature()

    def log_device_info(self) -> None:
        """Log a one-line summary of the GPU (or CPU fallback)."""
        if not self.has_gpu:
            logger.info("GPU: not detected — all inference will run on CPU")
            return

        info = self._gpu_info
        logger.info(
            "GPU: %s  |  VRAM: %s MiB  |  CUDA: %s  |  Temp: %s°C",
            info.get("name", "unknown"),
            info.get("memory_total_mib", "?"),
            info.get("cuda_version", "?"),
            self.temperature(),
        )

    def wait_if_hot(self) -> bool:
        """
        Block until the GPU temperature is below the PAUSE threshold.

        Call this before submitting a GPU-intensive task.  If the GPU is
        not hot, this function returns immediately.

        Returns
        -------
        bool
            True  if it is safe to proceed with GPU work.
            False if the maximum wait time was exceeded and the GPU is
                  still hot — caller should fall back to CPU.
        """
        if not self.has_gpu:
            return True   # No GPU → always safe to proceed (on CPU).

        temp = self.temperature()
        if temp < self.THRESHOLDS["pause"]:
            return True   # Temperature is fine.

        # GPU is too hot; wait for it to cool down.
        logger.warning(
            "GPU temperature is %d°C (≥ %d°C threshold) — "
            "pausing GPU work until it cools to %d°C …",
            temp,
            self.THRESHOLDS["pause"],
            self.THRESHOLDS["cool"],
        )

        waited = 0
        while waited < self.MAX_WAIT_SECONDS:
            time.sleep(self.POLL_INTERVAL_SECONDS)
            waited += self.POLL_INTERVAL_SECONDS
            temp = self.temperature()

            if temp <= self.THRESHOLDS["cool"]:
                logger.info(
                    "GPU cooled to %d°C — resuming GPU work", temp
                )
                return True

            logger.info(
                "GPU still at %d°C (target: ≤%d°C) — waiting … (%ds elapsed)",
                temp, self.THRESHOLDS["cool"], waited,
            )

        # Still too hot after MAX_WAIT_SECONDS.
        logger.error(
            "GPU temperature did not drop below %d°C after %d seconds — "
            "falling back to CPU for this task",
            self.THRESHOLDS["cool"],
            self.MAX_WAIT_SECONDS,
        )
        return False

    def wait_until_cool(
        self,
        target_temp: int | None = None,
        max_wait: int | None = None,
        poll_interval: int = 30,
    ) -> bool:
        """
        Block until the GPU temperature drops to *target_temp* degrees Celsius.

        Unlike ``wait_if_hot()`` — which only triggers when the GPU is already
        critically hot — this method is a *pre-flight* check for long-running
        GPU workloads (e.g. the LLM phase).  Sustained LLM inference on a
        laptop GPU will push temperatures from 76°C to 90–95°C within minutes.
        Starting from a lower baseline gives more headroom before thermal
        throttling kicks in and slows down phi4 / mistral-nemo inference.

        Parameters
        ----------
        target_temp : int | None
            Target temperature in °C.  Defaults to ``THRESHOLDS["pre_llm"]``
            (60°C).
        max_wait : int | None
            Maximum seconds to wait before giving up.  Defaults to
            ``PRE_LLM_MAX_WAIT_SECONDS`` (30 minutes).
        poll_interval : int
            Seconds between temperature checks.  Default: 30s.

        Returns
        -------
        bool
            True  — temperature reached target; safe to start GPU work.
            False — max_wait exceeded; GPU still above target.  Caller
                    should proceed anyway (GPU may be stable enough) but
                    should expect possible thermal throttling.
        """
        if not self.has_gpu:
            return True  # No GPU — nothing to wait for.

        target = target_temp if target_temp is not None else self.THRESHOLDS["pre_llm"]
        max_w  = max_wait   if max_wait   is not None else self.PRE_LLM_MAX_WAIT_SECONDS

        temp = self.temperature()
        if temp <= target:
            logger.info(
                "GPU temperature %d°C ≤ target %d°C — ready to start LLM phase",
                temp, target,
            )
            return True

        logger.info(
            "GPU temperature %d°C is above target %d°C — "
            "waiting for GPU to cool before starting LLM phase "
            "(max wait: %ds, poll every %ds) …",
            temp, target, max_w, poll_interval,
        )

        waited = 0
        while waited < max_w:
            time.sleep(poll_interval)
            waited += poll_interval
            temp = self.temperature()

            if temp <= target:
                logger.info(
                    "GPU cooled to %d°C — target %d°C reached after %ds; "
                    "starting LLM phase",
                    temp, target, waited,
                )
                return True

            logger.info(
                "GPU pre-LLM cooldown: %d°C → target %d°C  "
                "(%ds / %ds elapsed)",
                temp, target, waited, max_w,
            )

        # Timed out — log and let the caller decide whether to proceed.
        final_temp = self.temperature()
        logger.warning(
            "GPU pre-LLM cooldown timed out after %ds: temperature is %d°C "
            "(target was %d°C).  Starting LLM phase anyway — "
            "thermal throttling may slow down inference.",
            max_w, final_temp, target,
        )
        return False

    @contextmanager
    def gpu_task(self, task_name: str = "gpu task") -> Generator[None, None, None]:
        """
        Context manager that gates a GPU-intensive block behind a
        temperature check and logs the task name and duration.

        Usage:
            with monitor.gpu_task("phi4 mapping validation"):
                result = client.generate(model="phi4", ...)

        If the GPU is too hot, this waits until it cools before
        entering the block.  If it cannot cool in time, the block
        still runs — the caller is responsible for using CPU instead.
        """
        safe = self.wait_if_hot()
        if not safe:
            logger.warning(
                "gpu_task('%s'): GPU still hot — block will run "
                "(caller should have checked wait_if_hot() return value)",
                task_name,
            )

        start = time.perf_counter()

        # Log temperature at entry.
        if self.has_gpu:
            logger.debug(
                "GPU task starting: %s  (temp=%d°C)",
                task_name, self.temperature(),
            )

        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            if self.has_gpu:
                temp = self.temperature()
                if temp >= self.THRESHOLDS["warn"]:
                    logger.warning(
                        "GPU task '%s' finished in %.1fs — "
                        "temperature is now %d°C (threshold=%d°C)",
                        task_name, elapsed, temp, self.THRESHOLDS["warn"],
                    )
                else:
                    logger.debug(
                        "GPU task '%s' finished in %.1fs (temp=%d°C)",
                        task_name, elapsed, temp,
                    )

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #

    def get_faiss_device(self) -> str:
        """
        Return the device string for FAISS index creation.

        Returns "gpu" if a GPU is available and not too hot, else "cpu".
        This is used by the semantic search module to choose the right
        FAISS index factory.
        """
        if not self.has_gpu:
            return "cpu"
        if self.temperature() >= self.THRESHOLDS["pause"]:
            logger.warning(
                "GPU too hot (%d°C) — using CPU for FAISS indexing",
                self.temperature(),
            )
            return "cpu"
        return "gpu"

    def get_torch_device(self) -> str:
        """
        Return the PyTorch device string ("cuda" or "cpu").

        Use this when loading HuggingFace models (e.g. REBEL-large):
            device = monitor.get_torch_device()
            model = AutoModelForSeq2SeqLM.from_pretrained(name).to(device)
        """
        if not self.has_gpu:
            return "cpu"
        if self.temperature() >= self.THRESHOLDS["pause"]:
            logger.warning(
                "GPU too hot (%d°C) for torch model loading — using CPU",
                self.temperature(),
            )
            return "cpu"
        return "cuda"

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _detect_gpu(self) -> None:
        """
        Try to call nvidia-smi to detect GPU presence and basic info.

        Sets self.has_gpu = True and populates self._gpu_info if a GPU
        is found.  Silently sets has_gpu = False otherwise.
        """
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,temperature.gpu,"
                    "utilization.gpu,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                logger.debug(
                    "nvidia-smi returned non-zero exit code %d — no GPU",
                    result.returncode,
                )
                return

            line = result.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]

            if len(parts) < 6:
                logger.debug("nvidia-smi output unexpected: %s", line)
                return

            self._gpu_info = {
                "index":            parts[0],
                "name":             parts[1],
                "temperature":      int(parts[2]),
                "utilization_pct":  parts[3],
                "memory_total_mib": parts[4],
                "driver_version":   parts[5],
            }

            # Try to get the CUDA version from a separate query.
            try:
                cuda_result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5,
                )
                self._gpu_info["cuda_version"] = cuda_result.stdout.strip()
            except Exception:
                self._gpu_info["cuda_version"] = "unknown"

            self.has_gpu = True
            logger.info(
                "GPU detected: %s  (VRAM: %s MiB, driver: %s)",
                self._gpu_info["name"],
                self._gpu_info["memory_total_mib"],
                self._gpu_info["driver_version"],
            )

        except FileNotFoundError:
            # nvidia-smi is not installed → no GPU.
            logger.debug("nvidia-smi not found — GPU not available")
        except subprocess.TimeoutExpired:
            logger.warning("nvidia-smi timed out — assuming no GPU")
        except Exception as exc:
            logger.debug("GPU detection failed: %s — assuming no GPU", exc)

    def _query_temperature(self) -> float:
        """
        Query the current GPU temperature via nvidia-smi.

        Returns 0.0 on failure so the caller can safely compare against
        thresholds without crashing.
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return float(result.stdout.strip().splitlines()[0])
        except Exception as exc:
            logger.debug("Temperature query failed: %s", exc)
        return 0.0


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
# A single GpuMonitor instance is shared across the whole application.
# Import it wherever you need GPU-aware behaviour:
#
#     from src.gpu_monitor import gpu_monitor
#
#     with gpu_monitor.gpu_task("rebel-large inference"):
#         embeddings = model(inputs)

gpu_monitor = GpuMonitor()
