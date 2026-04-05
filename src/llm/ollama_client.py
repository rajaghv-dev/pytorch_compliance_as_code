"""
HTTP client for Ollama at http://localhost:11434.

Provides generate, embed, and model-listing capabilities with retry logic,
timeout handling, structured JSON parsing, and GPU thermal protection.

GPU THERMAL PROTECTION
----------------------
Every generate() call checks the GPU temperature via the module-level
gpu_monitor singleton before sending the request to Ollama.  If the GPU
is too hot (≥ 85°C), the call waits until it cools before proceeding.
This prevents thermal throttling and protects the hardware during long
LLM enrichment runs.

Ollama manages its own GPU memory allocation; we only gate the request
submission rate from this side.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import requests

from ..gpu_monitor import gpu_monitor

logger = logging.getLogger("pct.llm.ollama_client")


class OllamaError(Exception):
    """Raised when Ollama call fails after all retries."""
    pass


class OllamaClient:
    """HTTP client for Ollama API. Handles retries, timeouts, and JSON parsing."""

    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 120):
        """
        Initialise the Ollama HTTP client.

        Parameters
        ----------
        base_url : str
            Base URL of the Ollama server.
        timeout : int
            Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.logger = logging.getLogger("ollama_client")

    def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        format: str = "json",
        max_retries: int = 3,
    ) -> dict:
        """Call Ollama /api/generate. Return parsed JSON response. Retry 3x on failure.

        Parameters
        ----------
        model : str
            Model name (e.g. "phi4", "mistral-nemo").
        prompt : str
            User prompt text.
        system : str
            System prompt text.
        temperature : float
            Sampling temperature.
        format : str
            Response format — "json" requests JSON output from the model.
        max_retries : int
            Number of retry attempts on failure.

        Returns
        -------
        dict
            If format=="json", the parsed JSON object from the response field.
            Otherwise {"response": raw_string, "model": model}.

        Raises
        ------
        OllamaError
            If all retries are exhausted.
        """
        url = f"{self.base_url}/api/generate"
        # Thinking models (qwen3.5, etc.) return empty responses when
        # format="json" is set because the thinking tokens aren't valid JSON.
        # For these models, omit the format constraint and parse JSON from text.
        is_thinking_model = any(t in model for t in ("qwen3", "deepseek"))
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "options": {"temperature": temperature},
            "stream": False,
        }
        if format == "json" and not is_thinking_model:
            payload["format"] = format

        last_error: Exception | None = None

        for attempt in range(1, max_retries + 1):
            # Gate each LLM call on GPU temperature.
            # If the GPU is too hot, wait_if_hot() blocks until it cools.
            # Ollama itself runs on the GPU, so this protects against
            # sustained thermal load during batch LLM enrichment.
            gpu_monitor.wait_if_hot()

            start_time = time.time()
            try:
                resp = requests.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                elapsed = time.time() - start_time

                data = resp.json()
                response_text = data.get("response", "")

                self.logger.info(
                    "Ollama generate model=%s prompt_len=%d response_time=%.2fs",
                    model, len(prompt), elapsed,
                )

                # Parse the response field as JSON if json format was requested
                if format == "json":
                    try:
                        # Strip <think>...</think> blocks from thinking models
                        # (e.g. qwen3.5) before parsing JSON.
                        clean_text = self._strip_thinking(response_text)
                        parsed = json.loads(clean_text)
                        return parsed
                    except json.JSONDecodeError as e:
                        self.logger.warning(
                            "WARN: JSON parse error on attempt %d/%d: %s — response: %s",
                            attempt, max_retries, e, response_text[:200],
                        )
                        last_error = e
                        if attempt < max_retries:
                            time.sleep(2)
                            continue
                        raise OllamaError(
                            f"Failed to parse JSON response from {model} after "
                            f"{max_retries} retries: {e}"
                        ) from e
                else:
                    return {"response": response_text, "model": model}

            except requests.RequestException as e:
                elapsed = time.time() - start_time
                last_error = e
                self.logger.warning(
                    "WARN: Ollama request failed on attempt %d/%d (%.2fs): %s",
                    attempt, max_retries, elapsed, e,
                )
                if attempt < max_retries:
                    time.sleep(2)
                    continue

        self.logger.error(
            "ERR: All %d retries exhausted for model=%s", max_retries, model,
        )
        raise OllamaError(
            f"Ollama generate failed after {max_retries} retries: {last_error}"
        )

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove <think>...</think> blocks emitted by reasoning models (e.g. qwen3.5).

        Parameters
        ----------
        text : str
            Raw model response that may contain thinking blocks.

        Returns
        -------
        str
            Text with thinking blocks removed, JSON extracted.
        """
        import re
        # Remove all <think>...</think> blocks (including nested/multiline)
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        cleaned = cleaned.strip()

        # Try to extract JSON from the cleaned text
        if cleaned:
            # Find the first JSON array or object
            arr_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if arr_match:
                return arr_match.group(0)
            obj_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if obj_match:
                return obj_match.group(0)

        # Fallback: try the original text
        if not cleaned:
            arr_match = re.search(r"\[.*\]", text, re.DOTALL)
            if arr_match:
                return arr_match.group(0)
            obj_match = re.search(r"\{.*\}", text, re.DOTALL)
            if obj_match:
                return obj_match.group(0)

        return cleaned

    def embed(self, model: str, text: str) -> list[float]:
        """Call Ollama /api/embeddings. Return embedding vector.

        Parameters
        ----------
        model : str
            Embedding model name (e.g. "nomic-embed-text").
        text : str
            Text to embed.

        Returns
        -------
        list[float]
            Embedding vector.

        Raises
        ------
        OllamaError
            If the request fails.
        """
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": model, "prompt": text}

        start_time = time.time()
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            elapsed = time.time() - start_time

            data = resp.json()
            embedding = data.get("embedding", [])

            self.logger.info(
                "Ollama embed model=%s text_len=%d dims=%d response_time=%.2fs",
                model, len(text), len(embedding), elapsed,
            )
            return embedding

        except requests.RequestException as e:
            elapsed = time.time() - start_time
            self.logger.error(
                "ERR: Ollama embed failed (%.2fs): %s", elapsed, e,
            )
            raise OllamaError(f"Ollama embed failed: {e}") from e

    def list_models(self) -> list[str]:
        """Return list of available model names from /api/tags.

        Returns
        -------
        list[str]
            Model name strings.

        Raises
        ------
        OllamaError
            If the request fails.
        """
        url = f"{self.base_url}/api/tags"
        try:
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            self.logger.info("Ollama lists %d models", len(models))
            return models
        except requests.RequestException as e:
            self.logger.error("ERR: Failed to list Ollama models: %s", e)
            raise OllamaError(f"Failed to list models: {e}") from e

    def is_available(self) -> bool:
        """Check if Ollama is running and reachable.

        Returns
        -------
        bool
            True if Ollama responds to a health check.
        """
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def check_model_available(self, model: str) -> bool:
        """Check if a specific model is pulled.

        Parameters
        ----------
        model : str
            Model name to check.

        Returns
        -------
        bool
            True if the model is available locally.
        """
        try:
            models = self.list_models()
            # Check both exact match and prefix match (e.g. "phi4" matches "phi4:latest")
            for m in models:
                if m == model or m.startswith(f"{model}:"):
                    return True
            return False
        except OllamaError:
            return False
