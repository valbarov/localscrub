"""Minimal Ollama HTTP client — the default local model backend.

Stdlib-only on purpose: the base install stays pure-Python (ADR 0001), and
stage 2 plus corpus diversification share this one client. Alternative
backends (vLLM, Bedrock) implement the same ``generate`` shape later.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

DEFAULT_MODEL = "qwen3:14b"
DEFAULT_HOST = "http://localhost:11434"


class OllamaError(RuntimeError):
    """The Ollama server could not be reached or returned garbage."""


class OllamaClient:
    """Blocking single-prompt completion against a local Ollama server."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        *,
        schema: dict[str, object] | None = None,
        seed: int | None = None,
        temperature: float | None = None,
        think: bool = False,
    ) -> str:
        """Return the model's completion; ``schema`` forces JSON output."""
        payload: dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": think,
        }
        options: dict[str, object] = {}
        if seed is not None:
            options["seed"] = seed
        if temperature is not None:
            options["temperature"] = temperature
        if options:
            payload["options"] = options
        if schema is not None:
            payload["format"] = schema

        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.load(response)
        except (OSError, urllib.error.URLError, ValueError) as exc:
            raise OllamaError(f"Ollama request to {self.host} failed: {exc}") from exc
        reply = data.get("response")
        if not isinstance(reply, str):
            raise OllamaError(f"unexpected Ollama response shape: {list(data)}")
        return reply
