"""Small, dependency-free LLM clients for Termux.

The agent asks providers for JSON. Both clients intentionally use urllib from the
standard library so the project can run in a fresh Termux Python installation.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


class LLMError(RuntimeError):
    """Raised when an LLM request cannot be completed or parsed."""


def _request_json(url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=int(os.getenv("NABD_TIMEOUT", "90"))) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise LLMError(f"Provider HTTP {exc.code}: {detail[:800]}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"Network error: {exc.reason}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Provider returned invalid JSON: {raw[:500]}") from exc


def extract_json(text: str) -> Dict[str, Any]:
    """Parse JSON even when a model wraps it in a markdown code fence."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    raise LLMError(f"Model did not return a JSON object: {text[:500]}")


class LLMClient:
    def __init__(self, provider: str = "auto") -> None:
        requested = provider.lower()
        self.provider = self._select_provider(requested)
        if self.provider == "openai":
            self.model = os.getenv("NABD_OPENAI_MODEL", "gpt-4o-mini")
        else:
            self.model = os.getenv("NABD_GEMINI_MODEL", "gemini-2.0-flash")

    @staticmethod
    def _select_provider(requested: str) -> str:
        if requested in {"openai", "gemini"}:
            return requested
        if requested not in {"auto", ""}:
            raise LLMError("NABD_PROVIDER must be auto, openai, or gemini")
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("GEMINI_API_KEY"):
            return "gemini"
        raise LLMError("Set OPENAI_API_KEY or GEMINI_API_KEY before starting Nabd")

    def complete(self, system: str, user: str) -> str:
        if self.provider == "openai":
            return self._openai(system, user)
        return self._gemini(system, user)

    def complete_json(self, system: str, user: str) -> Dict[str, Any]:
        return extract_json(self.complete(system, user))

    def _openai(self, system: str, user: str) -> str:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise LLMError("OPENAI_API_KEY is not set")
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }
        data = _request_json(
            os.getenv("NABD_OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions"),
            payload,
            {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected OpenAI response: {str(data)[:800]}") from exc

    def _gemini(self, system: str, user: str) -> str:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise LLMError("GEMINI_API_KEY is not set")
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={key}"
        )
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }
        data = _request_json(url, payload, {"Content-Type": "application/json"})
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected Gemini response: {str(data)[:800]}") from exc
