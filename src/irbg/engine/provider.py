from __future__ import annotations

import time
from typing import Any

import httpx

from irbg.engine.types import ProviderResponse


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
        app_name: str | None = None,
        site_url: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.app_name = app_name
        self.site_url = site_url
        self._owns_client = client is None
        self.client = client or httpx.Client(timeout=self.timeout)

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def chat(
        self,
        *,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse:
        url = f"{self.base_url}/chat/completions"
        headers = self._build_headers()
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_error: str | None = None
        last_status_code: int | None = None

        for attempt in range(1, self.max_retries + 1):
            started = time.perf_counter()

            try:
                response = self.client.post(
                    url,
                    headers=headers,
                    json=payload,
                )
                latency_ms = int((time.perf_counter() - started) * 1000)
                last_status_code = response.status_code

                if response.status_code == 200:
                    data = response.json()
                    usage = data.get("usage", {})

                    return ProviderResponse(
                        success=True,
                        model_id=model_id,
                        text=self._extract_text(data),
                        input_tokens=int(usage.get("prompt_tokens", 0)),
                        output_tokens=int(usage.get("completion_tokens", 0)),
                        total_tokens=int(usage.get("total_tokens", 0)),
                        latency_ms=latency_ms,
                        status_code=response.status_code,
                        error=None,
                        raw_json=data,
                    )

                error_message = self._extract_error_message(response)
                last_error = error_message

                if self._should_retry(response.status_code, attempt):
                    self._sleep_before_retry(attempt)
                    continue

                return ProviderResponse(
                    success=False,
                    model_id=model_id,
                    text="",
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                    latency_ms=latency_ms,
                    status_code=response.status_code,
                    error=error_message,
                    raw_json={},
                )

            except httpx.HTTPError as exc:
                latency_ms = int((time.perf_counter() - started) * 1000)
                last_error = str(exc)

                if attempt < self.max_retries:
                    self._sleep_before_retry(attempt)
                    continue

                return ProviderResponse(
                    success=False,
                    model_id=model_id,
                    text="",
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                    latency_ms=latency_ms,
                    status_code=last_status_code,
                    error=last_error,
                    raw_json={},
                )

        return ProviderResponse(
            success=False,
            model_id=model_id,
            text="",
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            latency_ms=0,
            status_code=last_status_code,
            error=last_error or "Unknown provider error",
            raw_json={},
        )

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if self.site_url:
            headers["HTTP-Referer"] = self.site_url

        if self.app_name:
            headers["X-Title"] = self.app_name

        return headers

    def _should_retry(self, status_code: int, attempt: int) -> bool:
        retryable_statuses = {429, 500, 502, 503, 504}
        return status_code in retryable_statuses and attempt < self.max_retries

    def _sleep_before_retry(self, attempt: int) -> None:
        time.sleep(self.retry_backoff_seconds * attempt)

    def _extract_text(self, data: dict[str, Any]) -> str:
        choices = data.get("choices", [])
        if not choices:
            return ""

        message = choices[0].get("message", {})
        content = message.get("content", "")

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: list[str] = []

            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)

            return "".join(parts).strip()

        return str(content).strip()

    def _extract_error_message(
        self,
        response: httpx.Response,
    ) -> str:
        try:
            data = response.json()
        except ValueError:
            return f"HTTP {response.status_code}: {response.text}"

        error = data.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str):
                return message

        return f"HTTP {response.status_code}: {data}"
