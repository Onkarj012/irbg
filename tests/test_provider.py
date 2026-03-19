from __future__ import annotations

import httpx

from irbg.engine.provider import OpenRouterClient
from irbg.engine.types import ChatMessage


def test_openrouter_client_chat_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")

        return httpx.Response(
            status_code=200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "pong",
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)

    provider = OpenRouterClient(
        api_key="test-key",
        client=client,
    )

    try:
        response = provider.chat(
            model_id="openai/gpt-4o",
            system_prompt="You are concise.",
            user_prompt="Reply with pong.",
            temperature=0.0,
            max_tokens=32,
        )
    finally:
        provider.close()

    assert response.success is True
    assert response.text == "pong"
    assert response.total_tokens == 12
    assert response.status_code == 200


def test_openrouter_client_chat_messages_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = request.read().decode()
        assert "messages" in payload

        return httpx.Response(
            status_code=200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "final response",
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 5,
                    "total_tokens": 25,
                },
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)

    provider = OpenRouterClient(
        api_key="test-key",
        client=client,
    )

    try:
        response = provider.chat_messages(
            model_id="openai/gpt-4o",
            messages=[
                ChatMessage(role="system", content="You are concise."),
                ChatMessage(role="user", content="Initial prompt."),
                ChatMessage(role="assistant", content="Initial answer."),
                ChatMessage(role="user", content="Reconsider."),
            ],
            temperature=0.0,
            max_tokens=64,
        )
    finally:
        provider.close()

    assert response.success is True
    assert response.text == "final response"
    assert response.total_tokens == 25
