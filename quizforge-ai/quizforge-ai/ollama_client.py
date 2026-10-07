import json
from typing import Any

import requests

from config import safe_connection_diagnostics


class OllamaError(RuntimeError):
    pass


class OllamaAPIError(OllamaError):
    pass


def _http_error_message(status_code: int, response_text: str) -> str:
    if status_code == 401:
        return "Authentication failed: the Ollama API key is invalid or missing."
    if status_code == 402:
        return "Ollama Cloud payment or billing is required for this request."
    if status_code == 429:
        return "Ollama Cloud usage limit reached. Please try again later."
    if 500 <= status_code < 600:
        return "Ollama Cloud is unavailable or returned a server error. Please try again."
    return response_text.strip() or "Ollama Cloud returned an unexpected error."


def _request_chat(
    payload: dict[str, Any],
    api_url: str,
    api_key: str,
    timeout: int,
) -> dict[str, Any]:
    if not api_key:
        raise OllamaAPIError(
            "No Ollama API key is configured. Add it to .env locally or "
            "Streamlit Secrets when deployed."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
    except requests.Timeout as exc:
        raise OllamaAPIError("Ollama Cloud timed out. Please try again.") from exc
    except requests.RequestException as exc:
        raise OllamaAPIError("Ollama Cloud request failed. Please try again.") from exc

    if response.status_code in {401, 402, 429} or 500 <= response.status_code < 600:
        raise OllamaAPIError(_http_error_message(response.status_code, response.text))
    if response.status_code >= 400:
        raise OllamaAPIError(_http_error_message(response.status_code, response.text))

    try:
        result = response.json()
    except ValueError as exc:
        raise OllamaAPIError("Ollama Cloud returned invalid JSON.") from exc

    if not isinstance(result, dict):
        raise OllamaAPIError("Ollama Cloud returned an invalid response.")
    return result


def _get_message_content(result: dict[str, Any]) -> str:
    message = result.get("message")
    if not isinstance(message, dict):
        raise OllamaAPIError("Ollama Cloud returned an invalid response.")
    content = message.get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise OllamaAPIError("Ollama Cloud returned an empty response.")
    return content


def request_quiz(
    payload: dict[str, Any],
    api_url: str,
    api_key: str,
    timeout: int,
) -> dict[str, Any]:
    result = _request_chat(payload, api_url, api_key, timeout)
    content = _get_message_content(result)

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise OllamaAPIError("Ollama Cloud returned invalid JSON content.") from exc


def check_connection(api_url: str, api_key: str, model: str, timeout: int) -> dict[str, object]:
    diagnostics = safe_connection_diagnostics()
    if not api_key:
        return {
            **diagnostics,
            "success": False,
            "message": (
                "No Ollama API key is configured. Add it to .env locally or "
                "Streamlit Secrets when deployed."
            ),
        }

    try:
        result = _request_chat(
            payload={
                "model": model,
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "temperature": 0.0,
                "stream": False,
            },
            api_url=api_url,
            api_key=api_key,
            timeout=timeout,
        )
        _get_message_content(result)
    except OllamaAPIError as exc:
        return {
            **diagnostics,
            "success": False,
            "message": str(exc),
        }

    return {
        **diagnostics,
        "success": True,
        "message": "Ollama Cloud connection succeeded.",
    }


test_connection = check_connection
