import pytest
from streamlit.errors import StreamlitSecretNotFoundError

import config
from ollama_client import OllamaAPIError, check_connection, request_quiz
from quiz_validator import ValidationError, validate_quiz


def test_valid_quiz_passes():
    quiz = {
        "title": "Sample Quiz",
        "difficulty": "Easy",
        "questions": [
            {
                "id": 1,
                "type": "multiple-choice",
                "question": "Which topic is covered?",
                "options": ["A", "B", "C", "D"],
                "answer": "B",
                "evidence": ["The syllabus says A."],
            },
            {
                "id": 2,
                "type": "true-false",
                "question": "The syllabus includes the topic.",
                "answer": True,
                "evidence": ["The syllabus states this."],
            },
            {
                "id": 3,
                "type": "short-answer",
                "question": "What is the key idea?",
                "answer": "The key idea",
                "evidence": ["The syllabus describes it."],
            },
        ],
    }

    result = validate_quiz(quiz)

    assert result["valid"] is True
    assert result["errors"] == []


def test_missing_required_fields_are_reported():
    with pytest.raises(ValidationError, match="title"):
        validate_quiz({"questions": []})


def test_duplicate_question_ids_are_rejected():
    quiz = {
        "title": "Sample",
        "questions": [
            {"id": 1, "type": "multiple-choice", "question": "Q1", "options": ["A", "B", "C", "D"], "answer": "A", "evidence": ["Syllabus"]},
            {"id": 1, "type": "multiple-choice", "question": "Q2", "options": ["A", "B", "C", "D"], "answer": "B", "evidence": ["Syllabus"]},
        ],
    }

    with pytest.raises(ValidationError, match="duplicate"):
        validate_quiz(quiz)


def test_mcq_options_must_have_unique_answers():
    quiz = {
        "title": "Sample",
        "questions": [
            {"id": 1, "type": "multiple-choice", "question": "Q1", "options": ["A", "A", "C", "D"], "answer": "A", "evidence": ["Syllabus"]},
        ],
    }

    with pytest.raises(ValidationError, match="unique"):
        validate_quiz(quiz)


def test_answer_must_match_mcq_options():
    quiz = {
        "title": "Sample",
        "questions": [
            {"id": 1, "type": "multiple-choice", "question": "Q1", "options": ["A", "B", "C", "D"], "answer": "E", "evidence": ["Syllabus"]},
        ],
    }

    with pytest.raises(ValidationError, match="matching"):
        validate_quiz(quiz)


def test_question_evidence_is_required():
    quiz = {
        "title": "Sample",
        "questions": [{"id": 1, "type": "multiple-choice", "question": "Q1", "options": ["A", "B", "C", "D"], "answer": "A", "evidence": []}],
    }

    with pytest.raises(ValidationError, match="evidence"):
        validate_quiz(quiz)


def test_questions_type_must_be_supported():
    quiz = {
        "title": "Sample",
        "questions": [{"id": 1, "type": "essay", "question": "Q1", "answer": "A", "evidence": ["Syllabus"]}],
    }

    with pytest.raises(ValidationError, match="type"):
        validate_quiz(quiz)


def test_config_strips_env_values(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "  raw-key  ")
    monkeypatch.setenv("OLLAMA_API_URL", " https://ollama.com/api/chat ")
    monkeypatch.setenv("OLLAMA_MODEL", " model:latest ")
    monkeypatch.setenv("REQUEST_TIMEOUT", " 45 ")

    settings = config.get_settings()

    assert settings.ollama_api_key == "raw-key"
    assert settings.ollama_api_url == "https://ollama.com/api/chat"
    assert settings.ollama_model == "model:latest"
    assert settings.request_timeout == 45


def test_environment_values_take_precedence_over_streamlit_secrets(monkeypatch):
    monkeypatch.setattr(
        config,
        "_read_streamlit_secrets",
        lambda: {
            "OLLAMA_API_KEY": "secret-key",
            "OLLAMA_API_URL": "https://secrets.example/chat",
            "OLLAMA_MODEL": "secrets-model",
            "REQUEST_TIMEOUT": "60",
        },
    )
    monkeypatch.setenv("OLLAMA_API_KEY", "fake-env-key")
    monkeypatch.setenv("OLLAMA_API_URL", "https://env.example/chat")
    monkeypatch.setenv("OLLAMA_MODEL", "env-model")
    monkeypatch.setenv("REQUEST_TIMEOUT", "45")

    settings = config.get_settings()

    assert settings.ollama_api_key == "fake-env-key"
    assert settings.ollama_api_url == "https://env.example/chat"
    assert settings.ollama_model == "env-model"
    assert settings.request_timeout == 45


def test_streamlit_secrets_are_used_when_environment_is_unset(monkeypatch):
    for name in config.DEFAULT_SETTINGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        config,
        "_read_streamlit_secrets",
        lambda: {
            "OLLAMA_API_KEY": "fake-secrets-key",
            "OLLAMA_API_URL": "https://secrets.example/chat",
            "OLLAMA_MODEL": "secrets-model",
            "REQUEST_TIMEOUT": "75",
        },
    )

    settings = config.get_settings()

    assert settings.ollama_api_key == "fake-secrets-key"
    assert settings.ollama_api_url == "https://secrets.example/chat"
    assert settings.ollama_model == "secrets-model"
    assert settings.request_timeout == 75


def test_safe_defaults_are_used_without_environment_or_streamlit_secrets(monkeypatch):
    for name in config.DEFAULT_SETTINGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "_read_streamlit_secrets", lambda: {})

    settings = config.get_settings()

    assert settings.ollama_api_key == ""
    assert settings.ollama_api_url == "https://ollama.com/api/chat"
    assert settings.ollama_model == "gemma4:31b"
    assert settings.request_timeout == 120


def test_missing_streamlit_secrets_are_handled_safely(monkeypatch):
    class MissingSecrets:
        @property
        def secrets(self):
            raise StreamlitSecretNotFoundError("No secrets configured.")

    monkeypatch.setattr(config, "st", MissingSecrets())

    assert config._read_streamlit_secrets() == {}


def test_request_uses_exact_header_and_no_key_leak(monkeypatch):
    captured = {}

    class Response:
        status_code = 200

        def json(self):
            return {"message": {"content": '{"ok": true}'}}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("ollama_client.requests.post", fake_post)

    result = request_quiz({"model": "test-model", "messages": []}, "https://ollama.com/api/chat", "secret-key", 30)

    assert result == {"ok": True}
    assert captured["headers"] == {
        "Authorization": "Bearer secret-key",
        "Content-Type": "application/json",
    }
    assert "secret-key" not in str({
        "key_loaded": True,
        "key_length": len("secret-key"),
        "api_url": "https://ollama.com/api/chat",
        "model": "test-model",
    })


def test_connection_diagnostics_without_exposing_key(monkeypatch):
    class Response:
        status_code = 200

        def json(self):
            return {"message": {"content": '{"ok": true}'}}

    monkeypatch.setattr("ollama_client.requests.post", lambda *args, **kwargs: Response())

    secret_key = "fake-test-key"
    monkeypatch.setenv("OLLAMA_API_KEY", secret_key)
    status = check_connection("https://ollama.com/api/chat", secret_key, "test-model", 30)

    assert status["key_loaded"] is True
    assert status["key_length"] == len(secret_key)
    assert status["api_url"] == "https://ollama.com/api/chat"
    assert status["model"] == config.get_settings().ollama_model
    assert secret_key not in str(status)


def test_missing_connection_key_returns_deployment_guidance(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.setattr(config, "_read_streamlit_secrets", lambda: {})
    monkeypatch.setattr(
        "ollama_client.requests.post",
        lambda *args, **kwargs: pytest.fail("Unexpected network call"),
    )

    result = check_connection(
        "https://ollama.com/api/chat",
        "",
        "test-model",
        30,
    )

    assert result["success"] is False
    assert result["message"] == (
        "No Ollama API key is configured. Add it to .env locally or "
        "Streamlit Secrets when deployed."
    )


def test_connection_accepts_plain_text_model_reply(monkeypatch):
    captured = {}

    class Response:
        status_code = 200

        def json(self):
            return {"message": {"content": "OK"}}

    def fake_post(url, headers, json, timeout):
        captured["payload"] = json
        return Response()

    monkeypatch.setattr("ollama_client.requests.post", fake_post)

    status = check_connection(
        "https://ollama.com/api/chat",
        "test-key",
        "test-model",
        30,
    )

    assert status["success"] is True
    assert status["message"] == "Ollama Cloud connection succeeded."
    assert "format" not in captured["payload"]


def test_request_quiz_still_requires_json_content(monkeypatch):
    class Response:
        status_code = 200

        def json(self):
            return {"message": {"content": "OK"}}

    monkeypatch.setattr("ollama_client.requests.post", lambda *args, **kwargs: Response())

    with pytest.raises(OllamaAPIError, match="invalid JSON content"):
        request_quiz(
            {"model": "test-model", "messages": []},
            "https://ollama.com/api/chat",
            "test-key",
            30,
        )


def test_unauthorized_is_reported_cleanly(monkeypatch):
    class Response:
        status_code = 401
        text = "Unauthorized"

    monkeypatch.setattr("ollama_client.requests.post", lambda *args, **kwargs: Response())

    with pytest.raises(OllamaAPIError, match="401|Authentication"):
        request_quiz({"model": "test-model", "messages": []}, "https://ollama.com/api/chat", "secret-key", 30)
