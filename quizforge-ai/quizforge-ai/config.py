import os
from collections.abc import Mapping
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator
from streamlit.errors import StreamlitSecretNotFoundError

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=False)

PROJECT_ROOT = BASE_DIR
SYLLABUS_DIR = PROJECT_ROOT / "syllabus"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
ASSETS_DIR = PROJECT_ROOT / "assets"

DEFAULT_SETTINGS = {
    "OLLAMA_API_KEY": "",
    "OLLAMA_API_URL": "https://ollama.com/api/chat",
    "OLLAMA_MODEL": "gemma4:31b",
    "REQUEST_TIMEOUT": "120",
}


class Settings(BaseModel):
    ollama_api_key: str = Field(default="")
    ollama_api_url: str = Field(default="https://ollama.com/api/chat")
    ollama_model: str = Field(default="gemma4:31b")
    request_timeout: int = Field(default=120)

    @field_validator("request_timeout")
    @classmethod
    def validate_timeout(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("REQUEST_TIMEOUT must be greater than zero")
        return value


def _read_streamlit_secrets() -> Mapping[str, object]:
    try:
        return st.secrets
    except StreamlitSecretNotFoundError:
        return {}


def _get_setting_value(
    name: str,
    secrets: Mapping[str, object],
) -> str:
    environment_value = os.getenv(name)
    if environment_value and environment_value.strip():
        return environment_value.strip()

    secret_value = secrets.get(name)
    if secret_value is not None and str(secret_value).strip():
        return str(secret_value).strip()

    return DEFAULT_SETTINGS[name]


def get_settings() -> Settings:
    secrets = _read_streamlit_secrets()
    try:
        request_timeout = int(_get_setting_value("REQUEST_TIMEOUT", secrets))
    except ValueError:
        request_timeout = int(DEFAULT_SETTINGS["REQUEST_TIMEOUT"])

    try:
        return Settings(
            ollama_api_key=_get_setting_value("OLLAMA_API_KEY", secrets),
            ollama_api_url=_get_setting_value("OLLAMA_API_URL", secrets),
            ollama_model=_get_setting_value("OLLAMA_MODEL", secrets),
            request_timeout=request_timeout,
        )
    except (TypeError, ValueError, ValidationError):
        return Settings()


def safe_connection_diagnostics() -> dict[str, object]:
    settings = get_settings()
    return {
        "key_loaded": bool(settings.ollama_api_key),
        "key_length": len(settings.ollama_api_key),
        "api_url": settings.ollama_api_url,
        "model": settings.ollama_model,
    }


SETTINGS = get_settings()
