import json
from typing import Any

from ollama_client import OllamaAPIError, request_quiz
from quiz_validator import validate_quiz
from utils import normalize_list


SYSTEM_PROMPT = """You are a quiz generator. Use only the syllabus provided in the user message.
Do not use outside knowledge or invent facts. If the syllabus is insufficient for a requested question, omit that question.
Return only valid JSON matching this schema:
{
  "title": "string",
  "difficulty": "Easy|Medium|Hard",
  "questions": [
    {
      "id": 1,
      "type": "multiple-choice|true-false|short-answer",
      "question": "string",
      "options": ["string", "string", "string", "string"],
      "answer": true|false|"string",
      "evidence": ["string", "string"]
    }
  ]
}
Rules:
- Never use information outside the syllabus.
- Every question must include evidence excerpts from the syllabus.
- Multiple-choice questions require exactly four unique options and an answer matching one option.
- Keep answers concise and factual.
- Use no conversation or markdown.
- Do not include any additional keys.
"""


def build_prompt(syllabus: str, title: str, difficulty: str, question_count: int, question_types: list[str], include_explanations: bool) -> str:
    normalized_types = [item for item in question_types if item]
    section = "\n" if syllabus.strip() else ""
    return (
        f"Create exactly {question_count} quiz questions using only this syllabus:{section}{syllabus}\n"
        f"Title: {title}\n"
        f"Difficulty: {difficulty}\n"
        f"Question types to use: {', '.join(normalized_types) if normalized_types else 'Use all supported types'}\n"
        f"Include explanations: {str(include_explanations).lower()}\n"
        "Do not invent facts. If the syllabus cannot support a question, omit it and return fewer questions."
    )


def generate_quiz(
    syllabus: str,
    title: str,
    difficulty: str,
    question_count: int,
    question_types: list[str],
    include_explanations: bool,
    model: str,
    api_url: str,
    api_key: str,
    timeout: int,
) -> dict[str, Any]:
    if not syllabus.strip():
        raise ValueError("Syllabus is empty. Add syllabus content before generating.")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(syllabus, title, difficulty, question_count, question_types, include_explanations)},
        ],
        "temperature": 0.2,
        "stream": False,
        "format": "json",
    }

    raw = request_quiz(payload, api_url, api_key, timeout)
    if not isinstance(raw, dict):
        raise OllamaAPIError("The Ollama response did not contain a JSON object.")

    validate_quiz(raw)
    if include_explanations:
        return raw
    for question in raw.get("questions", []):
        question.pop("explanation", None)
    return raw
