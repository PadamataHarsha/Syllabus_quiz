from __future__ import annotations

from typing import Any

SUPPORTED_TYPES = {"multiple-choice", "true-false", "short-answer"}


class ValidationError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _question_error(question: dict[str, Any], message: str) -> str:
    index = question.get("id", "unknown")
    return f"Question {index}: {message}"


def validate_quiz(quiz: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(quiz, dict):
        raise ValidationError("Quiz payload must be an object.")

    title = quiz.get("title")
    _require(isinstance(title, str) and title.strip(), "Quiz title is required.")
    questions = quiz.get("questions")
    _require(isinstance(questions, list) and questions, "At least one question is required.")

    seen_ids: set[int | str] = set()
    errors: list[str] = []
    validated: list[dict[str, Any]] = []

    for question in questions:
        if not isinstance(question, dict):
            errors.append("Each question must be an object.")
            continue

        question_id = question.get("id")
        if question_id in seen_ids:
            errors.append(_question_error(question, "duplicate question ID detected."))
        else:
            seen_ids.add(question_id)

        question_type = question.get("type")
        if question_type not in SUPPORTED_TYPES:
            errors.append(_question_error(question, "unsupported question type."))

        prompt = question.get("question")
        if not isinstance(prompt, str) or not prompt.strip():
            errors.append(_question_error(question, "Question text is required."))

        evidence = question.get("evidence")
        if not isinstance(evidence, list) or not evidence or not all(str(item).strip() for item in evidence):
            errors.append(_question_error(question, "evidence is required."))

        answer = question.get("answer")
        if question_type == "multiple-choice":
            options = question.get("options")
            if not isinstance(options, list) or len(options) != 4:
                errors.append(_question_error(question, "exactly four unique options are required."))
            elif len(set(str(option).strip() for option in options)) != 4:
                errors.append(_question_error(question, "options must be unique."))
            elif answer not in options:
                errors.append(_question_error(question, "answer is not matching one of the options."))
        elif question_type == "true-false":
            if answer not in (True, False):
                errors.append(_question_error(question, "answer must be true or false."))
        elif question_type == "short-answer":
            if not isinstance(answer, str) or not answer.strip():
                errors.append(_question_error(question, "short-answer text is required."))

        if not errors or not errors[-1].startswith("Question"):
            validated.append(question)

    if errors:
        raise ValidationError("; ".join(errors))

    result = {"valid": True, "errors": [], "quiz": quiz}
    return result
