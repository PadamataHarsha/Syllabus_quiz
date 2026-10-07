from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

import quiz_generator
from config import ASSETS_DIR, SETTINGS, SYLLABUS_DIR, safe_connection_diagnostics
from ollama_client import OllamaAPIError, check_connection
from quiz_validator import ValidationError
from utils import list_syllabus_files, read_text_file

st.set_page_config(page_title="QuizForge", page_icon="🧠", layout="wide")


def load_css() -> None:
    css_path = ASSETS_DIR / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def get_selected_syllabus() -> str:
    subject = st.session_state.get("subject")
    uploaded = st.session_state.get("uploaded_syllabus")
    if uploaded:
        return uploaded
    if not subject:
        return ""
    file_path = SYLLABUS_DIR / subject
    if file_path.exists():
        return read_text_file(file_path)
    return ""


def render_quiz() -> None:
    quiz = st.session_state.get("quiz")
    if not quiz:
        return
    reveal = st.session_state.get("reveal_answers", False)
    for question in quiz["questions"]:
        with st.container():
            st.markdown(f"### {question['id']}. {question['question']}")
            question_type = question.get("type")
            if question_type == "multiple-choice":
                options = question.get("options", [])
                answer = st.radio(
                    "Choose one",
                    options,
                    key=f"answer-{question['id']}",
                    label_visibility="collapsed",
                )
                if reveal:
                    correct = question.get("answer")
                    status = "✅ Correct" if answer == correct else "❌ Incorrect"
                    st.caption(f"{status}. Correct answer: **{correct}**")
            elif question_type == "true-false":
                answer = st.radio(
                    "Choose",
                    [True, False],
                    key=f"answer-{question['id']}",
                    label_visibility="collapsed",
                )
                if reveal:
                    correct = question.get("answer")
                    status = "✅ Correct" if answer == correct else "❌ Incorrect"
                    st.caption(f"{status}. Correct answer: **{correct}**")
            else:
                answer = st.text_input("Your answer", key=f"answer-{question['id']}")
                if reveal:
                    st.caption(f"**Answer:** {question.get('answer', '')}")
            evidence = question.get("evidence", [])
            if evidence:
                st.caption("**Syllabus evidence:**")
                for item in evidence:
                    st.caption(f"• {item}")
            explanation = question.get("explanation")
            if reveal and explanation:
                st.caption(f"**Explanation:** {explanation}")
            st.markdown("---")


def main() -> None:
    load_css()
    st.title("🧠 QuizForge")
    st.caption("Generate quiz questions from the syllabus you select — no outside knowledge.")

    subjects = list_syllabus_files(SYLLABUS_DIR)
    subject_col, upload_col = st.columns([1, 1])
    with subject_col:
        selected_subject = st.selectbox(
            "Subject syllabus",
            ["Select a subject"] + subjects,
            index=0,
            key="subject_selector",
        )
        if selected_subject != "Select a subject":
            st.session_state.subject = selected_subject
        else:
            st.session_state.subject = ""
    with upload_col:
        uploaded = st.file_uploader(
            "Use a temporary syllabus (.txt)",
            type=["txt"],
            key="syllabus_upload",
        )
        if uploaded is not None:
            content = uploaded.read().decode("utf-8")
            st.session_state.uploaded_syllabus = content
            st.caption("Temporary syllabus uploaded and ready for this session.")
        elif "uploaded_syllabus" in st.session_state:
            st.session_state.uploaded_syllabus = ""

    syllabus = get_selected_syllabus()
    if syllabus:
        st.subheader("Syllabus preview")
        st.code(syllabus, language="text")
    else:
        st.info("Choose a built-in syllabus or upload a .txt file.")

    diagnostics = safe_connection_diagnostics()
    with st.container():
        st.subheader("Ollama Cloud")
        st.caption(
            f"Key loaded: {'Yes' if diagnostics['key_loaded'] else 'No'} | "
            f"Key length: {diagnostics['key_length']} | "
            f"URL: {diagnostics['api_url']} | "
            f"Model: {diagnostics['model']}"
        )
        if st.button("Test Ollama Connection", type="primary"):
            if not SETTINGS.ollama_api_key:
                st.error(
                    "No Ollama API key is configured. Add it to .env locally or "
                    "Streamlit Secrets when deployed."
                )
            else:
                with st.spinner("Testing Ollama Cloud connection..."):
                    result = check_connection(
                        SETTINGS.ollama_api_url,
                        SETTINGS.ollama_api_key,
                        SETTINGS.ollama_model,
                        SETTINGS.request_timeout,
                    )
                if result["success"]:
                    st.success(result["message"])
                else:
                    st.error(result["message"])

    with st.form("quiz_form", clear_on_submit=False):
        st.subheader("Quiz settings")
        quiz_title = st.text_input("Quiz title", value="Syllabus Quiz")
        difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
        question_count = st.slider("Question count", min_value=1, max_value=10, value=5)
        question_types = st.multiselect(
            "Question types",
            ["multiple-choice", "true-false", "short-answer"],
            default=["multiple-choice", "true-false", "short-answer"],
        )
        include_explanations = st.checkbox("Include explanations", value=False)
        generate_clicked = st.form_submit_button("Generate", type="primary")
        clear_clicked = st.form_submit_button("Clear")

    if clear_clicked:
        st.session_state.quiz = None
        st.session_state.reveal_answers = False
        st.session_state.generated = False
        st.rerun()

    if generate_clicked and not syllabus.strip():
        st.error("Choose a subject or upload a syllabus before generating.")

    if generate_clicked and syllabus.strip():
        if question_types:
            generation_signature = (
                syllabus,
                quiz_title,
                difficulty,
                question_count,
                tuple(question_types),
                include_explanations,
            )
            if st.session_state.get("generation_signature") == generation_signature:
                st.info("Using the existing quiz for these settings.")
            else:
                with st.spinner("Generating questions from the selected syllabus..."):
                    try:
                        quiz = quiz_generator.generate_quiz(
                            syllabus=syllabus,
                            title=quiz_title,
                            difficulty=difficulty,
                            question_count=question_count,
                            question_types=question_types,
                            include_explanations=include_explanations,
                            model=SETTINGS.ollama_model,
                            api_url=SETTINGS.ollama_api_url,
                            api_key=SETTINGS.ollama_api_key,
                            timeout=SETTINGS.request_timeout,
                        )
                        st.session_state.quiz = quiz
                        st.session_state.reveal_answers = False
                        st.session_state.generated = True
                        st.session_state.generation_signature = generation_signature
                        st.success("Quiz generated successfully.")
                    except (ValidationError, ValueError, OllamaAPIError) as exc:
                        st.error(str(exc))
                        st.session_state.quiz = None
                        st.session_state.generated = False
        else:
            st.warning("Select at least one question type.")

    if st.session_state.get("quiz"):
        st.subheader("Quiz")
        render_quiz()
        actions = st.columns([1, 1, 1])
        with actions[0]:
            if st.button("Reveal answers"):
                st.session_state.reveal_answers = True
                st.rerun()
        with actions[1]:
            quiz_json = st.session_state.quiz
            st.download_button(
                "Download quiz JSON",
                data=__import__("json").dumps(quiz_json, indent=2),
                file_name=f"{quiz_json['title']}.json",
                mime="application/json",
            )
        with actions[2]:
            quiz_md = Path("templates/quiz_template.md").read_text(encoding="utf-8")
            markdown = quiz_md.format(
                title=quiz_json["title"],
                difficulty=quiz_json["difficulty"],
                questions="\n".join(
                    f"{q['id']}. {q['question']}\n"
                    f"  - Options: {', '.join(q.get('options', []))}\n"
                    f"  - Answer: {q['answer']}\n"
                    for q in quiz_json["questions"]
                ),
            )
            st.download_button(
                "Download quiz",
                data=markdown,
                file_name=f"{quiz_json['title']}.md",
                mime="text/markdown",
            )

        answer_key = "\n".join(
            f"{q['id']}. {q.get('answer')}" for q in quiz_json["questions"]
        )
        st.download_button(
            "Download answer key",
            data=answer_key,
            file_name=f"{quiz_json['title']}-answer-key.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
