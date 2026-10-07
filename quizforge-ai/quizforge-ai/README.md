# QuizForge

QuizForge generates syllabus-bound quizzes with Ollama Cloud. The app uses the selected syllabus or a temporary `.txt` syllabus only; it never uses outside knowledge.

## Setup

```bash
cd /workspaces/LLM
python3 -m venv llmpro
source llmpro/bin/activate
pip install -r requirements.txt
```

For local use, add your Ollama API key to `.env` before generating quizzes. For Streamlit Community Cloud, add the settings under **App settings → Secrets**. The API key is never displayed.

## Run

```bash
cd /workspaces/LLM
source llmpro/bin/activate
streamlit run app.py
```

## Environment

The application reads:

- `OLLAMA_API_KEY`
- `OLLAMA_API_URL`
- `OLLAMA_MODEL`
- `REQUEST_TIMEOUT`

Environment variables take precedence over Streamlit Secrets, followed by safe defaults. Local `.env` values are loaded with `python-dotenv`.

For Streamlit Community Cloud, add the following TOML in the app's Secrets settings and replace the placeholder with your own key:

```toml
OLLAMA_API_KEY = "your-ollama-api-key"
OLLAMA_API_URL = "https://ollama.com/api/chat"
OLLAMA_MODEL = "gemma4:31b"
REQUEST_TIMEOUT = 120
```

Use `.env.example` as the template for local configuration; keep `.env` private.
