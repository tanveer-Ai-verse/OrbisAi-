# 🪐 OrbisAi — Research Intelligence & Innovation Auditor

OrbisAi helps students ingest research papers quickly and pressure-test the novelty of their own project ideas before committing months of work to them.

## Features

### 📚 Research Intelligence Engine
- Upload one or many PDF / TXT research papers at once
- **Three-Layer Analysis** per paper:
  - **Layer 1 — The Hook:** a 3-sentence summary (Problem, Solution, Why it matters)
  - **Layer 2 — The Flow:** a step-by-step methodology breakdown
  - **Layer 3 — The Critique:** 3 specific limitations or gaps, identified by AI
- **Study Guide Generator:** one click exports a Markdown study guide with key takeaways and quiz questions

### 🧭 Innovation Auditor (Idea Novelty Engine)
- Describe a project idea and get a **Novelty Percentage (0–100%)**, grounded in a live arXiv search rather than the model's guess alone
- **Pivot Suggestions:** if novelty is below 60%, OrbisAi proactively suggests 3 specific research gaps or technical pivots
- **Debate Mode:** toggle it on to argue back — OrbisAi becomes a Devil's Advocate that stress-tests your project's scope

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain (`langchain-groq`, `langchain-text-splitters`) |
| LLM | Groq API — `openai/gpt-oss-120b` |
| PDF parsing | `pypdf` |
| Literature search | `arxiv` (official arXiv API, no key required) |
| Novelty scoring signal | `faiss-cpu` + `scikit-learn` (TF-IDF similarity between your idea and retrieved abstracts) |
| Structured output | Pydantic schemas via LangChain's `with_structured_output` |

## ⚠️ A note on the model

Groq deprecated `llama-3.3-70b-versatile` (and `llama-3.1-8b-instant`) on **August 16, 2026**. Since this affects any of your other apps still pinned to that model, OrbisAi ships with **`openai/gpt-oss-120b`** — Groq's official recommended replacement, with a comparable 131K-token context window and stronger reasoning. This is a single constant near the top of `app.py`:

```python
GROQ_MODEL = "openai/gpt-oss-120b"
```

Change it there if you'd rather use `qwen/qwen3.6-27b` (Groq's other suggested replacement) or a future model — nothing else in the app needs to change.

## Project Structure

```
orbisai/
├── app.py              # Full application (single file, modular functions)
├── requirements.txt    # Pinned, production-ready dependencies
└── README.md
```

`app.py` is organized into clear sections: theme/CSS, Pydantic schemas, LLM client factories, text extraction, arXiv + similarity search, analysis pipelines, session state, and the two-tab UI — each a self-contained set of functions rather than one long script.

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url> orbisai
cd orbisai
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get a free Groq API key

Sign up at [console.groq.com](https://console.groq.com) and generate an API key.

### 3. Add the key to Streamlit secrets

**Local development** — create `.streamlit/secrets.toml` in the project root (this file should never be committed to git):

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

**Streamlit Community Cloud** — after deploying, go to your app → **Settings → Secrets**, and paste the same line into the secrets editor.

### 4. Run locally

```bash
streamlit run app.py
```

## Deploying to Streamlit Community Cloud

1. Push this repo (with `app.py` and `requirements.txt` at the root) to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select your repo, branch, and `app.py` as the entry point.
3. Before or right after the first deploy, open **Settings → Secrets** and add:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
4. Click **Reboot app** if you added the secret after the first deploy.

No `packages.txt` is needed — every dependency here ships prebuilt Linux wheels, so builds are fast.

## How the Innovation Auditor works

1. Your idea is converted into a concise arXiv search query by the LLM.
2. OrbisAi searches arXiv live and retrieves the most relevant abstracts.
3. A TF-IDF + FAISS similarity pass scores lexical closeness between your idea and each abstract — this gives the LLM concrete evidence instead of asking it to "guess" novelty from memory.
4. The LLM combines that evidence with its own reasoning to produce a novelty score, verdict, and — if the score is below 60% — three concrete pivot suggestions.
5. Debate Mode reuses that same context so the AI can argue consistently with the verdict it already gave.

## Troubleshooting

| Issue | Fix |
|---|---|
| `GROQ_API_KEY not found` on startup | Add the key to `.streamlit/secrets.toml` (local) or Settings → Secrets (Cloud), then rerun/reboot. |
| `faiss-cpu` fails to build on Cloud | Rare on standard Streamlit Cloud images, but if it happens, add a `packages.txt` file containing `libgomp1` to the repo root and redeploy. |
| arXiv search returns nothing for a niche idea | Expected for very novel or narrow ideas — OrbisAi still produces a novelty assessment using the LLM's own reasoning, just without external evidence. |
| A PDF produces no extracted text | It's likely a scanned/image-only PDF with no text layer; OCR isn't included in this build. Try a text-based PDF or a `.txt` file instead. |
| Analysis feels slow on very long papers | `openai/gpt-oss-120b` is a reasoning model, so it "thinks" before answering — this trades a little latency for more reliable analysis. |

## License

This project is provided as-is for educational and personal use.
