# ============================================================
# OrbisAi — Research Intelligence & Innovation Auditor
# ============================================================
# A focused, production-ready Streamlit app for Streamlit Community Cloud.
# Stack: Streamlit + LangChain + langchain-groq (Groq API)
#
# Two tools:
#   1) Research Intelligence Engine — upload papers, get a 3-layer
#      breakdown (Hook / Flow / Critique), export a study guide.
#   2) Innovation Auditor — score the novelty of a project idea against
#      live arXiv search results, get pivot suggestions, debate the verdict.
# ============================================================

import hashlib
import html as html_lib
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import faiss
import arxiv
from pypdf import PdfReader
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================
# CONFIG
# ============================================================
APP_NAME = "OrbisAi"
APP_ICON = "🪐"

# Groq deprecated `llama-3.3-70b-versatile` on 2026-08-16.
# `openai/gpt-oss-120b` is Groq's official recommended replacement:
# production-tier, 131K context, strong reasoning. Change this one
# constant to swap models later.
GROQ_MODEL = "openai/gpt-oss-120b"

# Based on your 8,000 token limit, reducing this will prevent 413 errors.
MAX_PAPER_CHARS = 20_000 

NOVELTY_THRESHOLD = 60
ARXIV_MAX_RESULTS = 8

st.set_page_config(
    page_title="OrbisAi | Research Intelligence",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# THEME — preserved glassmorphism / neon cyberpunk CSS
# ============================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&family=Share+Tech+Mono&display=swap');

:root {
  --neon-blue: #00d4ff;
  --neon-purple: #b400ff;
  --neon-pink: #ff006e;
  --neon-cyan: #00fff9;
  --neon-green: #00ff88;
  --bg-dark: #020010;
  --glass: rgba(0, 212, 255, 0.05);
  --glass-border: rgba(0, 212, 255, 0.2);
}

html, body, [class*="css"] {
  font-family: 'Rajdhani', sans-serif;
  background-color: var(--bg-dark) !important;
  color: #c0e8ff !important;
}

.stApp {
  background: radial-gradient(ellipse at 20% 50%, rgba(0,50,120,0.3) 0%, transparent 60%),
              radial-gradient(ellipse at 80% 20%, rgba(120,0,180,0.2) 0%, transparent 60%),
              radial-gradient(ellipse at 50% 100%, rgba(0,180,255,0.1) 0%, transparent 60%),
              #020010;
  min-height: 100vh;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(0,5,30,0.98) 0%, rgba(5,0,40,0.98) 100%) !important;
  border-right: 1px solid var(--glass-border) !important;
  box-shadow: 4px 0 30px rgba(0,212,255,0.08);
}

h1 { font-family: 'Orbitron', monospace !important; color: var(--neon-cyan) !important;
     text-shadow: 0 0 20px var(--neon-blue), 0 0 40px rgba(0,212,255,0.3); }
h2, h3 { font-family: 'Orbitron', monospace !important; color: var(--neon-blue) !important;
          text-shadow: 0 0 10px rgba(0,212,255,0.5); }

.hero { text-align: center; padding: 0.5rem 0 1.5rem 0; }
.hero-title { font-family: 'Orbitron', monospace; font-size: 2.6rem; font-weight: 900;
  color: var(--neon-cyan); text-shadow: 0 0 20px var(--neon-blue), 0 0 40px rgba(0,212,255,0.3); }
.hero-sub { font-family: 'Rajdhani', sans-serif; color: rgba(192,232,255,0.65);
  letter-spacing: 3px; font-size: 0.9rem; text-transform: uppercase; margin-top: -8px; }

.metric-card {
  background: var(--glass);
  border: 1px solid var(--glass-border);
  border-radius: 12px;
  padding: 1.2rem 1.5rem;
  text-align: center;
  backdrop-filter: blur(12px);
  box-shadow: 0 0 20px rgba(0,212,255,0.1), inset 0 1px 0 rgba(255,255,255,0.05);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.metric-card:hover { transform: translateY(-4px); box-shadow: 0 8px 30px rgba(0,212,255,0.25); }
.metric-value { font-family: 'Orbitron', monospace; font-size: 2.2rem;
  font-weight: 900; color: var(--neon-cyan); text-shadow: 0 0 15px var(--neon-blue); }
.metric-label { font-size: 0.85rem; color: rgba(192,232,255,0.7);
  text-transform: uppercase; letter-spacing: 2px; margin-top: 4px; }

.paper-card {
  background: linear-gradient(135deg, rgba(0,10,40,0.9), rgba(5,0,30,0.95));
  border: 1px solid rgba(0,212,255,0.25);
  border-left: 3px solid var(--neon-blue);
  border-radius: 10px;
  padding: 1.1rem 1.3rem;
  margin: 0.6rem 0;
  backdrop-filter: blur(8px);
  transition: all 0.3s ease;
  box-shadow: 0 2px 15px rgba(0,0,0,0.5);
}
.paper-card:hover { border-left-color: var(--neon-cyan); box-shadow: 0 4px 25px rgba(0,212,255,0.2); }
.paper-title { font-family: 'Orbitron', monospace; font-size: 0.82rem;
  color: var(--neon-cyan); margin-bottom: 6px; font-weight: 700; }
.paper-meta  { font-size: 0.78rem; color: rgba(192,232,255,0.6); margin: 3px 0; }
.paper-abstract { font-size: 0.85rem; color: rgba(192,232,255,0.85); line-height: 1.6; margin-top: 6px; }
.paper-card a { color: var(--neon-blue); }

.critique-card {
  background: linear-gradient(135deg, rgba(40,0,20,0.85), rgba(20,0,10,0.9));
  border: 1px solid rgba(255,0,110,0.3);
  border-left: 3px solid var(--neon-pink);
  border-radius: 10px;
  padding: 1rem 1.2rem;
  margin: 0.5rem 0;
  font-size: 0.88rem;
  color: rgba(255,220,235,0.9);
}

.neon-badge {
  display: inline-block; padding: 3px 12px; border-radius: 20px;
  font-size: 0.72rem; font-family: 'Share Tech Mono', monospace;
  text-transform: uppercase; letter-spacing: 1px; margin: 3px 4px 3px 0;
}
.badge-blue   { background: rgba(0,212,255,0.15); border: 1px solid var(--neon-blue); color: var(--neon-blue); }
.badge-purple { background: rgba(180,0,255,0.15); border: 1px solid var(--neon-purple); color: var(--neon-purple); }
.badge-pink   { background: rgba(255,0,110,0.15); border: 1px solid var(--neon-pink); color: var(--neon-pink); }
.badge-green  { background: rgba(0,255,136,0.15); border: 1px solid var(--neon-green); color: var(--neon-green); }

.chat-user {
  background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(0,100,200,0.1));
  border: 1px solid rgba(0,212,255,0.3);
  border-radius: 12px 12px 4px 12px;
  padding: 0.8rem 1rem; margin: 6px 0; text-align: right;
}
.chat-ai {
  background: linear-gradient(135deg, rgba(180,0,255,0.1), rgba(0,0,60,0.8));
  border: 1px solid rgba(180,0,255,0.3);
  border-radius: 12px 12px 12px 4px;
  padding: 0.8rem 1rem; margin: 6px 0;
}
.chat-ai-label { font-family: 'Orbitron', monospace; font-size: 0.68rem;
  color: var(--neon-purple); margin-bottom: 4px; letter-spacing: 1px; }

.stButton > button {
  background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(100,0,255,0.2)) !important;
  border: 1px solid var(--neon-blue) !important;
  color: var(--neon-cyan) !important;
  font-family: 'Orbitron', monospace !important;
  font-size: 0.75rem !important; font-weight: 700 !important;
  letter-spacing: 1px !important; text-transform: uppercase !important;
  border-radius: 6px !important; transition: all 0.3s ease !important;
  box-shadow: 0 0 10px rgba(0,212,255,0.15) !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, rgba(0,212,255,0.35), rgba(100,0,255,0.35)) !important;
  box-shadow: 0 0 20px rgba(0,212,255,0.4) !important; transform: translateY(-2px) !important;
}

.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
[data-testid="stChatInput"] textarea {
  background: rgba(0,10,40,0.8) !important;
  border: 1px solid rgba(0,212,255,0.3) !important;
  color: var(--neon-cyan) !important;
  border-radius: 8px !important;
  font-family: 'Rajdhani', sans-serif !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus { border-color: var(--neon-cyan) !important; box-shadow: 0 0 15px rgba(0,212,255,0.3) !important; }

.stSelectbox > div > div { background: rgba(0,10,40,0.8) !important; border: 1px solid rgba(0,212,255,0.3) !important; border-radius: 8px !important; }

.section-header {
  font-family: 'Orbitron', monospace; font-size: 1rem; color: var(--neon-cyan);
  text-transform: uppercase; letter-spacing: 3px; padding: 0.5rem 0;
  border-bottom: 1px solid rgba(0,212,255,0.3); margin: 1.2rem 0 1rem 0;
  text-shadow: 0 0 10px var(--neon-blue);
}

.sidebar-logo {
  font-family: 'Orbitron', monospace; font-size: 1.3rem; font-weight: 900;
  text-align: center; padding: 0.8rem 0;
  background: linear-gradient(135deg, var(--neon-cyan), var(--neon-purple));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}

.comparison-card {
  background: linear-gradient(135deg, rgba(0,10,40,0.9), rgba(5,0,30,0.95));
  border: 1px solid rgba(0,212,255,0.3);
  border-top: 3px solid var(--neon-purple);
  border-radius: 10px; padding: 1.1rem 1.3rem; margin: 0.6rem 0;
  font-size: 0.88rem; color: rgba(192,232,255,0.9);
}

.novelty-track {
  width: 100%; height: 10px; border-radius: 5px;
  background: rgba(0,212,255,0.08); border: 1px solid rgba(0,212,255,0.2);
  margin: 10px 0 4px 0; overflow: hidden;
}
.novelty-fill {
  height: 100%; border-radius: 5px;
  background: linear-gradient(90deg, var(--neon-purple), var(--neon-cyan));
}

.upload-zone {
  border: 2px dashed rgba(0,212,255,0.4); border-radius: 12px; padding: 2rem;
  text-align: center; background: rgba(0,5,30,0.5); font-size: 0.9rem; color: rgba(192,232,255,0.7);
}

.roadmap-item { display: flex; gap: 12px; align-items: flex-start; padding: 8px 0; border-bottom: 1px solid rgba(0,212,255,0.1); font-size: 0.9rem; }
.roadmap-dot { width: 11px; height: 11px; border-radius: 50%; background: var(--neon-cyan); margin-top: 5px; flex-shrink: 0; box-shadow: 0 0 8px var(--neon-blue); }

.stDataFrame { border: 1px solid rgba(0,212,255,0.2) !important; border-radius: 8px; }
footer { display: none !important; }
#MainMenu { visibility: hidden; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(0,0,20,0.5); }
::-webkit-scrollbar-thumb { background: var(--neon-blue); border-radius: 3px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def esc(text) -> str:
    """HTML-escape any LLM- or upload-derived text before it goes into unsafe_allow_html markup."""
    return html_lib.escape(str(text if text is not None else ""))


# ============================================================
# API KEY
# ============================================================
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    if not GROQ_API_KEY or not str(GROQ_API_KEY).strip():
        raise KeyError
except Exception:
    GROQ_API_KEY = None

if not GROQ_API_KEY:
    st.markdown('<div class="hero"><div class="hero-title">🪐 ORBISAI</div></div>', unsafe_allow_html=True)
    st.error("🔑 **GROQ_API_KEY not found.** OrbisAi needs a Groq API key to run.")
    with st.expander("How to add it", expanded=True):
        st.markdown(
            "**Get a free key:** [console.groq.com](https://console.groq.com)\n\n"
            "**Local development** — create `.streamlit/secrets.toml` in your project root:\n"
            "```toml\nGROQ_API_KEY = \"your-key-here\"\n```\n\n"
            "**Streamlit Community Cloud** — go to your app → **Settings → Secrets** and add the same line."
        )
    st.stop()


# ============================================================
# PYDANTIC SCHEMAS (structured LLM outputs)
# ============================================================
class ThreeLayerAnalysis(BaseModel):
    problem: str = Field(description="One sentence: the core problem or research question the paper addresses.")
    solution: str = Field(description="One sentence: the proposed solution or approach.")
    why_it_matters: str = Field(description="One sentence: why this research matters / its significance.")
    methodology_steps: List[str] = Field(description="4 to 8 short, sequential steps describing how the researchers carried out the work, in order.")
    limitations: List[str] = Field(description="Exactly 3 specific potential limitations or gaps, tied to what this paper actually does.")


class QuizItem(BaseModel):
    question: str = Field(description="A quiz question testing understanding of the paper.")
    answer: str = Field(description="A concise answer to the question.")


class StudyGuideContent(BaseModel):
    key_takeaways: List[str] = Field(description="4 to 6 concise key takeaways from the paper.")
    quiz: List[QuizItem] = Field(description="Exactly 5 quiz questions with answers.")


class SearchQuery(BaseModel):
    query: str = Field(description="A concise 3-8 word arXiv search query capturing the core technical concepts.")


class NoveltyResult(BaseModel):
    novelty_percentage: int = Field(ge=0, le=100, description="Novelty score, 0 = identical to existing work, 100 = highly novel/unexplored.")
    verdict: str = Field(description="A short one-to-three word verdict, e.g. 'Highly Novel', 'Moderate Overlap', 'Largely Derivative'.")
    reasoning: str = Field(description="2-4 sentence explanation grounded in the evidence provided.")
    overlapping_works: List[str] = Field(default_factory=list, description="Titles of the existing works that most reduce novelty, if any.")


class PivotSuggestions(BaseModel):
    pivots: List[str] = Field(description="Exactly 3 specific, actionable research gaps or technical pivots, 1-2 sentences each.")


# ============================================================
# LLM CLIENTS
# ============================================================
@st.cache_resource(show_spinner=False)
def get_analysis_llm() -> ChatGroq:
    return ChatGroq(model=GROQ_MODEL, groq_api_key=GROQ_API_KEY, temperature=0.2, timeout=60, max_retries=2)


@st.cache_resource(show_spinner=False)
def get_creative_llm() -> ChatGroq:
    return ChatGroq(model=GROQ_MODEL, groq_api_key=GROQ_API_KEY, temperature=0.5, timeout=60, max_retries=2)


@st.cache_resource(show_spinner=False)
def get_debate_llm() -> ChatGroq:
    return ChatGroq(model=GROQ_MODEL, groq_api_key=GROQ_API_KEY, temperature=0.75, timeout=60, max_retries=2)


def run_structured(llm: ChatGroq, schema_cls, system_prompt: str, user_prompt: str):
    """Invoke the LLM with a structured-output schema. Returns (result, error_message)."""
    try:
        structured_llm = llm.with_structured_output(schema_cls)
        result = structured_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        return result, None
    except Exception as e:
        return None, str(e)


def run_debate_turn(llm: ChatGroq, idea_text: str, novelty: NoveltyResult, history: list, user_message: str):
    system_prompt = f"""You are OrbisAi's Devil's Advocate in DEBATE MODE. Your job is to rigorously stress-test a student's project idea and the novelty verdict already given — like a tough but fair thesis committee member, not a hostile critic.

Original idea: {idea_text}
Novelty verdict: {novelty.novelty_percentage}% ({novelty.verdict}). {novelty.reasoning}

Rules:
- Challenge weak points directly with specific, hard follow-up questions.
- Do not simply agree to be agreeable. If the student makes a genuinely good counter-argument, acknowledge it honestly, then keep pushing on whatever is still weak.
- Stay constructive: the goal is a stronger project, not discouragement.
- Keep replies to 3-6 sentences unless the question truly requires more."""
    messages = [SystemMessage(content=system_prompt)]
    for m in history:
        messages.append(HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]))
    messages.append(HumanMessage(content=user_message))
    try:
        response = llm.invoke(messages)
        return response.content, None
    except Exception as e:
        return None, str(e)


# ============================================================
# TEXT EXTRACTION
# ============================================================
def get_file_hash(uploaded_file) -> str:
    uploaded_file.seek(0)
    content = uploaded_file.read()
    uploaded_file.seek(0)
    return hashlib.md5(content).hexdigest()


def extract_text_from_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    uploaded_file.seek(0)
    try:
        if name.endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            pages = []
            for page in reader.pages:
                try:
                    pages.append(page.extract_text() or "")
                except Exception:
                    continue
            return "\n".join(pages).strip()
        elif name.endswith(".txt"):
            raw = uploaded_file.read()
            try:
                return raw.decode("utf-8").strip()
            except UnicodeDecodeError:
                return raw.decode("latin-1", errors="ignore").strip()
        return ""
    except Exception:
        return ""


def prepare_paper_text(raw_text: str, max_chars: int = MAX_PAPER_CHARS) -> Tuple[str, bool]:
    """Return text safely sized for the LLM context window. If truncation is needed,
    keep the start (intro/methods) and end (results/conclusion) rather than a naive head-cut."""
    if len(raw_text) <= max_chars:
        return raw_text, False

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100, separators=["\n\n", "\n", ". ", " ", ""])
    chunks = splitter.split_text(raw_text)

    head_budget = int(max_chars * 0.7)
    tail_budget = max_chars - head_budget

    head, head_len = [], 0
    for c in chunks:
        if head_len + len(c) > head_budget:
            break
        head.append(c)
        head_len += len(c)

    tail, tail_len = [], 0
    for c in reversed(chunks):
        if tail_len + len(c) > tail_budget:
            break
        tail.append(c)
        tail_len += len(c)
    tail.reverse()

    combined = "\n".join(head) + "\n\n[... middle section omitted for length ...]\n\n" + "\n".join(tail)
    return combined, True


# ============================================================
# ARXIV SEARCH + SIMILARITY
# ============================================================
@st.cache_data(show_spinner=False, ttl=3600, max_entries=200)
def search_arxiv_papers(query: str, max_results: int = ARXIV_MAX_RESULTS) -> List[dict]:
    client = arxiv.Client(page_size=max_results, delay_seconds=1, num_retries=2)
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    results = []
    for r in client.results(search):
        results.append({
            "title": r.title.strip(),
            "abstract": r.summary.strip().replace("\n", " "),
            "authors": ", ".join(a.name for a in list(r.authors)[:3]),
            "year": r.published.year,
            "url": r.entry_id,
        })
    return results


def extract_search_query(llm: ChatGroq, idea_text: str) -> str:
    result, err = run_structured(
        llm, SearchQuery,
        system_prompt="You convert a student's project idea into a concise, effective arXiv search query.",
        user_prompt=f"Project idea:\n{idea_text}\n\nGenerate a concise arXiv search query (3-8 words) using the core technical terms.",
    )
    if result and result.query.strip():
        return result.query.strip()
    return " ".join(idea_text.split()[:12])


def compute_similarity_scores(idea_text: str, documents: List[str]) -> np.ndarray:
    if not documents:
        return np.array([])
    corpus = [idea_text] + documents
    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
    try:
        tfidf = vectorizer.fit_transform(corpus).toarray().astype("float32")
    except ValueError:
        return np.zeros(len(documents), dtype="float32")
    faiss.normalize_L2(tfidf)
    idea_vec, doc_vecs = tfidf[0:1], tfidf[1:]
    index = faiss.IndexFlatIP(doc_vecs.shape[1])
    index.add(doc_vecs)
    scores, _ = index.search(idea_vec, len(documents))
    return scores[0]


# ============================================================
# ANALYSIS PIPELINES
# ============================================================
ANALYSIS_SYSTEM_PROMPT = (
    "You are OrbisAi's Research Intelligence Engine. You read academic papers and distill them for "
    "busy students with precision and clarity. Be concrete and specific to THIS paper — never generic. "
    "Base every claim strictly on the provided paper text."
)


def build_analysis_prompt(paper_text: str) -> str:
    return f"""Analyze this research paper text and produce a three-layer breakdown.

PAPER TEXT:
{paper_text}

Requirements:
- problem / solution / why_it_matters: one sentence each.
- methodology_steps: 4-8 short sequential steps describing HOW the work was carried out, in order.
- limitations: exactly 3 specific limitations or gaps tied to what this paper actually does (not generic caveats)."""


def analyze_paper(paper_text: str):
    clean_text, truncated = prepare_paper_text(paper_text)
    result, err = run_structured(get_analysis_llm(), ThreeLayerAnalysis, ANALYSIS_SYSTEM_PROMPT, build_analysis_prompt(clean_text))
    return result, truncated, err


def generate_study_guide(paper_title: str, analysis: ThreeLayerAnalysis):
    system_prompt = "You are OrbisAi's Study Guide Generator, creating exam-ready study material from research papers."
    user_prompt = f"""Paper: {paper_title}

Problem: {analysis.problem}
Solution: {analysis.solution}
Why it matters: {analysis.why_it_matters}
Methodology: {'; '.join(analysis.methodology_steps)}
Limitations: {'; '.join(analysis.limitations)}

Generate 4-6 key takeaways and exactly 5 quiz questions (with concise answers) a student could use to study this paper."""
    return run_structured(get_analysis_llm(), StudyGuideContent, system_prompt, user_prompt)


def build_study_guide_markdown(paper_title: str, analysis: ThreeLayerAnalysis, guide: StudyGuideContent) -> str:
    lines = [
        f"# Study Guide: {paper_title}",
        "",
        f"_Generated by OrbisAi on {datetime.now().strftime('%Y-%m-%d')}_",
        "",
        "## The Hook",
        f"- **Problem:** {analysis.problem}",
        f"- **Solution:** {analysis.solution}",
        f"- **Why it matters:** {analysis.why_it_matters}",
        "",
        "## Methodology",
    ]
    lines += [f"{i}. {step}" for i, step in enumerate(analysis.methodology_steps, 1)]
    lines += ["", "## Key Limitations"]
    lines += [f"- {lim}" for lim in analysis.limitations]
    lines += ["", "## Key Takeaways"]
    lines += [f"- {kt}" for kt in guide.key_takeaways]
    lines += ["", "## Quiz Yourself", ""]
    for i, item in enumerate(guide.quiz, 1):
        lines += [f"**Q{i}. {item.question}**", "", f"> **Answer:** {item.answer}", ""]
    return "\n".join(lines)


def assess_novelty(idea_text: str, arxiv_results: List[dict], sim_scores: np.ndarray):
    evidence_lines = []
    for i, paper in enumerate(arxiv_results):
        sim = f"{sim_scores[i] * 100:.0f}%" if len(sim_scores) > i else "N/A"
        evidence_lines.append(f'- "{paper["title"]}" ({paper["year"]}) — lexical similarity: {sim}\n  Abstract: {paper["abstract"][:300]}')
    evidence_text = "\n".join(evidence_lines) if evidence_lines else "No closely related arXiv papers were found."

    system_prompt = (
        "You are OrbisAi's Innovation Auditor, an expert research advisor who evaluates how novel a "
        "student's project idea is relative to existing published research. Be rigorous and evidence-based. "
        "Do not inflate novelty scores — a genuinely derivative idea should score low."
    )
    user_prompt = f"""Student's project idea:
{idea_text}

Related existing research found via arXiv search:
{evidence_text}

Assess novelty (consider conceptual overlap, not just keyword overlap). Provide novelty_percentage (0-100), a short verdict, clear reasoning, and the specific overlapping works/titles that most reduce novelty, if any."""
    return run_structured(get_creative_llm(), NoveltyResult, system_prompt, user_prompt)


def generate_pivots(idea_text: str, novelty: NoveltyResult):
    system_prompt = (
        "You are OrbisAi's Innovation Auditor. The student's idea has low novelty. Propose specific, "
        "technically concrete pivots that preserve their core interest but differentiate the work."
    )
    user_prompt = f"""Idea: {idea_text}

Novelty assessment: {novelty.novelty_percentage}% — {novelty.reasoning}
Overlapping works: {', '.join(novelty.overlapping_works) if novelty.overlapping_works else 'N/A'}

Suggest exactly 3 specific, actionable research gaps or technical pivots, each 1-2 sentences, concrete enough to act on (not generic advice)."""
    return run_structured(get_creative_llm(), PivotSuggestions, system_prompt, user_prompt)


# ============================================================
# SESSION STATE
# ============================================================
def init_session_state():
    defaults = {
        "papers": {},
        "active_paper": None,
        "novelty_idea": "",
        "novelty_result": None,
        "novelty_pivots": None,
        "arxiv_hits": [],
        "debate_mode": False,
        "debate_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(f'<div class="sidebar-logo">{APP_ICON} ORBISAI</div>', unsafe_allow_html=True)
    st.caption("Research Intelligence & Innovation Auditor")
    st.markdown("---")
    st.markdown("**📚 Research Intelligence**  \nUpload papers → 3-layer breakdown → study guide export.")
    st.markdown("**🧭 Innovation Auditor**  \nDescribe an idea → novelty score vs. live arXiv search → debate the verdict.")
    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.metric("Papers loaded", len(st.session_state.papers))
    c2.metric("Analyzed", sum(1 for p in st.session_state.papers.values() if p.get("analysis")))
    st.markdown("---")
    st.caption(f"⚡ Powered by Groq · `{GROQ_MODEL}`")
    if st.button("🔄 Reset Session", width="stretch"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ============================================================
# HERO
# ============================================================
st.markdown(
    f'<div class="hero"><div class="hero-title">{APP_ICON} ORBISAI</div>'
    f'<div class="hero-sub">Research Intelligence &amp; Innovation Auditor</div></div>',
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["📚  Research Intelligence Engine", "🧭  Innovation Auditor"])

# ============================================================
# TAB 1 — RESEARCH INTELLIGENCE ENGINE
# ============================================================
with tab1:
    st.markdown('<div class="section-header">Upload Research Papers</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT research papers", type=["pdf", "txt"],
        accept_multiple_files=True, label_visibility="collapsed",
    )

    if uploaded_files:
        for f in uploaded_files:
            fh = get_file_hash(f)
            if fh not in st.session_state.papers:
                text = extract_text_from_file(f)
                if not text.strip():
                    st.warning(f"Couldn't extract text from **{esc(f.name)}** — it may be a scanned/image-only PDF.")
                    continue
                st.session_state.papers[fh] = {"name": f.name, "text": text, "analysis": None, "study_guide": None, "truncated": False}
        if st.session_state.papers and st.session_state.active_paper not in st.session_state.papers:
            st.session_state.active_paper = next(iter(st.session_state.papers))

    if not st.session_state.papers:
        st.markdown(
            '<div class="upload-zone">📄 Drop one or more research papers (PDF or TXT) above to get a '
            "3-layer breakdown — The Hook, The Flow, and The Critique — plus a downloadable study guide.</div>",
            unsafe_allow_html=True,
        )
    else:
        summary_rows = [
            {"Paper": p["name"], "Words": f'{len(p["text"].split()):,}', "Status": "✅ Analyzed" if p["analysis"] else "⏳ Not analyzed"}
            for p in st.session_state.papers.values()
        ]
        st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

        names = {fh: p["name"] for fh, p in st.session_state.papers.items()}
        keys_list = list(names.keys())
        default_idx = keys_list.index(st.session_state.active_paper) if st.session_state.active_paper in keys_list else 0
        selected = st.selectbox("Select a paper", options=keys_list, index=default_idx, format_func=lambda k: names[k])
        st.session_state.active_paper = selected
        paper = st.session_state.papers[selected]

        col_a, col_b = st.columns([3, 1])
        with col_a:
            analyze_clicked = st.button("🔍 Run Three-Layer Analysis", width="stretch")
        with col_b:
            remove_clicked = st.button("🗑️ Remove", width="stretch")

        if remove_clicked:
            del st.session_state.papers[selected]
            st.session_state.active_paper = None
            st.rerun()

        if analyze_clicked:
            with st.spinner("Reading the paper and running the three-layer breakdown..."):
                result, truncated, err = analyze_paper(paper["text"])
            if err:
                st.error(f"Analysis failed: {err}")
            else:
                paper["analysis"] = result
                paper["truncated"] = truncated
                paper["study_guide"] = None
                st.rerun()

        if paper["analysis"]:
            analysis: ThreeLayerAnalysis = paper["analysis"]
            if paper.get("truncated"):
                st.caption("⚠️ This paper was long — analysis is based on the beginning and end of the extracted text.")

            st.markdown('<div class="section-header">Layer 1 — The Hook</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="paper-card"><div class="paper-title">🎯 PROBLEM</div><div class="paper-abstract">{esc(analysis.problem)}</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="paper-card"><div class="paper-title">💡 SOLUTION</div><div class="paper-abstract">{esc(analysis.solution)}</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="paper-card"><div class="paper-title">⭐ WHY IT MATTERS</div><div class="paper-abstract">{esc(analysis.why_it_matters)}</div></div>', unsafe_allow_html=True)

            st.markdown('<div class="section-header">Layer 2 — The Flow</div>', unsafe_allow_html=True)
            steps_html = "".join(f'<div class="roadmap-item"><div class="roadmap-dot"></div><div>{esc(step)}</div></div>' for step in analysis.methodology_steps)
            st.markdown(f'<div class="paper-card">{steps_html}</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-header">Layer 3 — The Critique</div>', unsafe_allow_html=True)
            for lim in analysis.limitations:
                st.markdown(f'<div class="critique-card">⚠️ {esc(lim)}</div>', unsafe_allow_html=True)

            st.markdown("---")
            if st.button("📘 Generate Study Guide"):
                with st.spinner("Building your study guide..."):
                    guide, gerr = generate_study_guide(paper["name"], analysis)
                if gerr:
                    st.error(f"Couldn't generate study guide: {gerr}")
                else:
                    paper["study_guide"] = guide
                    st.rerun()

            if paper.get("study_guide"):
                md = build_study_guide_markdown(paper["name"], analysis, paper["study_guide"])
                safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in paper["name"].rsplit(".", 1)[0])
                st.download_button("⬇️ Download Study Guide (.md)", md.encode("utf-8"), file_name=f"{safe_name}_study_guide.md", mime="text/markdown")
                with st.expander("Preview Study Guide"):
                    st.markdown(md)

# ============================================================
# TAB 2 — INNOVATION AUDITOR
# ============================================================
with tab2:
    st.markdown('<div class="section-header">Describe Your Project Idea</div>', unsafe_allow_html=True)
    idea_input = st.text_area(
        "Project idea", value=st.session_state.novelty_idea, height=140,
        placeholder="Describe your project idea: what it does, how, and for whom...",
        label_visibility="collapsed",
    )
    analyze_idea_clicked = st.button("🧭 Analyze Novelty", type="primary")

    if analyze_idea_clicked:
        if not idea_input.strip():
            st.warning("Please describe your idea first.")
        else:
            st.session_state.novelty_idea = idea_input.strip()
            st.session_state.debate_history = []
            st.session_state.debate_mode = False
            with st.spinner("Searching current research and evaluating novelty..."):
                analysis_llm = get_analysis_llm()
                query = extract_search_query(analysis_llm, st.session_state.novelty_idea)
                try:
                    arxiv_hits = search_arxiv_papers(query)
                except Exception:
                    st.warning("Live arXiv search is temporarily unavailable — proceeding with the AI's own knowledge instead.")
                    arxiv_hits = []

                sim_scores = compute_similarity_scores(st.session_state.novelty_idea, [p["abstract"] for p in arxiv_hits]) if arxiv_hits else np.array([])
                result, err = assess_novelty(st.session_state.novelty_idea, arxiv_hits, sim_scores)

                if err:
                    st.error(f"Novelty assessment failed: {err}")
                else:
                    st.session_state.novelty_result = result
                    st.session_state.arxiv_hits = arxiv_hits
                    st.session_state.novelty_pivots = None
                    if result.novelty_percentage < NOVELTY_THRESHOLD:
                        pivots, perr = generate_pivots(st.session_state.novelty_idea, result)
                        if not perr:
                            st.session_state.novelty_pivots = pivots
            st.rerun()

    if st.session_state.novelty_result:
        result: NoveltyResult = st.session_state.novelty_result
        st.markdown("---")

        score_col, verdict_col = st.columns([1, 2])
        with score_col:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{result.novelty_percentage}%</div><div class="metric-label">Novelty Score</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="novelty-track"><div class="novelty-fill" style="width:{result.novelty_percentage}%;"></div></div>', unsafe_allow_html=True)
        with verdict_col:
            st.markdown(f'<div class="paper-card"><div class="paper-title">VERDICT: {esc(result.verdict).upper()}</div><div class="paper-abstract">{esc(result.reasoning)}</div></div>', unsafe_allow_html=True)

        if result.overlapping_works:
            st.markdown('<div class="section-header">Closest Existing Work</div>', unsafe_allow_html=True)
            badges = "".join(f'<span class="neon-badge badge-purple">{esc(w)}</span>' for w in result.overlapping_works)
            st.markdown(badges, unsafe_allow_html=True)

        if st.session_state.arxiv_hits:
            with st.expander(f"📚 {len(st.session_state.arxiv_hits)} related papers found on arXiv"):
                for p in st.session_state.arxiv_hits:
                    st.markdown(
                        f'<div class="paper-card"><div class="paper-title">{esc(p["title"])}</div>'
                        f'<div class="paper-meta">👤 {esc(p["authors"])} &nbsp;·&nbsp; 📅 {p["year"]}</div>'
                        f'<div class="paper-abstract">{esc(p["abstract"][:280])}...</div>'
                        f'<a href="{esc(p["url"])}" target="_blank">View on arXiv →</a></div>',
                        unsafe_allow_html=True,
                    )

        if result.novelty_percentage < NOVELTY_THRESHOLD and st.session_state.novelty_pivots:
            st.markdown('<div class="section-header">💡 Suggested Pivots</div>', unsafe_allow_html=True)
            for pivot in st.session_state.novelty_pivots.pivots:
                st.markdown(f'<div class="comparison-card">{esc(pivot)}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.session_state.debate_mode = st.toggle("⚔️ Debate Mode — challenge the AI's assessment", value=st.session_state.debate_mode)

        if st.session_state.debate_mode:
            st.caption("OrbisAi will play Devil's Advocate and stress-test your project scope.")
            for msg in st.session_state.debate_history:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-user">{esc(msg["content"])}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-ai"><div class="chat-ai-label">⚔️ DEVIL\'S ADVOCATE</div>{esc(msg["content"])}</div>', unsafe_allow_html=True)

            debate_input = st.chat_input("Push back on the assessment...")
            if debate_input:
                st.session_state.debate_history.append({"role": "user", "content": debate_input})
                with st.spinner("Thinking..."):
                    response, derr = run_debate_turn(get_debate_llm(), st.session_state.novelty_idea, result, st.session_state.debate_history[:-1], debate_input)
                if derr:
                    st.error(f"Debate response failed: {derr}")
                    st.session_state.debate_history.pop()
                else:
                    st.session_state.debate_history.append({"role": "assistant", "content": response})
                st.rerun()
