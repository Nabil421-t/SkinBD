"""
RAG retriever: keyword search + Gemini answer.

Retrieval hierarchy (fastest → most semantic):
  1. Disk cache  — SHA-256 keyed JSON (rag/rag_cache.json); instant repeat answers
  2. ChromaDB    — persistent HNSW vector store (rag/chroma_db/); best semantic match
  3. FAISS       — flat binary index (rag/faiss_index.bin); fallback vector search
  4. BM25        — TF-IDF keyword search over .txt files; always available, no downloads

Provides: load_index(), retrieve(), answer_question()
"""

import json
import logging
import math
import os
import re
from collections import Counter

from dotenv import load_dotenv
from google import genai

from rag.cache import get as _cache_get, put as _cache_put, make_key as _cache_key
from rag.chroma_store import build_from_chunks as _chroma_build, query as _chroma_query, is_available as _chroma_available

# Graph RAG — lazy import; graceful fallback if kuzu not installed / graph not built
try:
    from graph.store import (
        get_symptoms as _graph_symptoms,
        get_related as _graph_related,
        is_available as _graph_available,
    )
    _GRAPH_IMPORT_OK = True
except ImportError:
    _GRAPH_IMPORT_OK = False
    def _graph_symptoms(_): return []   # type: ignore[misc]
    def _graph_related(_): return []    # type: ignore[misc]
    def _graph_available(): return False  # type: ignore[misc]

load_dotenv()

logger = logging.getLogger(__name__)

KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "knowledge")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "faiss_index.bin")
METADATA_PATH = os.path.join(os.path.dirname(__file__), "chunks_metadata.json")
EMBED_MODEL = "intfloat/multilingual-e5-small"
TOP_K = 5

_BENGALI_FALLBACK = "দুঃখিত, এই প্রশ্নের উত্তর দিতে পারছি না।"
_ENGLISH_FALLBACK = "Sorry, I could not find an answer."

_MEDICINE_SAFE_EN = (
    "For specific medication advice, please consult a licensed pharmacist or doctor. "
    "SkinAI Bangladesh provides triage and referral guidance only."
)
_MEDICINE_SAFE_BN = (
    "নির্দিষ্ট ওষুধের পরামর্শের জন্য অনুগ্রহ করে একজন লাইসেন্সপ্রাপ্ত ফার্মাসিস্ট বা ডাক্তারের সাথে পরামর্শ করুন। "
    "SkinAI Bangladesh শুধুমাত্র ট্রাইয়েজ ও রেফারেল নির্দেশিকা প্রদান করে।"
)

# Regex patterns that identify medicine-specific content in a model response
_MEDICINE_PATTERN = re.compile(
    r"\b\d+\s*mg\b"                           # dosages like "500 mg"
    r"|\b(tablet|capsule|cream|ointment|syrup|injection|drops)\b"  # dosage forms
    r"|\bওষুধ\s+(খান|নিন|দিন)\b"              # Bengali: "take medicine"
    r"|\bট্যাবলেট\b|\bক্যাপসুল\b|\bমলম\b"    # Bengali dosage forms
    r"|\b\w+(cillin|mycin|zole|prazole|cortisone|steroid)\b",      # drug-class suffixes
    re.IGNORECASE,
)


def _redact_medicine_names(answer: str, lang: str = "en") -> str:
    """Replace any medicine/dosage content with a safe referral message."""
    if _MEDICINE_PATTERN.search(answer):
        return _MEDICINE_SAFE_BN if lang == "bn" else _MEDICINE_SAFE_EN
    return answer

# Canonical disease name map: lowercase-underscore key → Kuzu graph key
_DISEASE_CLASSES: dict[str, str] = {
    "atopic_dermatitis":     "Atopic_Dermatitis",
    "contact_dermatitis":    "Contact_Dermatitis",
    "eczema":                "Eczema",
    "scabies":               "Scabies",
    "seborrheic_dermatitis": "Seborrheic_Dermatitis",
    "tinea":                 "Tinea",
    "vitiligo":              "Vitiligo",
}


def _extract_disease_class(disease_context: str) -> str | None:
    """Normalize any disease_context string to a Kuzu graph disease key, or None."""
    normalized = disease_context.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized in _DISEASE_CLASSES:
        return _DISEASE_CLASSES[normalized]
    for key, graph_name in _DISEASE_CLASSES.items():
        if key in normalized:
            return graph_name
    return None


_PROMPT_TEMPLATE = """\
You are SkinAI Bangladesh — a helpful medical information assistant for rural Bangladesh.
A patient has asked: "{question}"

Your task:
1. Use the medical context below to give a clear, helpful answer about the skin condition mentioned.
2. The context is in English — if the patient asked in Bengali or asked for a Bengali answer, translate your answer into natural Bengali (বাংলা).
3. Always explain what the condition is, what causes it, and when to see a doctor.
4. Do NOT recommend specific medication names. Do NOT make a diagnosis.
5. Keep the answer concise — 3 to 5 sentences.
6. If patient memory or prior conversation is provided below, tailor your guidance to their clinical history (age, duration, symptoms, prior visits) and maintain continuity with earlier messages.

{disease_note}{memory_note}{graph_note}Medical context (CDC · NIH · WHO · DGHS Bangladesh):
{context}

Answer in {lang_label}:\
"""

# Romanized Bengali keywords — signal Bengali reply AND are stripped before BM25 search
_ROMANIZED_BN_TRIGGERS = {
    "bangla", "banglai", "bengali", "bolo", "bolun", "ki",
    "ami", "apni", "amar", "apnar", "rog", "janai", "janate",
    "kora", "kore", "hobe", "hobey", "chai", "chahi", "bolte",
    "somporkay", "somporke", "niye", "keno", "kothay",
}

# --- Singletons ---
_chunks: list[dict] = []          # raw chunks from .txt files — always populated
_index = None                     # faiss.Index — optional
_metadata: list[dict] | None = None
_embed_model = None
_gemini_client: genai.Client | None = None
_idf: dict[str, float] = {}       # precomputed IDF for BM25


# --- Keyword helpers ---

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zঀ-৿]+", text.lower())


# Bengali & Banglish → English medical synonym expansion so BM25 works on Bengali queries
# against the English CDC/NIH/WHO/DGHS corpus.
_BN_MEDICAL_SYNONYMS: dict[str, list[str]] = {
    "চুলকানি": ["itching", "pruritus", "itch"],
    "চুলকাচ্ছে": ["itching", "pruritus", "itch"],
    "khujli": ["itching", "pruritus", "itch"],
    "khuj": ["itching", "itch"],
    "দাদ": ["tinea", "ringworm", "fungal"],
    "টিনিয়া": ["tinea", "ringworm", "fungal"],
    "dad": ["tinea", "ringworm", "fungal"],
    "খোস": ["scabies", "mite", "burrow", "contagious"],
    "পাঁচড়া": ["scabies", "mite", "contagious"],
    "একজিমা": ["eczema", "atopic", "dermatitis"],
    "অ্যাটোপিক": ["atopic", "dermatitis", "eczema"],
    "কন্টাক্ট": ["contact", "dermatitis", "allergy"],
    "সেবোরিক": ["seborrheic", "dermatitis", "dandruff"],
    "শ্বেতী": ["vitiligo", "white", "patches"],
    "খুশকি": ["seborrheic", "dandruff", "scalp"],
    "লাল": ["redness", "erythema", "rash"],
    "ফুসকুড়ি": ["rash", "blister", "lesion"],
    "ফোসকা": ["blister", "blistering"],
    "জ্বর": ["fever", "infection", "urgent"],
    "ছড়িয়ে": ["spreading", "worsening"],
    "ব্যথা": ["pain", "infection"],
    "রক্ত": ["bleeding", "emergency"],
    "ডাক্তার": ["doctor", "refer", "clinic", "hospital"],
    "হাসপাতাল": ["hospital", "emergency", "clinic"],
    "বিপজ্জনক": ["serious", "urgent", "doctor", "complications"],
    "বয়স": ["elderly", "adult", "risk"],
    "পরিবার": ["family", "contagious", "spread", "household"],
    "ছোঁয়াচে": ["contagious", "spread", "transmission"],
    "choyache": ["contagious", "spread", "transmission"],
    "মলম": ["treatment", "care", "doctor"],
    "ক্রিম": ["treatment", "care", "doctor"],
}


def _search_tokens(question: str, disease_context: str | None = None) -> list[str]:
    """Return expanded BM25 search tokens: strip Romanized filler + expand BN/Banglish medical terms."""
    tokens = _tokenize(question)
    cleaned = [t for t in tokens if t not in _ROMANIZED_BN_TRIGGERS]
    base_tokens = cleaned if cleaned else tokens

    expanded = list(base_tokens)
    for tok in base_tokens:
        for bn_key, en_terms in _BN_MEDICAL_SYNONYMS.items():
            if bn_key in tok:
                expanded.extend(en_terms)

    if disease_context:
        disease_key = _extract_disease_class(disease_context)
        if disease_key:
            expanded.extend(_tokenize(disease_key.replace("_", " ")))

    return expanded


def _build_idf(chunks: list[dict]) -> dict[str, float]:
    N = len(chunks)
    df: Counter = Counter()
    for c in chunks:
        for tok in set(_tokenize(c.get("text", ""))):
            df[tok] += 1
    return {tok: math.log((N - f + 0.5) / (f + 0.5) + 1) for tok, f in df.items()}


def _bm25_score(query_tokens: list[str], chunk_text: str, k1: float = 1.5, b: float = 0.75) -> float:
    avg_dl = sum(len(_tokenize(c.get("text", ""))) for c in _chunks) / max(len(_chunks), 1)
    tokens = _tokenize(chunk_text)
    dl = len(tokens)
    tf_map = Counter(tokens)
    score = 0.0
    for tok in query_tokens:
        if tok not in _idf:
            continue
        tf = tf_map.get(tok, 0)
        score += _idf[tok] * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1)))
    return score


# --- Load chunks from .txt files ---

def _load_chunks_from_files() -> list[dict]:
    chunks = []
    if not os.path.isdir(KNOWLEDGE_DIR):
        return chunks
    for fname in sorted(os.listdir(KNOWLEDGE_DIR)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(KNOWLEDGE_DIR, fname)
        try:
            with open(path, encoding="utf-8") as f:
                raw = f.read().strip()
        except Exception:
            continue
        source, topic, body = "", "", raw
        if "---" in raw:
            header, _, body = raw.partition("---")
            for line in header.splitlines():
                if line.startswith("SOURCE:"):
                    source = line.split(":", 1)[1].strip()
                elif line.startswith("TOPIC:"):
                    topic = line.split(":", 1)[1].strip()
        chunks.append({
            "filename": fname,
            "source": source,
            "topic": topic,
            "text": raw,
        })
    return chunks


# --- Optional FAISS upgrade ---

def _try_load_faiss() -> bool:
    global _index, _metadata, _embed_model
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        return False
    try:
        import faiss as _faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer

        _index = _faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, encoding="utf-8") as f:
            _metadata = json.load(f)

        class _STWrapper:
            def __init__(self):
                self._model = SentenceTransformer(EMBED_MODEL)
            def encode(self, texts):
                emb = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)
                return (emb / norms).astype("float32")

        _embed_model = _STWrapper()
        logger.info("FAISS semantic search loaded: %d vectors.", _index.ntotal)
        return True
    except Exception as exc:
        logger.warning("FAISS/ST load failed (%s) — using BM25 keyword search.", exc)
        _index = None
        _metadata = None
        _embed_model = None
        return False


# --- Gemini ---

def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
    return _gemini_client


# --- Public API ---

def load_index() -> bool:
    """
    Load knowledge base. Always returns True if any .txt files exist.
    Optionally upgrades to FAISS semantic search if index files are present.
    Never raises.
    """
    global _chunks, _idf

    try:
        _chunks = _load_chunks_from_files()
        if not _chunks:
            logger.warning("No knowledge .txt files found in %s", KNOWLEDGE_DIR)
            return False
        _idf = _build_idf(_chunks)
        logger.info("BM25 index ready: %d chunks from knowledge files.", len(_chunks))
    except Exception as exc:
        logger.error("Failed to load knowledge files: %s", exc)
        return False

    # Try ChromaDB persistent vector store (best-effort, non-blocking)
    _chroma_build(_chunks)

    # Try FAISS as secondary vector upgrade (best-effort, non-blocking)
    _try_load_faiss()
    return True


def _detect_lang(text: str) -> str:
    # Unicode Bengali script — definitive
    if any("ঀ" <= ch <= "৿" for ch in text):
        return "bn"
    # Romanized Bengali: user typed in Latin script but wants Bengali reply
    words = set(re.findall(r"[a-z]+", text.lower()))
    if words & _ROMANIZED_BN_TRIGGERS:
        return "bn"
    return "en"


def _rerank_chunks(
    question: str,
    candidates: list[dict],
    disease_context: str | None = None,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Stage-2 Clinical Context Reranker:
    Re-scores RRF hybrid candidates using:
      1. Reciprocal Rank Fusion score (Dense + Sparse agreement)
      2. Expanded query-to-chunk keyword overlap density
      3. Active disease_context topic alignment boost
      4. DGHS Bangladesh local clinical guideline boost
    """
    q_tokens = set(_search_tokens(question, disease_context=disease_context))
    active_disease = (_extract_disease_class(disease_context) or "").lower() if disease_context else ""
    active_disease_words = [w for w in active_disease.split("_") if w]

    reranked: list[tuple[float, dict]] = []
    for chunk in candidates:
        base_rrf = float(chunk.get("_rrf_score", 0.0))
        text_lower = chunk.get("text", "").lower()
        topic_lower = chunk.get("topic", "").lower().replace("-", " ").replace("_", " ")
        source_upper = chunk.get("source", "").upper()

        chunk_tokens = set(_tokenize(text_lower))
        overlap = len(q_tokens & chunk_tokens) / max(len(q_tokens), 1)

        disease_boost = 0.0
        if active_disease_words and all(w in topic_lower for w in active_disease_words):
            disease_boost = 0.30
        elif active_disease_words and any(w in topic_lower for w in active_disease_words):
            disease_boost = 0.15

        dghs_boost = 0.08 if "DGHS" in source_upper else 0.0

        final_score = base_rrf + (0.35 * overlap) + disease_boost + dghs_boost
        clean_chunk = {k: v for k, v in chunk.items() if k != "_rrf_score"}
        reranked.append((final_score, clean_chunk))

    reranked.sort(key=lambda x: -x[0])
    return [c for _, c in reranked[:top_k]]


def retrieve(
    question: str,
    top_k: int = TOP_K,
    disease_context: str | None = None,
) -> list[dict]:
    """
    True Hybrid Retrieval:
      1. Dense semantic search (ChromaDB or FAISS) — Top 15
      2. Sparse BM25 search with BN/Banglish medical synonym expansion — Top 15
      3. Reciprocal Rank Fusion (RRF, k=60)
      4. Clinical Context Reranker → Top-K
    """
    if not question.strip():
        return []

    candidate_k = max(top_k * 3, 15)
    dense_results: list[dict] = []
    sparse_results: list[dict] = []

    # 1A. Dense semantic retrieval (ChromaDB preferred, FAISS fallback)
    if _chroma_available():
        dense_results = _chroma_query(question, top_k=candidate_k)
    elif _index is not None and _metadata is not None and _embed_model is not None:
        try:
            qvec = _embed_model.encode([question]).reshape(1, -1)
            k = min(candidate_k, len(_metadata))
            _scores, indices = _index.search(qvec, k)
            dense_results = [_metadata[i] for i in indices[0] if 0 <= i < len(_metadata)]
        except Exception as exc:
            logger.warning("FAISS search failed, relying on BM25: %s", exc)

    # 1B. Sparse BM25 retrieval (runs IN PARALLEL with dense search, not just fallback)
    if _chunks:
        query_tokens = _search_tokens(question, disease_context=disease_context)
        scored = [
            (_bm25_score(query_tokens, c.get("text", "")), i, c)
            for i, c in enumerate(_chunks)
        ]
        scored.sort(key=lambda x: -x[0])
        sparse_results = [c for score, _, c in scored[:candidate_k] if score > 0]

    # 2. Reciprocal Rank Fusion (RRF)
    rrf_k = 60
    fused_map: dict[str, dict] = {}
    rrf_scores: dict[str, float] = {}

    for rank, c in enumerate(dense_results):
        key = c.get("filename") or c.get("text", "")[:80]
        fused_map[key] = dict(c)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (rrf_k + rank + 1))

    for rank, c in enumerate(sparse_results):
        key = c.get("filename") or c.get("text", "")[:80]
        if key not in fused_map:
            fused_map[key] = dict(c)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (rrf_k + rank + 1))

    if not fused_map:
        return _chunks[:top_k] if _chunks else []

    candidates = []
    for key, chunk in fused_map.items():
        chunk["_rrf_score"] = rrf_scores[key]
        candidates.append(chunk)

    # 3. Stage-2 Reranking
    return _rerank_chunks(question, candidates, disease_context=disease_context, top_k=top_k)


# Primary + fallback Gemini models (configurable via GEMINI_MODEL env var)
_GEMINI_MODELS = [
    m for m in [
        os.getenv("GEMINI_MODEL", "").strip(),
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ]
    if m
]


def answer_question(
    question: str,
    lang: str | None = None,
    disease_context: str | None = None,
    patient_history: dict | None = None,
    chat_history: list[dict] | None = None,
    memory_note: str | None = None,
) -> str:
    """
    Full Hybrid + Rerank + Memory-aware RAG pipeline:
    retrieve top-k chunks → inject patient memory + Kuzu graph → Gemini answer.
    Always returns a str. Never raises.
    """
    if not question or not question.strip():
        return _ENGLISH_FALLBACK

    if lang is None:
        lang = _detect_lang(question)

    fallback = _BENGALI_FALLBACK if lang == "bn" else _ENGLISH_FALLBACK
    lang_label = "Bengali (বাংলা)" if lang == "bn" else "English"

    # Build Agent Memory context (if provided explicitly or via patient_history / chat_history)
    resolved_memory_note = memory_note or ""
    if not resolved_memory_note and (patient_history or chat_history):
        mem_parts: list[str] = []
        if patient_history:
            h_items = [
                f"{k}: {', '.join(v) if isinstance(v, list) else v}"
                for k, v in patient_history.items()
                if v
            ]
            if h_items:
                mem_parts.append("- Clinical history (voice/form): " + " | ".join(h_items))
        if chat_history:
            turns = "\n".join(
                f"  {str(m.get('role', 'user')).capitalize()}: {str(m.get('content', ''))[:200]}"
                for m in chat_history[-4:]
                if m.get("content")
            )
            if turns:
                mem_parts.append("- Recent conversation memory:\n" + turns)
        if mem_parts:
            resolved_memory_note = "Patient memory & conversation context:\n" + "\n".join(mem_parts) + "\n\n"

    # Context-aware disk cache key (prevents returning generic cached answer when memory is active)
    cache_ctx = f"{disease_context or ''}|{resolved_memory_note[:160]}"
    _ck = _cache_key(question, cache_ctx, lang)
    _cached = _cache_get(_ck)
    if _cached:
        logger.info("answer_question: cache hit (key=%s…)", _ck[:12])
        return _cached

    if not _chunks:
        logger.warning("answer_question called before load_index().")
        return fallback

    chunks = retrieve(question, disease_context=disease_context)
    if not chunks:
        return fallback

    context = "\n\n---\n\n".join(
        f"[{c.get('source', '')} | {c.get('topic', '')}]\n{c.get('text', '')}"
        for c in chunks
    )
    disease_note = (
        f"The patient has been diagnosed with: {disease_context}\n"
        if disease_context
        else ""
    )

    # Graph RAG — traverse Kuzu knowledge graph to inject structured clinical context
    graph_note = ""
    if disease_context and _GRAPH_IMPORT_OK and _graph_available():
        disease_key = _extract_disease_class(disease_context)
        if disease_key:
            try:
                syms = _graph_symptoms(disease_key)
                related = _graph_related(disease_key)
                parts: list[str] = []
                if syms:
                    sym_labels = [
                        f"{s['name_bn']} ({s['name']})" + (" ⚠" if s.get("is_escalation") else "")
                        for s in syms[:6]
                    ]
                    parts.append(f"Clinical graph — linked symptoms: {', '.join(sym_labels)}.")
                if related:
                    parts.append(f"Differential conditions sharing symptoms: {', '.join(related[:3])}.")
                if parts:
                    graph_note = "Graph knowledge (Kuzu disease-symptom graph):\n" + " ".join(parts) + "\n\n"
            except Exception:
                pass  # graph errors never block the answer

    prompt = _PROMPT_TEMPLATE.format(
        lang_label=lang_label,
        context=context,
        question=question.strip(),
        disease_note=disease_note,
        memory_note=resolved_memory_note,
        graph_note=graph_note,
    )

    client = _get_gemini_client()
    models_to_try = list(dict.fromkeys(_GEMINI_MODELS))
    gen_config = None
    try:
        from google.genai import types as _genai_types
        gen_config = _genai_types.GenerateContentConfig(
            thinking_config=_genai_types.ThinkingConfig(thinking_budget=0),
            max_output_tokens=300,
        )
    except Exception:
        gen_config = None

    for attempt in range(3):
        model_name = models_to_try[min(attempt, len(models_to_try) - 1)]
        try:
            kwargs = {"model": model_name, "contents": prompt}
            if gen_config is not None:
                kwargs["config"] = gen_config
            response = client.models.generate_content(**kwargs)
            answer = _redact_medicine_names(response.text.strip(), lang=lang)
            _cache_put(_ck, answer)
            return answer
        except Exception as exc:
            logger.warning("Gemini attempt %d (%s) failed: %s", attempt + 1, model_name, exc)

    return fallback
