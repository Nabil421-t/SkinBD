"""
orchestrator/clinical_orchestrator.py
─────────────────────────────────────
Central Clinical Agent Orchestrator for SkinAI Bangladesh.

Unifies all channels (Web, Telegram/WhatsApp, Offline /lite, MCP) behind one
memory-aware, safety-first decision engine:
  1. Reads & updates Patient Memory (profile, preferences, past visits, chat turns).
  2. Detects Intent & Mid-Chat Emergency Red Flags (জ্বর/ছড়িয়ে/ব্যথা/রক্ত).
  3. Dynamically routes to ONLY the tools needed:
       • memory_check      — always consulted for continuity & repeat-visit check
       • triage_engine     — runs on new diagnosis OR mid-chat red-flag escalation
                             (with Signal 4 longitudinal memory escalation)
       • hospital_finder   — runs ONLY when Tier >= 2 or user explicitly asks for
                             a hospital (skips slow Gemini LLM on pure map queries)
       • rag_retrieve + rag_answer — runs Hybrid RRF + Reranker + Memory Prompt
                                     for medical questions
  4. Emits an explicit `tools_called` audit trail and `@traceable` LangSmith spans.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Optional LangSmith tracing — works seamlessly even if langsmith is not installed
try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):  # type: ignore[misc]
        def _dec(fn):
            return fn
        return _dec

from memory.agent_memory import memory_store
from severity.engine import ESCALATION_KEYWORDS, compute_tier, explain_tier
from rag import retriever as rag_retriever
from map.hospital_finder import find_nearest_hospitals, get_district_coords

_HOSPITAL_TRIGGERS = {
    "হাসপাতাল", "মেডিকেল", "জরুরি হাসপাতাল", "কোথায়", "ঠিকানা", "কাছের",
    "hospital", "nearest clinic", "emergency room", "nearest", "where is", "address",
}

_MEDICAL_INFO_TRIGGERS = {
    "কী", "কি", "কেন", "লক্ষণ", "চিকিৎসা", "ছোঁয়াচে", "বিপজ্জনক", "করব", "ক্রিম", "ওষুধ",
    "দাদ", "একজিমা", "খোস", "পাঁচড়া", "চুলকানি", "শ্বেতী", "টিনিয়া",
    "what", "why", "how", "cause", "symptom", "contagious", "dangerous", "treatment",
    "tinea", "scabies", "eczema", "ringworm", "itching", "khujli", "choyache",
}

# Triggers indicating the user's question depends on personal history, prior turns,
# or longitudinal follow-up — ONLY then should we pay the memory lookup & prompt token cost.
_MEMORY_NEEDED_TRIGGERS = {
    "আমার", "আগের", "আগে", "বয়স", "আবার", "কমছে না", "বাড়ছে", "বাড়ছে", "এখন", "ফলো-আপ",
    "এটা কি", "এইটা", "তাহলে", "এরপর", "আমারটা",
    "my", "previous", "prior", "before", "again", "age", "still", "worse", "not improving",
    "follow-up", "followup", "earlier", "last time", "this dangerous", "for me",
}


def needs_memory_lookup(
    user_message: str,
    disease_class: Optional[str] = None,
    patient_history: Optional[dict] = None,
    prior_chat: Optional[list[dict]] = None,
    prior_tier: Optional[int] = None,
    wants_hospital: bool = False,
    explicit_district: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Adaptive Memory Gate (runs in <0.05 ms before touching the memory store).
    Returns (use_memory: bool, reason: str).

    Skips memory lookup & prompt injection on straightforward, self-contained
    general medical questions (e.g. "দাদ হলে কী করব?", "খোস-পাঁচড়া কি ছোঁয়াচে?"),
    cutting unnecessary tool spans and prompt token overhead.
    """
    low = (user_message or "").lower()

    # 1. Clinical triage / new photo needs episodic visit memory to detect repeat unimproved visits
    if disease_class is not None:
        return True, "triage_repeat_visit_check"

    # 2. Mid-chat emergency red flags need memory to retrieve prior disease & district for Tier 3 escalation
    if has_red_flag_keywords(user_message):
        return True, "emergency_red_flag_escalation"

    # 3. Hospital query without an explicit district in the message needs remembered district
    if wants_hospital and not explicit_district:
        return True, "missing_district_from_memory"

    # 4. Explicit patient history / prior conversation / prior tier supplied
    if patient_history or prior_chat or prior_tier is not None:
        return True, "active_patient_context_provided"

    # 5. Personal / follow-up pronouns or longitudinal phrases in the question
    if any(tok in low for tok in _MEMORY_NEEDED_TRIGGERS):
        return True, "personal_or_followup_reference"

    # 6. Otherwise: straightforward general medical question → Direct Hybrid RAG (skip memory!)
    return False, "straightforward_question_skip_memory"

_BN_DISTRICT_MAP = {
    "ঢাকা": "dhaka", "চট্টগ্রাম": "chattogram", "চিটাগাং": "chattogram",
    "রংপুর": "rangpur", "সিলেট": "sylhet", "খুলনা": "khulna",
    "বরিশাল": "barisal", "রাজশাহী": "rajshahi", "ময়মনসিংহ": "mymensingh",
    "কুমিল্লা": "cumilla", "যশোর": "jessore", "ফরিদপুর": "faridpur",
    "বগুড়া": "bogura", "দিনাজপুর": "dinajpur",
}


def detect_district_in_text(text: str) -> Optional[str]:
    low = (text or "").lower()
    for bn, en in _BN_DISTRICT_MAP.items():
        if bn in (text or ""):
            return en
    for token in low.replace("?", " ").replace(",", " ").split():
        if get_district_coords(token):
            return token
    return None


def is_hospital_query(text: str) -> bool:
    low = (text or "").lower()
    return any(t in low for t in _HOSPITAL_TRIGGERS)


def is_medical_question(text: str) -> bool:
    low = (text or "").lower()
    return any(t in low for t in _MEDICAL_INFO_TRIGGERS) or "?" in low or "।" in low


def has_red_flag_keywords(text: str) -> bool:
    return any(kw in (text or "") for kw in ESCALATION_KEYWORDS)


def format_hospitals_reply(hospitals: list[dict], district: str, lang: str = "bn") -> str:
    if not hospitals:
        return (
            "নিকটস্থ হাসপাতাল খুঁজে পাওয়া যায়নি।"
            if lang == "bn"
            else "No nearby hospitals found."
        )
    header = (
        f"🏥 **{district.title()}**-এর নিকটস্থ জরুরি হাসপাতালসমূহ:\n"
        if lang == "bn"
        else f"🏥 Nearest emergency hospitals in **{district.title()}**:\n"
    )
    lines = [header]
    for i, h in enumerate(hospitals, 1):
        phone = f" · 📞 {h['phone']}" if h.get("phone") else ""
        lines.append(f"{i}. **{h['name']}** — {h['address']} ({h['dist_km']} km){phone}")
    return "\n".join(lines)


# --- Traced Tool Wrappers (visible as child spans in LangSmith) ---

@traceable(name="orchestrator_tool_memory_check", run_type="tool")
def _tool_memory_check(user_id: str, disease_class: Optional[str], district: Optional[str],
                       patient_history: Optional[dict], prior_chat: Optional[list[dict]]) -> dict:
    mem = memory_store.get(user_id)
    mem.update_profile(patient_history=patient_history, district=district)
    if prior_chat and not mem.chat_turns:
        for turn in prior_chat[-4:]:
            if turn.get("content"):
                mem.add_chat_turn(turn.get("role", "user"), turn["content"])
    is_repeat = mem.has_unimproved_repeat_visit(disease_class) if disease_class else False
    return {
        "snapshot": mem.snapshot(),
        "is_repeat_unimproved": is_repeat,
        "memory_note": mem.format_for_prompt(),
    }


@traceable(name="orchestrator_tool_triage_engine", run_type="tool")
def _tool_triage(disease_class: str, confidence: float, transcript: str, repeat_unimproved: bool) -> dict:
    tier_res = compute_tier(
        disease_class=disease_class,
        confidence=confidence,
        transcript=transcript,
        repeat_unimproved=repeat_unimproved,
    )
    reasons = explain_tier(
        disease_class=disease_class,
        confidence=confidence,
        transcript=transcript,
        repeat_unimproved=repeat_unimproved,
    )
    return {"tier_result": tier_res, "reasons": reasons}


@traceable(name="orchestrator_tool_hospital_finder", run_type="tool")
def _tool_hospital_finder(district: str, n: int = 3) -> list[dict]:
    coords = get_district_coords(district)
    if not coords:
        return []
    return find_nearest_hospitals(coords[0], coords[1], n=n)


@traceable(name="orchestrator_tool_hybrid_rag_retrieve", run_type="retriever")
def _tool_hybrid_rag_retrieve(question: str, disease_context: Optional[str], top_k: int = 5) -> list[dict]:
    if not rag_retriever._chunks:
        rag_retriever.load_index()
    return rag_retriever.retrieve(question, top_k=top_k, disease_context=disease_context)


@traceable(name="orchestrator_tool_hybrid_rag_answer", run_type="chain")
def _tool_hybrid_rag_answer(
    question: str,
    lang: Optional[str],
    disease_context: Optional[str],
    patient_history: Optional[dict],
    chat_history: Optional[list[dict]],
    memory_note: str,
) -> dict:
    chunks = _tool_hybrid_rag_retrieve(question, disease_context=disease_context)
    answer = rag_retriever.answer_question(
        question=question,
        lang=lang,
        disease_context=disease_context,
        patient_history=patient_history,
        chat_history=chat_history,
        memory_note=memory_note,
    )
    return {
        "answer": answer,
        "retrieved_topics": [f"{c.get('source', '')} | {c.get('topic', '')}" for c in chunks],
    }


# --- Main Orchestrator Entry Point ---

@traceable(name="SkinAI_Clinical_Orchestrator", run_type="chain")
def orchestrate_turn(
    user_id: str,
    user_message: str = "",
    disease_class: Optional[str] = None,
    confidence: Optional[float] = None,
    disease_context: Optional[str] = None,
    district: Optional[str] = None,
    patient_history: Optional[dict] = None,
    prior_chat: Optional[list[dict]] = None,
    prior_tier: Optional[int] = None,
    record_new_visit: bool = False,
) -> dict[str, Any]:
    """
    Central Clinical Orchestrator.
    Decides dynamically which tools to invoke based on Memory, Red Flags, and Intent.
    """
    t0 = time.perf_counter()
    tools_called: list[str] = []

    # Resolve district from explicit arg or message text first (before touching memory)
    inferred_district = district or detect_district_in_text(user_message)
    red_flag_hit = has_red_flag_keywords(user_message)
    wants_hospital = is_hospital_query(user_message)
    wants_medical_info = is_medical_question(user_message)

    # Resolve active disease from explicit args first
    resolved_disease = disease_class
    if not resolved_disease and disease_context:
        resolved_disease = rag_retriever._extract_disease_class(disease_context)

    # 1. Adaptive Memory Gate: decide whether this question needs memory or direct RAG/Tool
    use_memory, memory_gate_reason = needs_memory_lookup(
        user_message=user_message,
        disease_class=disease_class,
        patient_history=patient_history,
        prior_chat=prior_chat,
        prior_tier=prior_tier,
        wants_hospital=wants_hospital,
        explicit_district=inferred_district,
    )

    mem = None
    mem_info = {"snapshot": {}, "is_repeat_unimproved": False, "memory_note": ""}
    if use_memory:
        mem = memory_store.get(user_id)
        if not resolved_disease and mem.visits:
            resolved_disease = mem.visits[-1].disease
        mem_info = _tool_memory_check(
            user_id=user_id,
            disease_class=resolved_disease,
            district=inferred_district,
            patient_history=patient_history,
            prior_chat=prior_chat,
        )
        tools_called.append("memory_check")

    resolved_conf = confidence if confidence is not None else (
        mem.visits[-1].confidence if (mem and mem.visits) else 0.75
    )

    active_district = (mem.district if mem else None) or inferred_district
    is_repeat = mem_info["is_repeat_unimproved"]

    # 2. Triage Engine: run on new diagnosis OR mid-chat red flag OR repeat visit check
    tier_result = None
    tier_reasons: list[dict] = []
    if disease_class is not None or (resolved_disease and red_flag_hit):
        # If mid-chat red flag escalates a previously known case (e.g. fever + bleeding),
        # ensure urgent escalation when multiple red flags appear.
        red_flag_count = sum(1 for kw in ESCALATION_KEYWORDS if kw in (user_message or ""))
        t_out = _tool_triage(
            disease_class=resolved_disease or "Tinea",
            confidence=resolved_conf,
            transcript=user_message,
            repeat_unimproved=is_repeat or (red_flag_count >= 2),
        )
        tier_result = t_out["tier_result"]
        tier_reasons = t_out["reasons"]
        tools_called.append("triage_engine")

        if (record_new_visit or disease_class is not None) and mem is not None:
            mem.record_visit(
                disease=resolved_disease or "Tinea",
                confidence=resolved_conf,
                tier=tier_result["tier"],
                history=patient_history or mem.clinical_profile,
                district=active_district,
            )

    current_tier = (
        tier_result["tier"]
        if tier_result is not None
        else (prior_tier if prior_tier is not None else (mem.visits[-1].tier if (mem and mem.visits) else None))
    )

    # 3. Hospital Finder: run ONLY when Tier >= 2 (on emergency/triage) OR user explicitly asks
    hospitals: list[dict] = []
    should_find_hospital = (
        (wants_hospital and bool(active_district))
        or (red_flag_hit and (current_tier or 0) >= 2 and bool(active_district))
    )
    if should_find_hospital and active_district:
        hospitals = _tool_hospital_finder(active_district, n=3)
        tools_called.append("hospital_finder")

    # 4. Hybrid + Rerank RAG: run for medical questions — SKIP on pure hospital-only queries!
    pure_hospital_query = wants_hospital and not red_flag_hit and not any(
        w in (user_message or "").lower()
        for w in ("দাদ", "একজিমা", "খোস", "পাঁচড়া", "চুলকানি", "লক্ষণ", "ছোঁয়াচে", "tinea", "scabies", "eczema", "cause")
    )

    rag_answer = ""
    retrieved_topics: list[str] = []
    lang = rag_retriever._detect_lang(user_message) if user_message.strip() else ((mem.preferred_lang if mem else None) or "bn")

    if user_message.strip() and not pure_hospital_query and (wants_medical_info or red_flag_hit or disease_class is None):
        ctx_str = disease_context or (
            f"{resolved_disease.replace('_', ' ')} ({int(resolved_conf * 100)}% confidence)"
            if resolved_disease
            else None
        )
        rag_out = _tool_hybrid_rag_answer(
            question=user_message,
            lang=lang,
            disease_context=ctx_str,
            patient_history=(patient_history or (mem.clinical_profile if mem else None)) if use_memory else None,
            chat_history=(mem.chat_turns if mem else None) if use_memory else None,
            memory_note=mem_info["memory_note"] if use_memory else "",
        )
        rag_answer = rag_out["answer"]
        retrieved_topics = rag_out["retrieved_topics"]
        tools_called.extend(["rag_retrieve", "rag_answer"])

        if mem is not None:
            mem.add_chat_turn("user", user_message, lang=lang)
            mem.add_chat_turn("assistant", rag_answer, lang=lang)

    # 5. Synthesize final unified response
    response_sections: list[str] = []
    if tier_result and red_flag_hit:
        response_sections.append(
            f"🚨 **জরুরি সতর্কতা (Tier {tier_result['tier']} · {tier_result['urgency_label']}):** "
            f"{tier_result['bengali_text']}"
        )
    if hospitals:
        response_sections.append(format_hospitals_reply(hospitals, active_district or "", lang=lang))
    if rag_answer:
        response_sections.append(rag_answer)
    if not response_sections and tier_result:
        response_sections.append(
            f"Tier {tier_result['tier']} ({tier_result['urgency_label']}) → {tier_result['facility']} | "
            f"{tier_result['bengali_text']}"
        )

    final_reply = "\n\n".join(response_sections)
    total_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "tools_called": tools_called,
        "memory_gate": {"used_memory": use_memory, "reason": memory_gate_reason},
        "final_tier": current_tier,
        "tier_result": tier_result,
        "tier_reasons": tier_reasons,
        "hospitals": hospitals,
        "hospital_finder_called": "hospital_finder" in tools_called,
        "retrieved_topics": retrieved_topics,
        "used_patient_memory": bool(mem_info["memory_note"]),
        "answer": final_reply,
        "memory_snapshot": mem.snapshot() if mem else {},
        "total_latency_ms": total_ms,
    }
