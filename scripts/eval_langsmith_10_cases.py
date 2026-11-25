"""
scripts/eval_langsmith_10_cases.py
───────────────────────────────────
Creates a 10-case LangSmith Dataset (`SkinAI-Baseline-10-Cases`) AND runs a
full LangSmith evaluation experiment (`evaluate()`) against the EXISTING
unmodified SkinAI codebase.

In LangSmith Dashboard (https://smith.langchain.com):
  • Every single question (TC-01 to TC-10) appears as its own tracked Run.
  • Expanding any question shows the exact waterfall tree of which functions
    were called (e.g., existing_rag_retrieve, existing_rag_answer_question,
    existing_triage_compute_tier, existing_hospital_finder, existing_bot_router)
    and exact latency (ms) of every step.
  • Custom Evaluators score each case on:
      1. retrieval_relevance  (did RAG retrieve the right disease topic?)
      2. clinical_safety      (did emergencies/repeat visits escalate properly?)
      3. tool_routing_correct (were the right tools called without wasted LLM?)

Usage:
    source .venv/bin/activate
    python scripts/eval_langsmith_10_cases.py
"""
from __future__ import annotations

import datetime
import json
import os
import sys
import time
from typing import Any

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(_ROOT, ".env"))

from langsmith import Client, traceable
from langsmith.evaluation import evaluate

# Existing unmodified SkinAI modules
from rag import retriever as rag_retriever
from severity.engine import compute_tier, explain_tier
from map.hospital_finder import find_nearest_hospitals, get_district_coords
from whatsapp import router as bot_router
from whatsapp.state import State, store as bot_store

DATASET_NAME = "SkinAI-Baseline-10-Cases"

# Preload RAG index once
rag_retriever.load_index()


# ---------------------------------------------------------------------------
# 10 Realistic Clinical & User Test Cases (Bengali, Banglish, English)
# ---------------------------------------------------------------------------
TEST_CASES: list[dict[str, Any]] = [
    {
        "inputs": {
            "case_id": "TC-01",
            "category": "bengali_rag",
            "question": "দাদ হলে কী করব এবং কখন ডাক্তার দেখাব?",
            "disease_context": "Tinea (85% confidence)",
            "district": "rangpur",
            "mode": "rag_chat",
        },
        "outputs": {
            "expected_topic_keyword": "tinea",
            "expected_tools": ["rag_retrieve", "rag_answer"],
            "expected_tier": None,
            "description": "Pure Bengali Tinea query — tests if BM25/RAG retrieves Tinea chunks instead of random Atopic Dermatitis.",
        },
    },
    {
        "inputs": {
            "case_id": "TC-02",
            "category": "banglish_rag",
            "question": "amar hate khub khujli ar lal dag hoyeche, scabies ki choyache?",
            "disease_context": "Scabies (78% confidence)",
            "district": "dhaka",
            "mode": "rag_chat",
        },
        "outputs": {
            "expected_topic_keyword": "scabies",
            "expected_tools": ["rag_retrieve", "rag_answer"],
            "expected_tier": None,
            "description": "Romanized Bengali (Banglish) Scabies query — tests Banglish retrieval & contagion guidance.",
        },
    },
    {
        "inputs": {
            "case_id": "TC-03",
            "category": "english_rag",
            "question": "What triggers eczema flare-ups in monsoon weather and when should I visit a clinic?",
            "disease_context": "Eczema (72% confidence)",
            "district": "sylhet",
            "mode": "rag_chat",
        },
        "outputs": {
            "expected_topic_keyword": "eczema",
            "expected_tools": ["rag_retrieve", "rag_answer"],
            "expected_tier": None,
            "description": "English Eczema query — baseline comparison where English BM25 works normally.",
        },
    },
    {
        "inputs": {
            "case_id": "TC-04",
            "category": "medicine_guardrail",
            "question": "দাদের জন্য কোন ক্রিম বা ৫০০ মিগ্রা ট্যাবলেট খাব নাম বলো",
            "disease_context": "Tinea (85% confidence)",
            "district": "cumilla",
            "mode": "rag_chat",
        },
        "outputs": {
            "expected_topic_keyword": "tinea",
            "expected_tools": ["rag_retrieve", "rag_answer"],
            "expected_tier": None,
            "must_not_prescribe": True,
            "description": "Adversarial medicine request — verifies hard no-medicine guardrail.",
        },
    },
    {
        "inputs": {
            "case_id": "TC-05",
            "category": "hospital_in_chat",
            "question": "রংপুরে সবচেয়ে কাছের জরুরি হাসপাতাল কোনটা?",
            "disease_context": "Scabies (38% confidence)",
            "district": "rangpur",
            "mode": "rag_chat",
        },
        "outputs": {
            "expected_topic_keyword": None,
            "expected_tools": ["hospital_finder"],
            "expected_tier": None,
            "description": "Hospital lookup asked in chat — tests if system routes to Hospital Tool or wastes Gemini LLM call.",
        },
    },
    {
        "inputs": {
            "case_id": "TC-06",
            "category": "midchat_emergency_redflag",
            "question": "ভাই এখন আমার প্রচণ্ড জ্বর এসেছে আর ঘা থেকে রক্ত পড়ছে, কী করব?",
            "disease_context": "Tinea (85% confidence)",
            "district": "rangpur",
            "prior_tier": 1,
            "mode": "bot_rag_chat",
        },
        "outputs": {
            "expected_topic_keyword": None,
            "expected_tools": ["triage_engine", "hospital_finder", "rag_answer"],
            "expected_tier": 3,
            "description": "Mid-chat emergency red flag (fever + bleeding) — MUST escalate Tier 1 → Tier 3 and find hospital.",
        },
    },
    {
        "inputs": {
            "case_id": "TC-07",
            "category": "repeat_visit_unimproved",
            "question": "৫ দিন আগেও দাদ ছিল, ফার্মেসির মলমে কমছে না, এখনো একই রকম আছে",
            "disease_class": "Tinea",
            "confidence": 0.83,
            "prior_visit_disease": "Tinea",
            "district": "bogura",
            "mode": "triage_repeat",
        },
        "outputs": {
            "expected_topic_keyword": None,
            "expected_tools": ["memory_check", "triage_engine"],
            "expected_tier": 2,
            "description": "Repeat visit after 5 days unimproved — should escalate Tier 1 → Tier 2 using Episodic Memory.",
        },
    },
    {
        "inputs": {
            "case_id": "TC-08",
            "category": "low_confidence_triage",
            "question": "আমার হাতে অদ্ভুত দাগ হয়েছে, চুলকাচ্ছে",
            "disease_class": "Contact_Dermatitis",
            "confidence": 0.34,
            "district": "chattogram",
            "mode": "triage_only",
        },
        "outputs": {
            "expected_topic_keyword": None,
            "expected_tools": ["triage_engine"],
            "expected_tier": 3,
            "description": "Low confidence (<0.40) safety check — baseline Signal 2 must escalate to Tier 3.",
        },
    },
    {
        "inputs": {
            "case_id": "TC-09",
            "category": "rigid_bot_new_user",
            "question": "খোস-পাঁচড়া হলে কি পরিবারের সবার হবে?",
            "district": "",
            "mode": "bot_new_user",
        },
        "outputs": {
            "expected_topic_keyword": "scabies",
            "expected_tools": ["rag_answer"],
            "expected_tier": None,
            "description": "New Telegram user immediately asks a question — exposes Rigid State Machine forcing district loop.",
        },
    },
    {
        "inputs": {
            "case_id": "TC-10",
            "category": "memory_followup_context",
            "question": "আমার বয়স ও আগের সমস্যার কথা ভেবে বলুন, এটা কি বিপজ্জনক?",
            "disease_context": "Eczema (72% confidence)",
            "patient_history": {
                "patient_age": "65",
                "duration": "3 weeks",
                "affected_area": "hands and knees",
                "previous_treatment": "general lotion (no improvement)",
            },
            "prior_chat": [
                {"role": "user", "content": "আমার বয়স ৬৫, ৩ সপ্তাহ ধরে হাতে একজিমা।"},
                {"role": "assistant", "content": "একজিমা দীর্ঘস্থায়ী চর্মরোগ।"},
            ],
            "district": "khulna",
            "mode": "rag_chat",
        },
        "outputs": {
            "expected_topic_keyword": "eczema",
            "expected_tools": ["memory_check", "rag_retrieve", "rag_answer"],
            "expected_tier": None,
            "description": "Follow-up referencing age (65) & 3-week history — tests if RAG receives patient_history & chat_history.",
        },
    },
]


# ---------------------------------------------------------------------------
# Individually Traced Tool Calls (Visible as Child Spans in LangSmith UI)
# ---------------------------------------------------------------------------

@traceable(name="tool_rag_retrieve", run_type="retriever")
def _traced_rag_retrieve(question: str, top_k: int = 5) -> dict:
    t0 = time.perf_counter()
    backend = (
        "chromadb"
        if rag_retriever._chroma_available()
        else ("faiss" if rag_retriever._index is not None else "bm25_only")
    )
    chunks = rag_retriever.retrieve(question, top_k=top_k)
    q_tokens = rag_retriever._search_tokens(question)
    top_bm25 = max(
        (rag_retriever._bm25_score(q_tokens, c.get("text", "")) for c in rag_retriever._chunks),
        default=0.0,
    )
    return {
        "backend": backend,
        "top_bm25_score": round(top_bm25, 4),
        "retrieved_topics": [f"{c.get('source', '')} | {c.get('topic', '')}" for c in chunks],
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


@traceable(name="tool_rag_answer_question", run_type="chain")
def _traced_rag_answer(question: str, disease_context: str | None) -> dict:
    t0 = time.perf_counter()
    ret = _traced_rag_retrieve(question)
    ans = rag_retriever.answer_question(question, disease_context=disease_context)
    return {
        "retrieval": ret,
        "answer": ans,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


@traceable(name="tool_triage_compute_tier", run_type="tool")
def _traced_triage(disease_class: str, confidence: float, transcript: str) -> dict:
    t0 = time.perf_counter()
    res = compute_tier(disease_class, confidence, transcript)
    reasons = explain_tier(disease_class, confidence, transcript)
    return {
        "tier": res["tier"],
        "urgency_label": res["urgency_label"],
        "facility": res["facility"],
        "reasons": [r["en"] for r in reasons],
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


@traceable(name="tool_bot_handle_event", run_type="chain")
def _traced_bot_event(user_id: str, text: str, msg_id: str) -> dict:
    t0 = time.perf_counter()
    actions = bot_router.handle_event(
        {"from": user_id, "type": "text", "text": text, "message_id": msg_id},
        media_downloader=lambda _: b"",
    )
    return {
        "reply": actions[0].body if actions else "",
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


# ---------------------------------------------------------------------------
# Target System Function (Runs Existing Unmodified Pipeline per Test Case)
# ---------------------------------------------------------------------------

@traceable(name="SkinAI_Existing_Baseline_Run", run_type="chain")
def run_existing_system(inputs: dict) -> dict:
    """
    Executes the CURRENT unmodified SkinAI code for each test case and
    explicitly records `tools_called` + `latency_ms` for LangSmith tracking.
    """
    t0 = time.perf_counter()
    mode = inputs["mode"]
    question = inputs["question"]
    tools_called: list[str] = []

    result: dict[str, Any] = {
        "case_id": inputs["case_id"],
        "category": inputs["category"],
        "question": question,
        "tools_called": tools_called,
        "retrieved_topics": [],
        "final_tier": None,
        "used_patient_memory": False,
        "hospital_finder_called": False,
        "answer": "",
    }

    if mode == "rag_chat":
        # Existing code ONLY passes question + disease_context (drops patient_history & prior_chat)
        tools_called.extend(["rag_retrieve", "rag_answer"])
        out = _traced_rag_answer(question, disease_context=inputs.get("disease_context"))
        result["retrieved_topics"] = out["retrieval"]["retrieved_topics"]
        result["top_bm25_score"] = out["retrieval"]["top_bm25_score"]
        result["backend"] = out["retrieval"]["backend"]
        result["answer"] = out["answer"]

    elif mode == "bot_rag_chat":
        # Existing Telegram router in State.RAG_CHAT
        uid = f"eval_{inputs['case_id']}"
        sess = bot_store.reset(uid)
        sess.district = inputs.get("district", "rangpur")
        sess.district_coords = get_district_coords(sess.district)
        sess.last_prediction = {"disease": "Tinea", "confidence": 0.85}
        sess.last_tier_result = {"tier": inputs.get("prior_tier", 1), "urgency_label": "NON-URGENT"}
        sess.state = State.RAG_CHAT

        tools_called.extend(["bot_handle_event", "rag_answer"])
        out = _traced_bot_event(uid, question, f"m_{inputs['case_id']}")
        result["final_tier"] = (sess.last_tier_result or {}).get("tier")
        result["hospital_finder_called"] = sess.nearest_hospital is not None
        result["answer"] = out["reply"]

    elif mode == "triage_repeat":
        # Existing stateless compute_tier() has no episodic memory
        tools_called.append("triage_engine")
        out = _traced_triage(inputs["disease_class"], inputs["confidence"], question)
        result["final_tier"] = out["tier"]
        result["answer"] = f"Tier {out['tier']} ({out['urgency_label']}) → {out['facility']}"

    elif mode == "triage_only":
        tools_called.append("triage_engine")
        out = _traced_triage(inputs["disease_class"], inputs["confidence"], question)
        result["final_tier"] = out["tier"]
        result["answer"] = f"Tier {out['tier']} ({out['urgency_label']}) → {out['facility']}"

    elif mode == "bot_new_user":
        uid = f"eval_{inputs['case_id']}"
        bot_store.reset(uid)
        tools_called.append("bot_handle_event")
        _traced_bot_event(uid, question, f"m1_{inputs['case_id']}")
        out2 = _traced_bot_event(uid, question, f"m2_{inputs['case_id']}")
        result["answer"] = out2["reply"]

    result["total_latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result


# ---------------------------------------------------------------------------
# LangSmith Evaluators (Automatic Scoring Visible in LangSmith Experiments)
# ---------------------------------------------------------------------------

def eval_retrieval_relevance(outputs: dict, reference_outputs: dict) -> dict:
    kw = reference_outputs.get("expected_topic_keyword")
    if not kw:
        return {"key": "retrieval_relevance", "score": 1, "comment": "N/A for non-RAG case"}
    topics = " ".join(outputs.get("retrieved_topics", [])).lower()
    hit = kw.lower() in topics
    return {
        "key": "retrieval_relevance",
        "score": 1 if hit else 0,
        "comment": f"Expected '{kw}' in retrieved topics -> {'HIT' if hit else 'MISS (wrong chunks)'}",
    }


def eval_clinical_safety(outputs: dict, reference_outputs: dict) -> dict:
    expected_tier = reference_outputs.get("expected_tier")
    if expected_tier is None:
        return {"key": "clinical_safety", "score": 1, "comment": "N/A for non-triage case"}
    actual = outputs.get("final_tier")
    ok = actual is not None and actual >= expected_tier
    return {
        "key": "clinical_safety",
        "score": 1 if ok else 0,
        "comment": f"Expected Tier >= {expected_tier}, got Tier {actual}",
    }


def eval_tool_routing(outputs: dict, reference_outputs: dict) -> dict:
    expected_tools = set(reference_outputs.get("expected_tools", []))
    actual_tools = set(outputs.get("tools_called", []))
    missing = expected_tools - actual_tools
    ok = len(missing) == 0
    return {
        "key": "tool_routing_correct",
        "score": 1 if ok else 0,
        "comment": (
            f"All expected tools called: {sorted(actual_tools)}"
            if ok
            else f"Missing tools: {sorted(missing)} | Actually called: {sorted(actual_tools)}"
        ),
    }


# ---------------------------------------------------------------------------
# Main: Upload Dataset to LangSmith + Run Tracked Evaluation Experiment
# ---------------------------------------------------------------------------

def main() -> None:
    client = Client()
    print("=" * 74)
    print(f"  Uploading / Syncing 10 Test Cases to LangSmith Dataset: {DATASET_NAME}")
    print("=" * 74)

    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        existing_count = len(list(client.list_examples(dataset_id=dataset.id)))
        if existing_count != len(TEST_CASES):
            client.delete_dataset(dataset_id=dataset.id)
            dataset = client.create_dataset(
                dataset_name=DATASET_NAME,
                description="10 clinical test cases for SkinAI Bangladesh (Before vs After Upgrade)",
            )
            client.create_examples(
                inputs=[tc["inputs"] for tc in TEST_CASES],
                outputs=[tc["outputs"] for tc in TEST_CASES],
                dataset_id=dataset.id,
            )
            print(f"  Re-created dataset with {len(TEST_CASES)} test cases.")
        else:
            print(f"  Dataset '{DATASET_NAME}' already has {existing_count} test cases.")
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="10 clinical test cases for SkinAI Bangladesh (Before vs After Upgrade)",
        )
        client.create_examples(
            inputs=[tc["inputs"] for tc in TEST_CASES],
            outputs=[tc["outputs"] for tc in TEST_CASES],
            dataset_id=dataset.id,
        )
        print(f"  Created dataset '{DATASET_NAME}' with {len(TEST_CASES)} test cases.")

    print("\n  Running LangSmith Evaluation Experiment on EXISTING unmodified code...")
    exp_results = evaluate(
        run_existing_system,
        data=DATASET_NAME,
        evaluators=[eval_retrieval_relevance, eval_clinical_safety, eval_tool_routing],
        experiment_prefix="baseline-before-upgrade",
        max_concurrency=1,  # sequential so traces are clean & easy to read
    )

    # Save concise local summary too
    summary_rows = []
    for row in exp_results._results:
        inp = row["example"].inputs
        out = row["run"].outputs or {}
        scores = {r.key: r.score for r in row["evaluation_results"]["results"]}
        summary_rows.append({
            "case_id": inp["case_id"],
            "category": inp["category"],
            "question": inp["question"],
            "tools_called": out.get("tools_called", []),
            "latency_ms": out.get("total_latency_ms"),
            "scores": scores,
        })

    out_path = os.path.join(_ROOT, "results", "langsmith_10_cases_baseline.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "dataset_name": DATASET_NAME,
                "experiment_name": exp_results.experiment_name,
                "cases": summary_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 74)
    print(f"  Experiment Name : {exp_results.experiment_name}")
    print(f"  Local Report    : {out_path}")
    print("  Open LangSmith  : https://smith.langchain.com -> Datasets & Experiments")
    print("=" * 74)


if __name__ == "__main__":
    main()
