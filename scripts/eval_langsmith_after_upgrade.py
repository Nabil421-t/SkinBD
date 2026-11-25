"""
scripts/eval_langsmith_after_upgrade.py
────────────────────────────────────────
Runs the EXACT SAME 10-case LangSmith Dataset (`SkinAI-Baseline-10-Cases`)
against the UPGRADED SkinAI system:
  1. Agent Memory Layer (Working + Episodic + Preference Memory)
  2. True Hybrid Search (Dense + BM25 with BN→EN Medical Synonyms + RRF) + Clinical Reranker
  3. Clinical Agent Orchestrator (Dynamic Tool Routing + Mid-Chat Re-Triage + Hospital Cache)

Logs:
  • A comparative Experiment (`after-upgrade-...`) in LangSmith Datasets & Experiments
    so you can compare Before vs. After side-by-side!
  • ALSO emits 10 individual top-level Traces directly into your Tracing Project
    (`SkinAI-Bangladesh-Baseline`) so you can click each question directly in Tracing Projects.
  • Saves `results/langsmith_10_cases_after_upgrade.json`.

Usage:
    source .venv/bin/activate
    python scripts/eval_langsmith_after_upgrade.py
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

from langsmith import traceable
from langsmith.evaluation import evaluate

from memory.agent_memory import memory_store
from orchestrator.clinical_orchestrator import orchestrate_turn
from rag import retriever as rag_retriever
from scripts.eval_langsmith_10_cases import (
    DATASET_NAME,
    TEST_CASES,
    eval_clinical_safety,
    eval_retrieval_relevance,
    eval_tool_routing,
)

# Pre-load RAG index
rag_retriever.load_index()


@traceable(name="SkinAI_Upgraded_Orchestrator_Run", run_type="chain")
def run_upgraded_system(inputs: dict) -> dict[str, Any]:
    """
    Runs each of the 10 test cases through the NEW Clinical Agent Orchestrator
    (Agent Memory + True Hybrid/Rerank RAG + 4-Signal Triage + Cached Hospital Finder).
    """
    case_id = inputs["case_id"]
    uid = f"upgraded_{case_id}"
    memory_store.clear(uid)

    mode = inputs["mode"]
    question = inputs["question"]

    # Seed episodic prior visit for TC-07 (repeat visit after 5 days unimproved)
    if mode == "triage_repeat" and inputs.get("prior_visit_disease"):
        mem = memory_store.get(uid)
        mem.record_visit(
            disease=inputs["prior_visit_disease"],
            confidence=0.85,
            tier=1,
            history={"duration": "5 days", "previous_treatment": "pharmacy cream"},
            district=inputs.get("district"),
        )

    # Seed prior visit for TC-06 (mid-chat emergency red flag from prior Tier-1 Tinea)
    if mode == "bot_rag_chat":
        mem = memory_store.get(uid)
        mem.record_visit(
            disease="Tinea",
            confidence=0.85,
            tier=inputs.get("prior_tier", 1),
            district=inputs.get("district", "rangpur"),
        )

    out = orchestrate_turn(
        user_id=uid,
        user_message=question,
        disease_class=inputs.get("disease_class"),
        confidence=inputs.get("confidence"),
        disease_context=inputs.get("disease_context"),
        district=inputs.get("district") or None,
        patient_history=inputs.get("patient_history"),
        prior_chat=inputs.get("prior_chat"),
        prior_tier=inputs.get("prior_tier"),
        record_new_visit=(mode in ("triage_repeat", "triage_only")),
    )

    return {
        "case_id": case_id,
        "category": inputs["category"],
        "question": question,
        "tools_called": out["tools_called"],
        "retrieved_topics": out["retrieved_topics"],
        "final_tier": out["final_tier"],
        "used_patient_memory": out["used_patient_memory"],
        "hospital_finder_called": out["hospital_finder_called"],
        "answer": out["answer"],
        "total_latency_ms": out["total_latency_ms"],
    }


def main() -> None:
    print("=" * 74)
    print(f"  Running UPGRADED System on LangSmith Dataset: {DATASET_NAME}")
    print("=" * 74)

    # 1. Run comparative Experiment on the SAME LangSmith dataset
    exp_results = evaluate(
        run_upgraded_system,
        data=DATASET_NAME,
        evaluators=[eval_retrieval_relevance, eval_clinical_safety, eval_tool_routing],
        experiment_prefix="after-upgrade-memory-hybrid-orch",
        max_concurrency=1,
    )

    # 2. Also emit 10 clean standalone traces directly into Tracing Projects
    #    so the user can click TC-01..TC-10 directly under "Tracing Projects"!
    print("\n  Sending 10 named traces to Tracing Project:", os.getenv("LANGSMITH_PROJECT"))
    for tc in TEST_CASES:
        cid = tc["inputs"]["case_id"]
        q_short = tc["inputs"]["question"][:45]

        @traceable(name=f"{cid}: {q_short}", run_type="chain")
        def _tracked_question(inp: dict) -> dict:
            return run_upgraded_system(inp)

        _tracked_question(tc["inputs"])

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
            "final_tier": out.get("final_tier"),
            "latency_ms": out.get("total_latency_ms"),
            "scores": scores,
        })

    out_path = os.path.join(_ROOT, "results", "langsmith_10_cases_after_upgrade.json")
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
    print(f"  Upgraded Experiment : {exp_results.experiment_name}")
    print(f"  Local Report        : {out_path}")
    print("  Compare Before/After: https://smith.langchain.com -> Datasets & Experiments")
    print("=" * 74)


if __name__ == "__main__":
    main()
