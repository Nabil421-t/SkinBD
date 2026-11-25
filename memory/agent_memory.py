"""
memory/agent_memory.py
──────────────────────
Consent-friendly Agent Memory Layer for SkinAI Bangladesh.

Three memory tiers:
  1. Working / Clinical Profile Memory — preserves the 9-field voice history
     (age, duration, affected area, symptoms, prior treatment) across tabs & turns.
  2. Episodic Longitudinal Visit Memory — tracks the last 5 visits per user
     (disease, confidence, tier, symptoms) and detects unimproved repeat visits
     to feed Signal 4 in severity/engine.py.
  3. User Preference & Conversational Memory — remembers district, preferred
     language (bn/en), and the last 6 chat turns for multi-turn continuity.

Auto-expires after 7 days (RETENTION_SECONDS), matching cases/store.py privacy.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

MEMORY_TTL_SECONDS = 7 * 24 * 3600  # 7 days retention


@dataclass
class VisitEpisode:
    timestamp: float
    disease: str
    confidence: float
    tier: int
    symptoms: list[str] = field(default_factory=list)
    duration: str = ""
    progression: str = ""
    previous_treatment: str = ""


@dataclass
class PatientMemory:
    user_id: str
    preferred_lang: Optional[str] = None
    district: Optional[str] = None
    patient_age: Optional[str] = None
    clinical_profile: dict = field(default_factory=dict)
    visits: list[VisitEpisode] = field(default_factory=list)
    chat_turns: list[dict[str, str]] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)

    def update_profile(
        self,
        patient_history: Optional[dict] = None,
        district: Optional[str] = None,
        preferred_lang: Optional[str] = None,
    ) -> None:
        if district:
            self.district = district.strip()
        if preferred_lang:
            self.preferred_lang = preferred_lang
        if patient_history:
            for k, v in patient_history.items():
                if v:
                    self.clinical_profile[k] = v
            if patient_history.get("patient_age"):
                self.patient_age = str(patient_history["patient_age"])
        self.updated_at = time.time()

    def record_visit(
        self,
        disease: str,
        confidence: float,
        tier: int,
        history: Optional[dict] = None,
        district: Optional[str] = None,
    ) -> VisitEpisode:
        self.update_profile(patient_history=history, district=district)
        h = history or {}
        ep = VisitEpisode(
            timestamp=time.time(),
            disease=disease,
            confidence=float(confidence),
            tier=int(tier),
            symptoms=list(h.get("symptoms", [])) if isinstance(h.get("symptoms"), list) else [],
            duration=str(h.get("duration", "")),
            progression=str(h.get("progression", "")),
            previous_treatment=str(h.get("previous_treatment", "")),
        )
        self.visits.append(ep)
        self.visits = self.visits[-5:]
        self.updated_at = time.time()
        return ep

    def add_chat_turn(
        self,
        role: str,
        content: str,
        lang: Optional[str] = None,
    ) -> None:
        if lang:
            self.preferred_lang = lang
        if content and content.strip():
            self.chat_turns.append({"role": role, "content": content.strip()})
            self.chat_turns = self.chat_turns[-6:]
        self.updated_at = time.time()

    def has_unimproved_repeat_visit(self, current_disease: str) -> bool:
        """
        True if the patient's previous visit had the same skin condition,
        indicating persistence / non-response to prior lower-tier care.
        """
        if not self.visits or not current_disease:
            return False
        norm_curr = current_disease.strip().lower().replace(" ", "_")
        last_norm = self.visits[-1].disease.strip().lower().replace(" ", "_")
        return last_norm == norm_curr

    def format_for_prompt(self) -> str:
        """Build a concise, structured memory note for injection into the RAG prompt."""
        parts: list[str] = []

        pref_bits = []
        if self.patient_age:
            pref_bits.append(f"Age: {self.patient_age}")
        if self.district:
            pref_bits.append(f"District: {self.district}")
        if self.preferred_lang:
            pref_bits.append(f"Preferred language: {'Bengali' if self.preferred_lang == 'bn' else 'English'}")
        if pref_bits:
            parts.append("- User preferences & profile: " + " | ".join(pref_bits))

        if self.clinical_profile:
            h_items = [
                f"{k}: {', '.join(v) if isinstance(v, list) else v}"
                for k, v in self.clinical_profile.items()
                if v
            ]
            if h_items:
                parts.append("- Clinical history (voice/form): " + " | ".join(h_items))

        if self.visits:
            v_summary = "; ".join(
                f"Visit {i + 1}: {v.disease.replace('_', ' ')} (Tier {v.tier}, {v.confidence:.0%})"
                for i, v in enumerate(self.visits[-3:])
            )
            parts.append("- Longitudinal visit memory: " + v_summary)

        if self.chat_turns:
            turns = "\n".join(
                f"  {t['role'].capitalize()}: {t['content'][:200]}"
                for t in self.chat_turns[-4:]
            )
            parts.append("- Recent conversation memory:\n" + turns)

        if not parts:
            return ""
        return "Patient memory & conversation context:\n" + "\n".join(parts) + "\n\n"

    def snapshot(self) -> dict:
        return {
            "user_id": self.user_id,
            "district": self.district,
            "preferred_lang": self.preferred_lang,
            "patient_age": self.patient_age,
            "visit_count": len(self.visits),
            "last_disease": self.visits[-1].disease if self.visits else None,
            "last_tier": self.visits[-1].tier if self.visits else None,
            "chat_turns": len(self.chat_turns),
        }


class AgentMemoryStore:
    """Thread-safe in-memory patient memory store with 7-day TTL eviction."""

    def __init__(self) -> None:
        self._store: dict[str, PatientMemory] = {}
        self._lock = threading.Lock()

    def get(self, user_id: str) -> PatientMemory:
        with self._lock:
            mem = self._store.get(user_id)
            if mem is None or (time.time() - mem.updated_at > MEMORY_TTL_SECONDS):
                mem = PatientMemory(user_id=user_id)
                self._store[user_id] = mem
            return mem

    def clear(self, user_id: Optional[str] = None) -> None:
        with self._lock:
            if user_id is None:
                self._store.clear()
            else:
                self._store.pop(user_id, None)


memory_store = AgentMemoryStore()
