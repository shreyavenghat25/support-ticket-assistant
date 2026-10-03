"""The ticket assistant: search, route (with confidence lanes), and draft a grounded reply."""
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import day3_triage  # noqa: E402
from day2_search import Searcher  # noqa: E402
from day3_triage import LLM  # noqa: E402
from day4_resolve import sources_block, validate  # noqa: E402
from day5_fix import NEW_RULES  # noqa: E402

day3_triage.MAX_RETRIES = 2  # an API request should fail fast, not wait minutes

K = 5
RULES = NEW_RULES + """
- Also add "sentiment": the customer's mood, one of "calm", "concerned", "frustrated", "angry"."""


def lane_for(agreement):
    """Day 5 decision D8: confidence lanes from agreement among the 5 similar tickets."""
    if agreement >= 4:
        return "auto"
    if agreement == 3:
        return "suggest"
    return "manual"


def subject_of(value):
    return value if isinstance(value, str) and value.strip() else "(no subject)"


class Assistant:
    def __init__(self):
        self.search = Searcher()
        self.kb = self.search.kb
        self.llm = LLM()

    def analyse(self, text):
        t0 = time.perf_counter()
        idxs = self.search.hybrid_top(text, K)
        nb = self.kb.iloc[idxs]
        search_ms = (time.perf_counter() - t0) * 1000

        dept_votes = Counter(nb["queue"]).most_common()
        prio_votes = Counter(nb["priority"]).most_common()
        type_votes = Counter(nb["type"]).most_common()
        agreement = dept_votes[0][1]
        routing = {
            "departments": [{"name": d, "votes": v} for d, v in dept_votes[:2]],
            "agreement": f"{agreement} of {K}",
            "lane": lane_for(agreement),
            "priority": prio_votes[0][0],
            "priority_confirmed": prio_votes[0][1] == K,
            "type": type_votes[0][0],
        }

        allowed = {f"KB-{i}" for i in idxs}
        raw, usage = self.llm.json(
            f"{RULES}\n\nPAST TICKETS:\n{sources_block(self.kb, idxs)}\n\nNEW TICKET:\n{text}")
        draft, removed, _ = validate(raw, allowed)
        if draft is None:
            draft = {"status": "unavailable", "summary": "", "steps": [], "clarifying_questions": [],
                     "confidence": None}
        sentiment = raw.get("sentiment") if isinstance(raw, dict) else None

        similar = [{"id": f"KB-{i}", "subject": subject_of(self.kb.iloc[i]["subject"]),
                    "department": self.kb.iloc[i]["queue"], "language": self.kb.iloc[i]["language"],
                    "answer": str(self.kb.iloc[i]["answer"])[:400]} for i in idxs]
        return {
            "routing": routing,
            "sentiment": sentiment,
            "reply": {k: draft.get(k) for k in ["status", "summary", "steps", "clarifying_questions", "confidence"]},
            "similar_tickets": similar,
            "diagnostics": {"search_ms": round(search_ms), "llm_ms": round(usage["ms"]),
                            "total_ms": round((time.perf_counter() - t0) * 1000),
                            "tokens_in": usage["in"], "tokens_out": usage["out"],
                            "invalid_citations_removed": removed, "llm_ok": raw is not None},
        }
