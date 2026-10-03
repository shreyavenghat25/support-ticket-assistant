"""
Day 4 - Draft a step-by-step resolution with citations to past tickets.
Usage:
    python scripts/day4_resolve.py try "My broadband drops every evening around 8"
    python scripts/day4_resolve.py eval        # 50 test tickets
Answers are cached in data/day4_cache.jsonl. Output: reports/day4_resolution.md
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from day2_search import Searcher, embed, ticket_text  # noqa: E402
from day3_triage import LLM  # noqa: E402

CACHE = Path("data/day4_cache.jsonl")
SEED = 1
PAUSE_SECONDS = 4
K = 5

RESOLVE_RULES = """You help a support agent reply to a customer ticket.
Return JSON only:
{"status": "draft" or "escalate",
 "summary": <one sentence describing the customer's problem>,
 "steps": [{"text": <one concrete action>, "sources": [<source ids>]}],
 "clarifying_questions": [<questions the agent should ask the customer>],
 "confidence": "low" | "medium" | "high"}
Rules:
- Base every step ONLY on the past tickets below. Each step must cite the source ids (e.g. "KB-123") it uses.
- Ignore past answers that only ask the customer for more details; do not copy them as steps.
- If the past tickets contain no real fix for this problem, set status to "escalate", leave steps empty,
  and put the details an agent should request in clarifying_questions.
- Write in the same language as the new ticket (English or German). Maximum 5 steps."""

NO_SOURCE_RULES = """You help a support agent reply to a customer ticket.
Return JSON only:
{"status": "draft" or "escalate",
 "summary": <one sentence describing the customer's problem>,
 "steps": [{"text": <one concrete action>, "sources": []}],
 "clarifying_questions": [<questions the agent should ask the customer>],
 "confidence": "low" | "medium" | "high"}
Write in the same language as the new ticket (English or German). Maximum 5 steps."""

JUDGE_RULES = """You check whether a drafted support reply is supported by the provided past tickets.
For each step, decide if the past tickets (their problems and agent answers) support that action.
Return JSON only: {"supported_steps": <int>, "total_steps": <int>, "helpfulness": <1-5>}
helpfulness: 1 = useless or generic, 5 = specific and directly actionable for this ticket."""


def sources_block(kb, idxs):
    parts = []
    for i in idxs:
        r = kb.iloc[i]
        parts.append(f"[KB-{i}] PROBLEM: {str(r['text'])[:500]}\nAGENT ANSWER: {str(r['answer'])[:600]}")
    return "\n\n".join(parts)


def validate(draft, allowed_ids):
    if not isinstance(draft, dict):
        return None, 0, 0
    steps = draft.get("steps") or []
    kept, bad = [], 0
    for s in steps:
        if not isinstance(s, dict) or not str(s.get("text", "")).strip():
            continue
        cites = [str(c) for c in (s.get("sources") or [])]
        if allowed_ids is not None and (not cites or any(c not in allowed_ids for c in cites)):
            bad += 1
            continue
        kept.append({"text": str(s["text"]).strip(), "sources": cites})
    draft["steps"] = kept
    if not kept and draft.get("status") == "draft" and allowed_ids is not None:
        draft["status"] = "escalate"
    return draft, bad, len(steps)


def draft_text(draft):
    if not draft:
        return ""
    parts = [str(draft.get("summary", ""))] + [s["text"] for s in draft.get("steps", [])]
    parts += [str(q) for q in draft.get("clarifying_questions", []) or []]
    return " ".join(p for p in parts if p)


def show(draft):
    if not draft:
        print("  (no valid output)")
        return
    print(f"  Status: {draft.get('status')}   Confidence: {draft.get('confidence')}")
    print(f"  Summary: {draft.get('summary')}")
    for n, s in enumerate(draft.get("steps", []), 1):
        cites = ", ".join(s["sources"]) if s["sources"] else "no source"
        print(f"  {n}. {s['text']}  [{cites}]")
    qs = draft.get("clarifying_questions") or []
    if qs:
        print("  Ask the customer:")
        for q in qs:
            print(f"   - {q}")


def try_one(text):
    s, llm = Searcher(), LLM()
    idxs = s.hybrid_top(text, K)
    allowed = {f"KB-{i}" for i in idxs}
    print("\nSimilar past tickets used as sources:")
    for i in idxs:
        print(f"  KB-{i}: {str(s.kb.iloc[i]['subject'])[:80]}")
    raw, u = llm.json(f"{RESOLVE_RULES}\n\nPAST TICKETS:\n{sources_block(s.kb, idxs)}\n\nNEW TICKET:\n{text}")
    draft, bad, total = validate(raw, allowed)
    print(f"\n===== Suggested reply ({u['ms']:.0f} ms, {u['in']} in / {u['out']} out tokens) =====")
    show(draft)
    if bad:
        print(f"\n  Removed {bad} of {total} steps with missing or invalid citations.")


def load_cache():
    if not CACHE.exists():
        return {}
    return {json.loads(l)["key"]: json.loads(l) for l in CACHE.read_text().splitlines()}


def cached_call(cache, llm, key, prompt):
    if key in cache:
        return cache[key]["raw"], cache[key]["usage"]
    raw, u = llm.json(prompt)
    with CACHE.open("a") as f:
        f.write(json.dumps({"key": key, "raw": raw, "usage": u}) + "\n")
    time.sleep(PAUSE_SECONDS)
    return raw, u


def evaluate(n):
    s, llm, cache = Searcher(), LLM(), load_cache()
    model = os.getenv("GEMINI_MODEL")
    rng = np.random.default_rng(SEED)
    test = pd.read_parquet("data/test_tickets.parquet").sample(n, random_state=SEED).reset_index(drop=True)
    test["text"] = ticket_text(test)

    texts = {"Random past answer": [], "Copy most similar answer": [],
             "LLM without sources": [], "LLM with sources": []}
    stats = {m: {"supported": 0, "steps": 0, "help": [], "escalate": 0, "invalid": 0, "proposed": 0,
                 "failed": 0, "tok_in": [], "tok_out": []} for m in ["LLM without sources", "LLM with sources"]}

    for i, t in test.iterrows():
        print(f"Ticket {i + 1}/{n}")
        idxs = s.hybrid_top(t["text"], K)
        allowed = {f"KB-{j}" for j in idxs}
        src = sources_block(s.kb, idxs)
        texts["Random past answer"].append(str(s.kb.iloc[int(rng.integers(len(s.kb)))]["answer"]))
        texts["Copy most similar answer"].append(str(s.kb.iloc[idxs[0]]["answer"]))

        for m, prompt, ids in [
            ("LLM without sources", f"{NO_SOURCE_RULES}\n\nNEW TICKET:\n{t['text']}", None),
            ("LLM with sources", f"{RESOLVE_RULES}\n\nPAST TICKETS:\n{src}\n\nNEW TICKET:\n{t['text']}", allowed),
        ]:
            raw, u = cached_call(cache, llm, f"{m}|{SEED}|{i}|{model}", prompt)
            draft, bad, proposed = validate(raw, ids)
            st = stats[m]
            st["tok_in"].append(u["in"]); st["tok_out"].append(u["out"])
            if draft is None:
                st["failed"] += 1
                texts[m].append("")
                continue
            st["invalid"] += bad; st["proposed"] += proposed
            st["escalate"] += draft.get("status") == "escalate"
            texts[m].append(draft_text(draft))
            if draft["steps"]:
                steps_txt = "\n".join(f"{k}. {x['text']}" for k, x in enumerate(draft["steps"], 1))
                jraw, _ = cached_call(cache, llm, f"judge|{m}|{SEED}|{i}|{model}",
                                      f"{JUDGE_RULES}\n\nPAST TICKETS:\n{src}\n\nNEW TICKET:\n{t['text']}"
                                      f"\n\nDRAFTED STEPS:\n{steps_txt}")
                if isinstance(jraw, dict):
                    st["supported"] += int(jraw.get("supported_steps", 0) or 0)
                    st["steps"] += int(jraw.get("total_steps", 0) or len(draft["steps"]))
                    if jraw.get("helpfulness"):
                        st["help"].append(float(jraw["helpfulness"]))

    print("Scoring closeness to the real agent answers...")
    real = embed(s.model, test["answer"].astype(str).tolist(), "passage")
    rows = []
    for m, outs in texts.items():
        ok = [k for k, o in enumerate(outs) if o]
        sim = float(np.mean(np.sum(embed(s.model, [outs[k] for k in ok], "passage") * real[ok], axis=1))) if ok else float("nan")
        row = {"Method": m, "Similarity to real answer": f"{sim:.3f}"}
        st = stats.get(m)
        if st:
            row.update({
                "Steps supported by sources": f"{st['supported'] / st['steps']:.0%}" if st["steps"] else "-",
                "Helpfulness (1-5)": f"{np.mean(st['help']):.2f}" if st["help"] else "-",
                "Escalated": f"{st['escalate'] / n:.0%}",
                "Invalid citations removed": f"{st['invalid']}/{st['proposed']}" if m == "LLM with sources" else "n/a",
                "Failed outputs": st["failed"],
                "Avg tokens (in/out)": f"{np.mean(st['tok_in']):.0f}/{np.mean(st['tok_out']):.0f}",
            })
        else:
            row.update({"Steps supported by sources": "-", "Helpfulness (1-5)": "-", "Escalated": "-",
                        "Invalid citations removed": "-", "Failed outputs": "-", "Avg tokens (in/out)": "free"})
        rows.append(row)

    report = (
        f"# Day 4 - Resolution drafting on {n} unseen test tickets\n\n"
        + pd.DataFrame(rows).to_markdown(index=False) + "\n\n"
        f"- Model: `{model}`, temperature 0, JSON output; sources = top {K} hybrid-search tickets.\n"
        "- **Similarity to real answer**: embedding similarity between the draft and the hidden real agent answer "
        "(compare with the Random row). Real answers often only ask for details, so this is a rough signal.\n"
        "- **Steps supported by sources** and **Helpfulness**: LLM judge (same model) given the same 5 past tickets. "
        "Self-judging can be lenient; validate on a hand-checked sample before trusting it.\n"
        "- **Invalid citations removed**: steps citing a ticket that was not provided, dropped automatically.\n"
        f"- Small sample ({n} tickets).\n"
    )
    Path("reports/day4_resolution.md").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "try" and len(sys.argv) > 2:
        try_one(" ".join(sys.argv[2:]))
    elif cmd == "eval":
        evaluate(int(sys.argv[2]) if len(sys.argv) > 2 else 50)
    else:
        print(__doc__)
