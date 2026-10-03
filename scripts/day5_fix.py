"""
Day 5 (C) - Fix the "already tried" failure and re-measure.
Run: python scripts/day5_fix.py          (same 50 tickets as Day 4; cached, so it can resume)
Output: reports/day5_fix.md
"""
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from day2_search import Searcher, ticket_text  # noqa: E402
from day3_triage import LLM  # noqa: E402
from day4_resolve import (JUDGE_RULES, K, RESOLVE_RULES, SEED, cached_call, load_cache,  # noqa: E402
                          show, sources_block, validate)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50

NEW_RULES = RESOLVE_RULES + """
- Never recommend an action that the new ticket or a past ticket says was ALREADY TRIED (for example
  "restarted the system", "passwords were reset", "queries were optimized"). Those are context, not fixes.
- If only already-tried actions remain, set status to "escalate" and ask clarifying questions instead."""

NEW_JUDGE = JUDGE_RULES + """
A step is NOT supported if the past tickets only mention the action as something already tried
(rather than recommending it), or if the new ticket says the customer already did it."""


def judge_prompt(rules, src, ticket, steps):
    steps_txt = "\n".join(f"{k}. {x['text']}" for k, x in enumerate(steps, 1))
    return f"{rules}\n\nPAST TICKETS:\n{src}\n\nNEW TICKET:\n{ticket}\n\nDRAFTED STEPS:\n{steps_txt}"


def judged(raw, n_steps):
    if not isinstance(raw, dict):
        return 0, n_steps
    return min(int(raw.get("supported_steps", 0) or 0), n_steps), n_steps


def main():
    s, llm, cache = Searcher(), LLM(), load_cache()
    model = os.getenv("GEMINI_MODEL")
    test = pd.read_parquet("data/test_tickets.parquet").sample(N, random_state=SEED).reset_index(drop=True)
    test["text"] = ticket_text(test)

    human = pd.read_csv("reports/judge_check.csv")
    verdict = {}
    for tk, g in human.groupby("ticket"):
        ys = (g["human"] == "y").sum()
        verdict[int(tk)] = "good" if ys == len(g) else ("bad" if ys == 0 else "mixed")

    rows, stats = [], {"old": {"draft": 0, "steps": 0, "sup": 0}, "new": {"draft": 0, "steps": 0, "sup": 0}}
    judge_v2_on_old = [0, 0]
    new_drafts = {}
    for i, t in test.iterrows():
        print(f"Ticket {i + 1}/{N}")
        idxs = s.hybrid_top(t["text"], K)
        allowed = {f"KB-{j}" for j in idxs}
        src = sources_block(s.kb, idxs)

        old_raw, _ = cached_call(cache, llm, f"LLM with sources|{SEED}|{i}|{model}",
                                 f"{RESOLVE_RULES}\n\nPAST TICKETS:\n{src}\n\nNEW TICKET:\n{t['text']}")
        old, _, _ = validate(old_raw, allowed)
        new_raw, _ = cached_call(cache, llm, f"v2|{SEED}|{i}|{model}",
                                 f"{NEW_RULES}\n\nPAST TICKETS:\n{src}\n\nNEW TICKET:\n{t['text']}")
        new, _, _ = validate(new_raw, allowed)
        new_drafts[i] = new

        for name, d in [("old", old), ("new", new)]:
            if d and d["steps"]:
                stats[name]["draft"] += 1
                jraw, _ = cached_call(cache, llm, f"judge_v2|{name}|{SEED}|{i}|{model}",
                                     judge_prompt(NEW_JUDGE, src, t["text"], d["steps"]))
                sup, tot = judged(jraw, len(d["steps"]))
                stats[name]["sup"] += sup
                stats[name]["steps"] += tot
                if name == "old" and i in verdict:
                    judge_v2_on_old[0] += sup
                    judge_v2_on_old[1] += tot

        if i in verdict:
            rows.append({"Ticket": i, "Your verdict (old draft)": verdict[i],
                         "Old prompt": "draft" if old and old["steps"] else "escalate",
                         "New prompt": "draft" if new and new["steps"] else "escalate",
                         "New steps": len(new["steps"]) if new else 0})

    cmp = pd.DataFrame(rows)
    bad = cmp[cmp["Your verdict (old draft)"] == "bad"]
    good = cmp[cmp["Your verdict (old draft)"] == "good"]
    summary = pd.DataFrame([
        {"Prompt": p, "Drafted (of 50)": stats[k]["draft"], "Escalated": f"{(N - stats[k]['draft']) / N:.0%}",
         "Steps": stats[k]["steps"],
         "Improved judge: supported": f"{stats[k]['sup'] / stats[k]['steps']:.0%}" if stats[k]["steps"] else "-"}
        for p, k in [("Old (Day 4)", "old"), ("New (already-tried rule)", "new")]])
    judges = pd.DataFrame([
        {"Who checked the old drafts": "Original LLM judge", "Supported": "100%"},
        {"Who checked the old drafts": "Improved LLM judge",
         "Supported": f"{judge_v2_on_old[0] / judge_v2_on_old[1]:.0%}" if judge_v2_on_old[1] else "-"},
        {"Who checked the old drafts": "You (hand check)", "Supported": "84%"}])

    report = (
        "# Day 5 (C) - Fixing the 'already tried' failure\n\n"
        "## Did the fix work on the drafts you hand-checked?\n\n" + cmp.to_markdown(index=False) + "\n\n"
        f"- Bad drafts (all steps already-tried) now escalated: **{(bad['New prompt'] == 'escalate').sum()} of {len(bad)}**\n"
        f"- Good drafts still drafted (not over-escalated): **{(good['New prompt'] == 'draft').sum()} of {len(good)}**\n\n"
        "## Overall on the 50 tickets\n\n" + summary.to_markdown(index=False) + "\n\n"
        "## Does the improved judge agree with you?\n\n" + judges.to_markdown(index=False) + "\n\n"
        "- Same 50 tickets and sources as Day 4; only the instructions changed.\n"
        "- Small sample; new drafts for the previously bad tickets are printed below the table in the terminal.\n"
    )
    Path("reports/day5_fix.md").write_text(report)
    print("\n" + report)
    print("New drafts for the tickets you marked bad or mixed:")
    for tk in cmp[cmp["Your verdict (old draft)"] != "good"]["Ticket"]:
        print(f"\n--- Ticket {tk} ---")
        show(new_drafts[tk])


if __name__ == "__main__":
    main()
