"""
Day 5 (D) - Hand-check the drafts written with the improved prompt (the 'already tried' rule), so the
post-fix result is a human number, not only the LLM judge's.
Run after scripts/day5_fix.py:  python scripts/day5_check_new_drafts.py   (press q any time; run again to continue)
Saves: reports/judge_check_v2.csv and reports/day5_new_drafts_check.md
"""
import json
import os
import sys
import textwrap
from math import sqrt
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from day2_search import Searcher, ticket_text  # noqa: E402
from day4_resolve import CACHE, K, SEED, validate  # noqa: E402

N = 50
OUT = Path("reports/judge_check_v2.csv")


def wrap(t, width=90, indent="      "):
    return textwrap.fill(str(t), width, initial_indent=indent, subsequent_indent=indent)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    from dotenv import load_dotenv
    load_dotenv()
    model = os.getenv("GEMINI_MODEL")
    cache = {json.loads(l)["key"]: json.loads(l) for l in CACHE.read_text().splitlines()}
    s = Searcher()
    test = pd.read_parquet("data/test_tickets.parquet").sample(N, random_state=SEED).reset_index(drop=True)
    test["text"] = ticket_text(test)

    items, judge = [], {}
    for i, t in test.iterrows():
        key = f"v2|{SEED}|{i}|{model}"
        if key not in cache:
            continue
        idxs = s.hybrid_top(t["text"], K)
        draft, _, _ = validate(cache[key]["raw"], {f"KB-{j}" for j in idxs})
        if not draft or not draft["steps"]:
            continue
        j = cache.get(f"judge_v2|new|{SEED}|{i}|{model}", {}).get("raw") or {}
        judge[i] = min(int(j.get("supported_steps", 0) or 0), len(draft["steps"]))
        for n, st in enumerate(draft["steps"], 1):
            items.append({"ticket": i, "step_no": n, "step": st["text"], "sources": ",".join(st["sources"]),
                          "ticket_text": t["text"]})
    if not items:
        print("No improved drafts found in the cache. Run scripts/day5_fix.py first.")
        return

    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["ticket", "step_no", "human"])
    seen = {(int(r.ticket), int(r.step_no)) for r in done.itertuples()}
    print(f"\n{len(items)} steps from {len(judge)} drafted tickets. Already checked: {len(seen)}.")

    rows = done.to_dict("records")
    for k, it in enumerate(items, 1):
        if (it["ticket"], it["step_no"]) in seen:
            continue
        print("\n" + "=" * 90)
        print(f"Step {k} of {len(items)}")
        print("  CUSTOMER TICKET:")
        print(wrap(it["ticket_text"][:400]))
        print(f"\n  DRAFTED STEP: {it['step']}")
        for sid in it["sources"].split(","):
            r = s.kb.iloc[int(sid.split("-")[1])]
            print(f"\n  CITED {sid} - problem:")
            print(wrap(str(r["text"])[:300]))
            print(f"  CITED {sid} - agent answer:")
            print(wrap(str(r["answer"])[:600]))
        a = ""
        while a not in ("y", "n", "u", "q"):
            a = input("\n  Is this step supported by the source AND not something the customer already tried?"
                      "  [y]es / [n]o / [u]nsure / [q]uit > ").strip().lower()
        if a == "q":
            break
        rows.append({"ticket": it["ticket"], "step_no": it["step_no"], "step": it["step"],
                     "sources": it["sources"], "human": a})
        pd.DataFrame(rows).to_csv(OUT, index=False)

    df = pd.DataFrame(rows)
    if len(df) < len(items):
        print(f"\nChecked {len(df)} of {len(items)} steps. Run again to continue.")
        return

    sure = df[df["human"].isin(["y", "n"])]
    yes, n_sure = int((sure["human"] == "y").sum()), len(sure)
    lo, hi = wilson(yes, n_sure)
    j_sup, j_tot = sum(judge.values()), len(df)
    report = (
        "# Day 5 (D) - Hand check of the improved drafts\n\n"
        f"- Steps checked: {len(df)} across {df['ticket'].nunique()} tickets "
        f"(yes {yes}, no {n_sure - yes}, unsure {int((df['human'] == 'u').sum())}).\n"
        f"- **Human: supported {yes / n_sure:.0%}** of steps, excluding unsure "
        f"(95% CI {lo:.0%}-{hi:.0%}).\n"
        f"- Improved LLM judge on the same steps: {j_sup / j_tot:.0%}.\n"
        f"- Before the fix (Day 5 B): human 84% (32 of 38).\n\n"
        "The judge's scores were not shown during labelling. One reviewer, small sample.\n"
    )
    Path("reports/day5_new_drafts_check.md").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
