"""
Day 5 (B) - Check the checker: hand-label whether each Day 4 drafted step is supported by its cited
sources, then compare with the LLM judge (shown only at the end, so it can't bias you).
Run: python scripts/day5_check_judge.py      (press q any time; run again to continue)
Saves: reports/judge_check.csv and reports/day5_judge_check.md
"""
import json
import os
import sys
import textwrap
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from day2_search import Searcher, ticket_text  # noqa: E402
from day4_resolve import CACHE, K, SEED, validate  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
OUT = Path("reports/judge_check.csv")


def wrap(t, width=90, indent="      "):
    return textwrap.fill(str(t), width, initial_indent=indent, subsequent_indent=indent)


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
        key = f"LLM with sources|{SEED}|{i}|{model}"
        if key not in cache:
            continue
        idxs = s.hybrid_top(t["text"], K)
        draft, _, _ = validate(cache[key]["raw"], {f"KB-{j}" for j in idxs})
        if not draft or not draft["steps"]:
            continue
        j = cache.get(f"judge|LLM with sources|{SEED}|{i}|{model}", {}).get("raw") or {}
        judge[i] = (int(j.get("supported_steps", 0) or 0), int(j.get("total_steps", 0) or len(draft["steps"])))
        for n, st in enumerate(draft["steps"], 1):
            items.append({"ticket": i, "step_no": n, "step": st["text"], "sources": ",".join(st["sources"]),
                          "ticket_text": t["text"]})

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
            a = input("\n  Does the cited source support this step?  [y]es / [n]o / [u]nsure / [q]uit > ").strip().lower()
        if a == "q":
            break
        rows.append({"ticket": it["ticket"], "step_no": it["step_no"], "step": it["step"],
                     "sources": it["sources"], "human": a})
        pd.DataFrame(rows).to_csv(OUT, index=False)

    df = pd.DataFrame(rows)
    if df.empty:
        return
    print(f"\nChecked {len(df)} of {len(items)} steps.")
    if len(df) < len(items):
        print("Run again to continue; the comparison is printed when all steps are checked.")
        return

    sure = df[df["human"].isin(["y", "n"])]
    human_rate = (sure["human"] == "y").mean()
    j_sup = sum(v[0] for v in judge.values())
    j_tot = sum(v[1] for v in judge.values())
    per_ticket = []
    for tk, g in df.groupby("ticket"):
        h = int((g["human"] == "y").sum())
        jsup, jtot = judge.get(tk, (0, len(g)))
        per_ticket.append({"Ticket": tk, "Steps": len(g), "You: supported": h, "Judge: supported": min(jsup, len(g)),
                           "Agree": "yes" if h == min(jsup, len(g)) else "no"})
    pt = pd.DataFrame(per_ticket)
    report = (
        "# Day 5 (B) - Checking the LLM judge by hand\n\n"
        f"- Steps checked: {len(df)} across {df['ticket'].nunique()} tickets "
        f"(yes {int((df['human'] == 'y').sum())}, no {int((df['human'] == 'n').sum())}, "
        f"unsure {int((df['human'] == 'u').sum())}).\n"
        f"- **Human: supported** {human_rate:.0%} of steps (excluding unsure).\n"
        f"- **LLM judge: supported** {j_sup / j_tot:.0%} of steps.\n"
        f"- Tickets where human and judge counted the same number of supported steps: "
        f"{(pt['Agree'] == 'yes').sum()} of {len(pt)}.\n\n"
        + pt.to_markdown(index=False) + "\n\n"
        "The judge was hidden during labelling. One reviewer, small sample.\n"
    )
    Path("reports/day5_judge_check.md").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
