"""
Day 5 (E) - Do near-duplicate neighbours inflate routing confidence?
The library had exact copies removed, but near-copies remain, so "5 of 5 agree" can be one ticket 5 times.
This compares the current neighbours with de-duplicated ones (skip a neighbour whose cosine similarity to an
already chosen one is >= 0.90) on the same 1,000 test tickets as Day 5 (A).
Run: python scripts/day5_dedupe_check.py
Output: reports/day5_dedupe_check.md
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from day2_search import Searcher, embed, ticket_text  # noqa: E402

N_TEST, SEED, K, POOL, SIM = 1000, 0, 5, 20, 0.90


def diverse(s, cands, k=K, sim=SIM):
    kept = []
    for c in cands:
        if all(float(s.emb[c] @ s.emb[j]) < sim for j in kept):
            kept.append(c)
        if len(kept) == k:
            break
    return kept + [c for c in cands if c not in kept][: k - len(kept)]


def lanes(df):
    out = []
    for name, rule in [("Auto (4-5 agree)", df["agree"] >= 4), ("Confirm (3)", df["agree"] == 3),
                       ("Manual (0-2)", df["agree"] <= 2)]:
        d = df[rule]
        out.append({"Lane": name, "Share": f"{len(d) / len(df):.0%}",
                    "Top-1 correct": f"{d['ok1'].mean():.1%}" if len(d) else "-",
                    "Top-2 correct": f"{d['ok2'].mean():.1%}" if len(d) else "-"})
    five = df[df["agree"] == 5]
    out.append({"Lane": "of which 5 of 5", "Share": f"{len(five) / len(df):.0%}",
                "Top-1 correct": f"{five['ok1'].mean():.1%}" if len(five) else "-", "Top-2 correct": "-"})
    return pd.DataFrame(out)


def row(true, labels):
    c = Counter(labels).most_common()
    return {"agree": c[0][1], "ok1": c[0][0] == true, "ok2": true in [lab for lab, _ in c[:2]]}


def main():
    s = Searcher()
    test = pd.read_parquet("data/test_tickets.parquet").sample(N_TEST, random_state=SEED).reset_index(drop=True)
    test["text"] = ticket_text(test)
    qvecs = embed(s.model, test["text"].tolist(), "query")
    cur, ded, dup_slots = [], [], 0
    for i, t in test.iterrows():
        cands = s.hybrid_top(t["text"], POOL, qvec=qvecs[i])
        a, b = cands[:K], diverse(s, cands)
        dup_slots += len(set(a) - set(b))
        cur.append(row(t["queue"], s.kb.iloc[a]["queue"].tolist()))
        ded.append(row(t["queue"], s.kb.iloc[b]["queue"].tolist()))
    report = (
        f"# Day 5 (E) - Near-duplicate neighbours and routing confidence ({N_TEST:,} test tickets)\n\n"
        f"- Neighbour slots taken by a near-copy (cosine >= {SIM}) of another neighbour: "
        f"**{dup_slots} of {N_TEST * K} ({dup_slots / (N_TEST * K):.1%})**.\n\n"
        "## Current neighbours (as deployed)\n\n" + lanes(pd.DataFrame(cur)).to_markdown(index=False) + "\n\n"
        "## De-duplicated neighbours\n\n" + lanes(pd.DataFrame(ded)).to_markdown(index=False) + "\n\n"
        "Switch to de-duplicated neighbours only if Auto-lane accuracy holds or improves.\n"
    )
    Path("reports/day5_dedupe_check.md").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
