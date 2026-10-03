"""
Day 5 (A) - Smart escalation: route automatically only when the 5 similar tickets agree.
Run: python scripts/day5_escalation.py
Output: reports/day5_escalation.md
"""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from day2_search import Searcher, ticket_text  # noqa: E402

N_TEST = 1000
SEED = 0
K = 5


def votes(labels):
    c = Counter(labels).most_common()
    return c[0][0], c[0][1], [lab for lab, _ in c[:2]]


def analyse(df, field):
    df = df.copy()
    df["ok1"] = df["top1"] == df["true"]
    df["ok2"] = [t in p for t, p in zip(df["true"], df["top2"])]

    by_level = []
    for a in sorted(df["agree"].unique(), reverse=True):
        d = df[df["agree"] == a]
        by_level.append({"Agreement (of 5)": f"{a} of 5", "Tickets": len(d), "Share": f"{len(d) / len(df):.0%}",
                         "Top-1 correct": f"{d['ok1'].mean():.0%}", "Top-2 correct": f"{d['ok2'].mean():.0%}"})

    policies = []
    for k in range(K, 0, -1):
        auto, rest = df[df["agree"] >= k], df[df["agree"] < k]
        if len(auto) == 0:
            continue
        policies.append({
            "Rule: auto-route if": f"{k}+ of 5 agree",
            "Automated (coverage)": f"{len(auto) / len(df):.0%}",
            "Accuracy on automated": f"{auto['ok1'].mean():.0%}",
            "Sent to humans": f"{len(rest) / len(df):.0%}",
            "Accuracy if humans' share were auto-routed": f"{rest['ok1'].mean():.0%}" if len(rest) else "-",
        })
    return (f"### {field}: accuracy by agreement level\n\n" + pd.DataFrame(by_level).to_markdown(index=False)
            + f"\n\n### {field}: automation rules\n\n" + pd.DataFrame(policies).to_markdown(index=False))


def main():
    s = Searcher()
    test = pd.read_parquet("data/test_tickets.parquet").sample(N_TEST, random_state=SEED).reset_index(drop=True)
    test["text"] = ticket_text(test)
    rows = {"Department": [], "Priority": []}
    for i, t in test.iterrows():
        if i % 100 == 0:
            print(f"Ticket {i}/{N_TEST}")
        nb = s.kb.iloc[s.hybrid_top(t["text"], K)]
        for field, col in [("Department", "queue"), ("Priority", "priority")]:
            top1, agree, top2 = votes(nb[col].tolist())
            rows[field].append({"true": t[col], "top1": top1, "agree": agree, "top2": top2})

    sections = [analyse(pd.DataFrame(rows[f]), f) for f in rows]
    report = (
        f"# Day 5 - Smart escalation on {N_TEST:,} unseen test tickets\n\n"
        "The 5 most similar past tickets vote on the label. Strong agreement = confident = route automatically; "
        "weak agreement = send to a human.\n\n" + "\n\n".join(sections) + "\n\n"
        "- **Coverage**: share of tickets the rule routes automatically.\n"
        "- **Accuracy on automated**: how often the automatic label matches the dataset label.\n"
        "- Dataset labels are imperfect (Day 1 hand check: many department labels don't fit the text), "
        "so true accuracy is likely higher than shown.\n"
    )
    Path("reports/day5_escalation.md").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
