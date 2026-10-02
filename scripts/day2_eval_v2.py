"""
Day 2 (v2) - Sharper search evaluation.
Run from the project root:  python scripts/day2_eval_v2.py
Output: reports/day2_retrieval_v2.md
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from day2_search import INDEX_DIR, TOP_K, Searcher, embed, tag_set, ticket_text

N_TEST = 1000
SEED = 0
POOL = 50
SHORT_WORDS = 15
N_BOOT = 2000
METHODS = ["Random", "Keyword (BM25)", "Meaning (embeddings)", "Hybrid"]


def rrf(ranked_lists, k=TOP_K, rrf_k=60):
    fused = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            fused[idx] = fused.get(idx, 0) + 1 / (rrf_k + rank + 1)
    return [i for i, _ in sorted(fused.items(), key=lambda x: -x[1])[:k]]


def score(tops, test_tags, kb_tags, test_queue, kb_queue, kb_ans, a_emb):
    topic_p, topic_rr, dept_p, ans = [], [], [], []
    for i, top in enumerate(tops):
        hits = [len(test_tags[i] & kb_tags[j]) >= 2 for j in top]
        topic_p.append(np.mean(hits))
        first = next((r for r, h in enumerate(hits, 1) if h), None)
        topic_rr.append(1 / first if first else 0.0)
        dept_p.append(np.mean([kb_queue[j] == test_queue[i] for j in top]))
        ans.append(float(np.mean(kb_ans[top] @ a_emb[i])))
    return {"topic_p": np.array(topic_p), "topic_rr": np.array(topic_rr),
            "dept_p": np.array(dept_p), "ans": np.array(ans)}


def paired_ci(a, b, rng):
    d = a - b
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    boots = d[idx].mean(axis=1)
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def verdict(mean, lo, hi):
    if lo > 0:
        return "real improvement"
    if hi < 0:
        return "real drop"
    return "could be luck"


def main():
    s = Searcher()
    rng = np.random.default_rng(SEED)
    kb_ans = np.load(INDEX_DIR / "kb_answer_emb.npy")
    kb_tags = [tag_set(r) for _, r in s.kb.iterrows()]
    kb_queue = s.kb["queue"].to_numpy()

    test = pd.read_parquet("data/test_tickets.parquet").sample(N_TEST, random_state=SEED).reset_index(drop=True)
    test_tags = [tag_set(r) for _, r in test.iterrows()]
    has_tags = np.array([len(t) >= 2 for t in test_tags])
    test_queue = test["queue"].to_numpy()
    print("Embedding the real answers of the test tickets...")
    a_emb = embed(s.model, test["answer"].astype(str).tolist(), "passage")

    variants = {
        "Full ticket (subject + body)": ticket_text(test).tolist(),
        f"Short message (first {SHORT_WORDS} words)": [" ".join(str(b).split()[:SHORT_WORDS]) for b in test["body"]],
    }

    sections = []
    for vname, queries in variants.items():
        print(f"\n== {vname} ==")
        q_emb = embed(s.model, queries, "query")
        tops = {m: [] for m in METHODS}
        ms = {m: [] for m in METHODS}
        for i, q in enumerate(queries):
            if i % 200 == 0:
                print(f"  query {i}/{len(queries)}")
            t0 = time.perf_counter(); b = s.bm25_top(q, POOL); tb = time.perf_counter() - t0
            t0 = time.perf_counter(); d = s.dense_top(q, POOL, q_emb[i]); td = time.perf_counter() - t0
            t0 = time.perf_counter(); h = rrf([b, d]); th = time.perf_counter() - t0
            tops["Random"].append(list(rng.choice(len(s.kb), TOP_K, replace=False)))
            tops["Keyword (BM25)"].append(b[:TOP_K]);        ms["Keyword (BM25)"].append(tb * 1000)
            tops["Meaning (embeddings)"].append(d[:TOP_K]);  ms["Meaning (embeddings)"].append(td * 1000)
            tops["Hybrid"].append(h);                        ms["Hybrid"].append((tb + td + th) * 1000)
            ms["Random"].append(0.0)

        res = {m: score(tops[m], test_tags, kb_tags, test_queue, kb_queue, kb_ans, a_emb) for m in METHODS}
        rows = []
        for m in METHODS:
            r = res[m]
            rows.append({
                "Method": m,
                "Topic precision@5": f"{r['topic_p'][has_tags].mean():.1%}",
                "Topic MRR": f"{r['topic_rr'][has_tags].mean():.3f}",
                "Same dept precision@5": f"{r['dept_p'].mean():.1%}",
                "Answer similarity (avg of 5)": f"{r['ans'].mean():.3f}",
                "Median ms": f"{np.median(ms[m]):.0f}",
            })
        table = pd.DataFrame(rows).to_markdown(index=False)

        comparisons = [("Meaning (embeddings)", "Keyword (BM25)"), ("Hybrid", "Meaning (embeddings)"),
                       ("Meaning (embeddings)", "Random")]
        lines = []
        for a, b in comparisons:
            mean, lo, hi = paired_ci(res[a]["topic_p"][has_tags], res[b]["topic_p"][has_tags], rng)
            lines.append(f"- {a} vs {b}, topic precision: {mean:+.1%} "
                         f"(95% range {lo:+.1%} to {hi:+.1%}) -> **{verdict(mean, lo, hi)}**")
            mean, lo, hi = paired_ci(res[a]["ans"], res[b]["ans"], rng)
            lines.append(f"- {a} vs {b}, answer similarity: {mean:+.3f} "
                         f"(95% range {lo:+.3f} to {hi:+.3f}) -> **{verdict(mean, lo, hi)}**")
        sections.append(f"## {vname}\n\n{table}\n\n### Is the difference real?\n\n" + "\n".join(lines))

    report = (
        f"# Day 2 (v2) - Search quality on {N_TEST:,} unseen test tickets\n\n"
        + "\n\n".join(sections) + "\n\n"
        "## How to read this\n\n"
        "- **Topic precision@5**: share of the top 5 that share 2+ specific tags with the test ticket "
        f"(only the {has_tags.sum():,} test tickets with 2+ specific tags).\n"
        "- **Topic MRR**: 1.0 if the first result matches, 0.5 if the second, ... 0 if none of the 5 match.\n"
        "- **Same dept precision@5**: share of the top 5 in the same department (labels imperfect, see D4).\n"
        "- **Answer similarity (avg of 5)**: average closeness of the 5 retrieved answers to the real answer. "
        "Compare against the Random row, not against 1.0.\n"
        "- **95% range**: if the range includes 0, the difference could be luck.\n"
        f"- Knowledge base: {len(s.kb):,} tickets. Keyword latency uses the pure-Python rank_bm25 library.\n"
    )
    Path("reports/day2_retrieval_v2.md").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
