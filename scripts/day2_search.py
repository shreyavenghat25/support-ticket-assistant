"""
Day 2 - Find similar past tickets: keyword (BM25) vs meaning (embeddings) vs hybrid.

Usage (run from the project root, with .venv active):
    python scripts/day2_search.py build
    python scripts/day2_search.py search "my wifi drops every evening"
    python scripts/day2_search.py eval
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_NAME = "intfloat/multilingual-e5-small"
INDEX_DIR = Path("data/index")
TOP_K = 5
EVAL_SAMPLE = 1000
SEED = 0
GENERIC_TAGS = {"tech support", "it", "feedback", "support", "customer support", "technical"}


def ticket_text(df):
    return (df["subject"].fillna("") + "\n" + df["body"].fillna("")).str.strip()


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def tag_set(row):
    tags = {str(row[c]).strip().lower() for c in row.index if c.startswith("tag_") and pd.notna(row[c])}
    return tags - GENERIC_TAGS


def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def embed(model, texts, kind):
    prefixed = [f"{kind}: {t}" for t in texts]
    return model.encode(prefixed, batch_size=64, normalize_embeddings=True,
                        show_progress_bar=len(texts) > 100, convert_to_numpy=True).astype(np.float32)


def build():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    kb = pd.read_parquet("data/kb_tickets.parquet")
    kb["text"] = ticket_text(kb)
    before = len(kb)
    kb["key"] = kb["text"].str.lower().str.split().str.join(" ")
    kb = kb.drop_duplicates("key").drop(columns="key").reset_index(drop=True)
    print(f"Knowledge base: {before:,} tickets -> {len(kb):,} after removing exact copies")
    model = load_embedder()
    print("Embedding tickets (meaning fingerprints)...")
    np.save(INDEX_DIR / "kb_text_emb.npy", embed(model, kb["text"].tolist(), "passage"))
    print("Embedding agent answers (used to score search quality)...")
    np.save(INDEX_DIR / "kb_answer_emb.npy", embed(model, kb["answer"].astype(str).tolist(), "passage"))
    kb.to_parquet(INDEX_DIR / "kb.parquet", index=False)
    json.dump({"model": MODEL_NAME, "tickets": len(kb)}, open(INDEX_DIR / "meta.json", "w"))
    print(f"Index saved to {INDEX_DIR}/")


class Searcher:
    def __init__(self, model=None):
        from rank_bm25 import BM25Okapi
        self.kb = pd.read_parquet(INDEX_DIR / "kb.parquet")
        self.emb = np.load(INDEX_DIR / "kb_text_emb.npy")
        self.model = model or load_embedder()
        self.bm25 = BM25Okapi([tokenize(t) for t in self.kb["text"]])

    def bm25_top(self, query, k):
        scores = self.bm25.get_scores(tokenize(query))
        return list(np.argsort(-scores)[:k])

    def dense_top(self, query, k, qvec=None):
        if qvec is None:
            qvec = embed(self.model, [query], "query")[0]
        scores = self.emb @ qvec
        return list(np.argsort(-scores)[:k])

    def hybrid_top(self, query, k, qvec=None, pool=50, rrf_k=60):
        fused = {}
        for ranked in (self.bm25_top(query, pool), self.dense_top(query, pool, qvec)):
            for rank, idx in enumerate(ranked):
                fused[idx] = fused.get(idx, 0) + 1 / (rrf_k + rank + 1)
        return [i for i, _ in sorted(fused.items(), key=lambda x: -x[1])[:k]]


def search(query):
    s = Searcher()
    for name, fn in [("Keyword (BM25)", s.bm25_top), ("Meaning (embeddings)", s.dense_top),
                     ("Hybrid", s.hybrid_top)]:
        print(f"\n===== {name} =====")
        for rank, idx in enumerate(fn(query, TOP_K), 1):
            r = s.kb.iloc[idx]
            print(f"{rank}. [{r['language']}] [{r['queue']}] {str(r['subject'])[:80]}")


def evaluate():
    s = Searcher()
    kb_ans = np.load(INDEX_DIR / "kb_answer_emb.npy")
    kb_tags = [tag_set(r) for _, r in s.kb.iterrows()]
    test = pd.read_parquet("data/test_tickets.parquet").sample(EVAL_SAMPLE, random_state=SEED)
    test["text"] = ticket_text(test)
    print("Embedding test tickets and their real answers...")
    q_emb = embed(s.model, test["text"].tolist(), "query")
    a_emb = embed(s.model, test["answer"].astype(str).tolist(), "passage")
    methods = {"Keyword (BM25)": lambda q, v: s.bm25_top(q, TOP_K),
               "Meaning (embeddings)": lambda q, v: s.dense_top(q, TOP_K, v),
               "Hybrid": lambda q, v: s.hybrid_top(q, TOP_K, v)}
    rows = []
    for name, fn in methods.items():
        tag_hit, queue_hit, ans_sim, lang_cross, times = [], [], [], [], []
        print(f"Scoring {name}...")
        for i, (_, t) in enumerate(test.iterrows()):
            start = time.perf_counter()
            top = fn(t["text"], q_emb[i])
            times.append((time.perf_counter() - start) * 1000)
            my_tags = tag_set(t)
            tag_hit.append(any(len(my_tags & kb_tags[j]) >= 2 for j in top))
            queue_hit.append(any(s.kb.iloc[j]["queue"] == t["queue"] for j in top))
            ans_sim.append(float(np.max(kb_ans[top] @ a_emb[i])))
            lang_cross.append(np.mean([s.kb.iloc[j]["language"] != t["language"] for j in top]))
        rows.append({
            "Method": name,
            "Topic match@5": f"{np.mean(tag_hit):.1%}",
            "Same dept@5": f"{np.mean(queue_hit):.1%}",
            "Best answer similarity": f"{np.mean(ans_sim):.3f}",
            "Other-language results": f"{np.mean(lang_cross):.1%}",
            "Median ms/query": f"{np.median(times):.0f}",
        })
    table = pd.DataFrame(rows).to_markdown(index=False)
    report = (
        "# Day 2 - Search quality on 1,000 unseen test tickets\n\n" + table + "\n\n"
        "- **Topic match@5**: at least one of the top 5 shares 2+ specific tags with the test ticket.\n"
        "- **Same dept@5**: at least one of the top 5 is in the same department (labels are imperfect, see D4).\n"
        "- **Best answer similarity**: how close the best retrieved ticket's answer is to the real answer "
        "(cosine, higher is better).\n"
        "- **Other-language results**: share of results in the other language (cross-lingual matching).\n"
        f"- Model: `{MODEL_NAME}`. Knowledge base: {len(s.kb):,} tickets after removing exact copies.\n"
    )
    Path("reports/day2_retrieval.md").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build":
        build()
    elif cmd == "search" and len(sys.argv) > 2:
        search(" ".join(sys.argv[2:]))
    elif cmd == "eval":
        evaluate()
    else:
        print(__doc__)
