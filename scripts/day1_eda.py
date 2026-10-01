"""
Day 1 - Data exploration for the Support Ticket Resolution Assistant.

Run:
    pip install pandas scikit-learn datasets pyarrow tabulate
    python day1_eda.py                 # downloads from Hugging Face
    python day1_eda.py --csv path.csv  # or use a local CSV

Outputs:
    reports/day1_eda.md       -> numbers to paste into your README / design doc
    data/kb_tickets.parquet   -> knowledge base (what the system can retrieve from)
    data/test_tickets.parquet -> held-out tickets (what you evaluate on)
"""
import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors

SEED = 42
NEAR_DUP_THRESHOLD = 0.90  # cosine similarity above this = same "group"

# Answers that only ask the customer for more info, rather than resolving anything.
INFO_REQUEST_PATTERNS = [
    r"could you (please )?(provide|share|specify|confirm|send|describe|tell)",
    r"please (provide|share|specify|confirm|send|let us know|include|inform)",
    r"kindly (provide|share|specify|confirm|send)",
    r"can you (please )?(provide|share|specify|confirm)",
    r"könnten sie (uns )?(bitte )?",
    r"bitte (teilen|senden|geben|bestätigen|nennen)",
    r"teilen sie uns",
]
# Answers that contain actual action steps.
RESOLUTION_PATTERNS = [
    r"\b(try|restart|reset|update|reinstall|clear|change|check|disable|enable|replace|reseat)\b",
    r"\b(versuchen sie|starten sie|aktualisieren|setzen sie|prüfen sie|ändern sie)\b",
    r"\b(step \d|first,|then,|finally,)",
]
TELECOM_TAGS = {"network", "connectivity", "vpn", "outage", "disruption", "router",
                "wifi", "bluetooth", "billing", "payment", "service", "internet"}


def load(csv_path):
    if csv_path:
        return pd.read_csv(csv_path)
    from datasets import load_dataset
    return load_dataset("Tobi-Bueck/customer-support-tickets", split="train").to_pandas()


def norm(s):
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def any_match(text, patterns):
    t = str(text).lower()
    return any(re.search(p, t) for p in patterns)


def near_dup_groups(texts, threshold):
    """Union-find over TF-IDF neighbours, so near-identical tickets share a group id."""
    X = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True).fit_transform(texts)
    nn = NearestNeighbors(n_neighbors=min(6, len(texts)), metric="cosine").fit(X)
    dist, idx = nn.kneighbors(X)
    parent = list(range(len(texts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(texts)):
        for d, j in zip(dist[i][1:], idx[i][1:]):
            if 1 - d >= threshold:
                parent[find(i)] = find(j)
    return np.array([find(i) for i in range(len(texts))])


def table(series, top=None):
    vc = series.value_counts(dropna=False)
    if top:
        vc = vc.head(top)
    pct = (vc / len(series) * 100).round(1)
    rows = "\n".join(f"| {k} | {v} | {p}% |" for k, v, p in zip(vc.index, vc.values, pct.values))
    return "| value | count | share |\n|---|---|---|\n" + rows


def main(csv_path):
    Path("reports").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    df = load(csv_path)
    out = ["# Day 1 - Dataset exploration\n"]

    # 1. Shape and missing values
    out.append(f"**Rows:** {len(df):,}  **Columns:** {len(df.columns)}\n")
    out.append("## Missing values\n")
    miss = df.isna().mean().mul(100).round(1)
    out.append("\n".join(f"- {c}: {m}%" for c, m in miss.items() if m > 0) or "- none")

    df = df.dropna(subset=["body", "answer"]).copy()
    df["subject"] = df["subject"].fillna("")
    df["text"] = (df["subject"] + "\n" + df["body"]).map(str)
    out.append(f"\nRows kept after dropping empty body/answer: **{len(df):,}**\n")

    # 2. Label distributions
    for col in ["type", "queue", "priority", "language"]:
        if col in df:
            out.append(f"\n## {col}\n" + table(df[col], top=15))
    if "version" in df:
        out.append("\n## version (possible proxy for data batches)\n" + table(df["version"], top=10))

    tag_cols = [c for c in df.columns if c.startswith("tag_")]
    if tag_cols:
        tags = df[tag_cols].stack().astype(str).str.strip()
        out.append("\n## Top 25 tags\n" + table(tags, top=25))
        lower_tags = df[tag_cols].apply(lambda r: {str(t).lower() for t in r if pd.notna(t)}, axis=1)
        telecom_share = lower_tags.map(lambda s: bool(s & TELECOM_TAGS)).mean() * 100
        out.append(f"\nTickets with at least one telecom-relevant tag: **{telecom_share:.1f}%**\n")

    # 3. Text lengths
    df["body_words"] = df["body"].str.split().str.len()
    df["answer_words"] = df["answer"].str.split().str.len()
    out.append("\n## Text length (words)\n")
    out.append(df[["body_words", "answer_words"]].describe().round(0).to_markdown())

    # 4. Duplicates
    df["exact_key"] = df["text"].map(lambda s: hashlib.md5(norm(s).encode()).hexdigest())
    exact = df["exact_key"].duplicated().sum()
    df["group"] = near_dup_groups(df["text"].tolist(), NEAR_DUP_THRESHOLD)
    n_groups = df["group"].nunique()
    out.append("\n## Duplicates\n")
    out.append(f"- Exact duplicates: **{exact:,}**")
    out.append(f"- Near-duplicate groups (cosine >= {NEAR_DUP_THRESHOLD}): **{n_groups:,}** groups "
               f"for {len(df):,} tickets ({(1 - n_groups / len(df)) * 100:.1f}% of tickets are near-copies)")

    # 5. Are the answers real resolutions?
    df["asks_info"] = df["answer"].map(lambda a: any_match(a, INFO_REQUEST_PATTERNS))
    df["has_steps"] = df["answer"].map(lambda a: any_match(a, RESOLUTION_PATTERNS))
    df["answer_kind"] = np.select(
        [df["has_steps"], df["asks_info"]], ["resolution", "info_request_only"], "other")
    out.append("\n## What do agent answers contain? (heuristic)\n" + table(df["answer_kind"]))
    out.append("\nSpot-check 30 of each kind by hand and report the heuristic's precision.\n")

    # 6. Baseline queue classifier (TF-IDF + logistic regression), group-aware split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, te = next(gss.split(df, groups=df["group"]))
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)
    Xtr, Xte = vec.fit_transform(df["text"].iloc[tr]), vec.transform(df["text"].iloc[te])
    out.append("\n## Baseline classifiers (TF-IDF + LogReg, group-aware 80/20 split)\n")
    for target in ["queue", "type", "priority"]:
        if target not in df:
            continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(Xtr, df[target].iloc[tr])
        pred = clf.predict(Xte)
        f1 = f1_score(df[target].iloc[te], pred, average="macro")
        out.append(f"- **{target}** macro-F1: **{f1:.3f}**")
        if target == "queue":
            rep = classification_report(df[target].iloc[te], pred, zero_division=0)
            Path("reports/queue_baseline_report.txt").write_text(rep)
    out.append("\nThese are the numbers your LLM/embedding approach must beat.\n")

    # 7. Save the split
    kb, test = df.iloc[tr], df.iloc[te]
    keep = [c for c in ["subject", "body", "answer", "type", "queue", "priority", "language",
                        "version", "answer_kind", "group"] + tag_cols if c in df]
    kb[keep].to_parquet("data/kb_tickets.parquet", index=False)
    test[keep].to_parquet("data/test_tickets.parquet", index=False)
    out.append(f"\n## Split\n- Knowledge base: {len(kb):,} tickets\n- Test: {len(test):,} tickets\n"
               "- Near-duplicate groups never cross the split (prevents leakage).\n")

    Path("reports/day1_eda.md").write_text("\n".join(out))
    print("\n".join(out))
    print("\nWrote reports/day1_eda.md, data/kb_tickets.parquet, data/test_tickets.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    main(ap.parse_args().csv)
