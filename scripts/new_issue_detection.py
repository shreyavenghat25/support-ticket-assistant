"""
New-issue detection: can the system notice a kind of ticket it has never seen?

Simulation: remove one whole department from the library, send its tickets in mixed with
normal traffic, flag tickets that look unfamiliar, group the flagged ones, and check whether
the biggest group is the new issue. Then add a few resolved examples and check the flags stop.

Run: python scripts/new_issue_detection.py      (no LLM calls; about 2-4 minutes)
Output: reports/new_issue_detection.md and data/index/novelty.json (threshold used by the app)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from day2_search import INDEX_DIR, embed, load_embedder, ticket_text  # noqa: E402

SCENARIOS = ["Billing and Payments", "Service Outages and Maintenance", "Returns and Exchanges"]
N_NEW, N_NORMAL, K, SEED = 150, 1500, 5, 0
FALSE_ALARM = 0.05
N_LEARN = 75
GERMAN_STOP = ("der die das und ist ich sie wir ihr ein eine einen einer zu mit für auf den dem des von "
               "nicht bitte uns unser unsere es im in sind haben hat wurde werden kann können sehr wie oder "
               "auch bei als an aus noch nach um so zur zum über dass diese dieser ihre ihnen mein meine "
               "wir unseren hallo vielen dank sehr geehrte geehrter").split()
STOP = list(ENGLISH_STOP_WORDS) + GERMAN_STOP + ["dear", "support", "team", "customer", "thank", "thanks",
                                                 "regards", "issue", "issues", "help", "assistance", "tel_num",
                                                 "acc_num", "name", "nan", "kundensupport", "kundenservice"]


def novelty(queries, library):
    sims = queries @ library.T
    top = np.sort(sims, axis=1)[:, -K:]
    return 1 - top.mean(axis=1)


def cluster_report(texts, emb, is_new, rng_seed):
    if len(texts) < 10:
        return []
    k = int(min(5, max(2, len(texts) // 25)))
    labels = KMeans(n_clusters=k, n_init=10, random_state=rng_seed).fit_predict(emb)
    tfidf = TfidfVectorizer(max_features=5000, stop_words=STOP, token_pattern=r"(?u)\b[^\W\d_]{3,}\b")
    X = tfidf.fit_transform(texts)
    vocab = np.array(tfidf.get_feature_names_out())
    out = []
    for c in range(k):
        m = labels == c
        terms = vocab[np.asarray(X[m].mean(axis=0)).ravel().argsort()[::-1][:6]]
        out.append({"size": int(m.sum()), "new_share": float(is_new[m].mean()), "keywords": ", ".join(terms)})
    return sorted(out, key=lambda r: -r["size"])


def main():
    rng = np.random.default_rng(SEED)
    kb = pd.read_parquet(INDEX_DIR / "kb.parquet")
    kb_emb = np.load(INDEX_DIR / "kb_text_emb.npy")
    test = pd.read_parquet("data/test_tickets.parquet")
    scenarios = [s for s in SCENARIOS if (test["queue"] == s).sum() >= 20 and (kb["queue"] == s).any()]

    parts = [test[test["queue"] == s].sample(min(N_NEW, (test["queue"] == s).sum()), random_state=SEED)
             for s in scenarios]
    others = test[~test["queue"].isin(scenarios)]
    parts.append(others.sample(min(N_NORMAL, len(others)), random_state=SEED))
    pool = pd.concat(parts).drop_duplicates().reset_index(drop=True)
    pool["text"] = ticket_text(pool)
    print(f"Embedding {len(pool):,} incoming tickets...")
    model = load_embedder()
    P = embed(model, pool["text"].tolist(), "query")

    rows, clusters_md = [], []
    for s in scenarios:
        print(f"\nScenario: '{s}' is a brand-new issue")
        lib = kb_emb[(kb["queue"] != s).to_numpy()]
        new_idx = np.where(pool["queue"] == s)[0]
        norm_idx = rng.permutation(np.where(pool["queue"] != s)[0])
        calib, ev = norm_idx[: len(norm_idx) // 2], norm_idx[len(norm_idx) // 2:]

        nov_new, nov_cal, nov_ev = novelty(P[new_idx], lib), novelty(P[calib], lib), novelty(P[ev], lib)
        thr = np.quantile(nov_cal, 1 - FALSE_ALARM)
        recall = float((nov_new > thr).mean())
        false_alarm = float((nov_ev > thr).mean())
        auc = roc_auc_score(np.r_[np.ones(len(nov_new)), np.zeros(len(nov_ev))], np.r_[nov_new, nov_ev])

        learn = rng.permutation(new_idx)
        added, later = learn[:N_LEARN], learn[N_LEARN:]
        recall_after = float((novelty(P[later], np.vstack([lib, P[added]])) > thr).mean()) if len(later) else float("nan")

        stream = np.r_[new_idx, ev]
        flagged = stream[np.r_[nov_new, nov_ev] > thr]
        cl = cluster_report(pool["text"].iloc[flagged].tolist(), P[flagged],
                            (pool["queue"].iloc[flagged] == s).to_numpy(), SEED)
        top = cl[0] if cl else {"size": 0, "new_share": float("nan"), "keywords": "-"}
        rows.append({"New issue (held-out department)": s,
                     "New tickets flagged": f"{recall:.0%}",
                     "False alarms (normal)": f"{false_alarm:.0%}",
                     "AUC": f"{auc:.2f}",
                     f"Flagged after adding {N_LEARN} resolved": f"{recall_after:.0%}",
                     "Biggest flagged group: % new issue": f"{top['new_share']:.0%}" if cl else "-"})
        clusters_md.append(f"### {s}\n\n" + (pd.DataFrame(cl).rename(columns={
            "size": "Flagged tickets", "new_share": "Share from new issue", "keywords": "Top keywords"})
            .assign(**{"Share from new issue": lambda d: d["Share from new issue"].map(lambda v: f"{v:.0%}")})
            .to_markdown(index=False) if cl else "Too few flagged tickets to group."))
        print(f"  flagged {recall:.0%} of new, {false_alarm:.0%} false alarms, AUC {auc:.2f}, after learning {recall_after:.0%}")

    full_nov = novelty(P[np.where(~pool["queue"].isin(scenarios))[0]], kb_emb)
    prod = {"k": K, "false_alarm_budget": FALSE_ALARM, "threshold": float(np.quantile(full_nov, 1 - FALSE_ALARM)),
            "note": "novelty = 1 - mean cosine similarity of the k closest library tickets"}
    (INDEX_DIR / "novelty.json").write_text(json.dumps(prod, indent=2))

    report = (
        "# New-issue detection (simulation)\n\n"
        "Each row removes one whole department from the library, then streams its tickets in among normal "
        "traffic. A ticket is flagged when its 5 closest library matches are unusually dissimilar "
        f"(threshold set so about {FALSE_ALARM:.0%} of normal tickets are flagged).\n\n"
        + pd.DataFrame(rows).to_markdown(index=False) + "\n\n"
        f"- **Flagged after adding {N_LEARN} resolved**: once a few resolved tickets of the new kind join the "
        "library, new arrivals of that kind should stop being flagged (evolving data).\n"
        "- **AUC**: 0.5 = no better than chance, 1.0 = perfect separation of new vs normal tickets.\n"
        "- **Biggest flagged group**: flagged tickets are grouped by meaning; this is the share of the "
        "largest group that belongs to the new issue.\n\n"
        "## Groups of flagged tickets\n\n" + "\n\n".join(clusters_md) + "\n\n"
        f"Production threshold (full library, {FALSE_ALARM:.0%} false-alarm budget): "
        f"{prod['threshold']:.4f}, saved to `data/index/novelty.json`.\n\n"
        "Caveats: department labels are imperfect (Day 1), and departments overlap in vocabulary, so some "
        "'new' tickets genuinely resemble tickets from other departments.\n"
    )
    Path("reports/new_issue_detection.md").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
