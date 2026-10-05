"""
Day 6 (B) - Length-aware new-issue threshold.
The new-issue threshold (D12) was set on full tickets (subject + body, median ~53 words). Short one-line
messages, the kind people type into a demo, naturally match the library less closely, so they cross that
threshold far more often. This measures the false-alarm rate on short versions of normal test tickets and sets
a separate threshold for short tickets with the same 5% false-alarm budget.
Run: python scripts/calibrate_novelty_short.py
Updates: data/index/novelty.json (adds threshold_short and short_max_words; the full-ticket threshold is kept)
Output: reports/novelty_short.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from day2_search import INDEX_DIR, embed, load_embedder  # noqa: E402

N, SEED, K = 1500, 0, 5
SHORT_WORDS = 20          # length of the simulated one-line messages
SHORT_MAX_WORDS = 30      # tickets up to this many words use the short threshold
FALSE_ALARM = 0.05


def novelty(queries, library):
    sims = queries @ library.T
    return 1 - np.sort(sims, axis=1)[:, -K:].mean(axis=1)


def main():
    cfg_path = INDEX_DIR / "novelty.json"
    cfg = json.loads(cfg_path.read_text())
    kb_emb = np.load(INDEX_DIR / "kb_text_emb.npy")
    test = pd.read_parquet("data/test_tickets.parquet").sample(N, random_state=SEED)
    full = (test["subject"].fillna("") + "\n" + test["body"].fillna("")).str.strip().tolist()
    short = [" ".join(str(b).split()[:SHORT_WORDS]) for b in test["body"].fillna("")]

    model = load_embedder()
    nov_full = novelty(embed(model, full, "query"), kb_emb)
    nov_short = novelty(embed(model, short, "query"), kb_emb)

    old = cfg["threshold"]
    new_short = float(np.quantile(nov_short, 1 - FALSE_ALARM))
    rows = [
        {"Normal test tickets": "Full tickets", "Flagged with current threshold": f"{(nov_full > old).mean():.1%}",
         "Flagged with length-aware thresholds": f"{(nov_full > old).mean():.1%}"},
        {"Normal test tickets": f"Short ({SHORT_WORDS}-word) versions",
         "Flagged with current threshold": f"{(nov_short > old).mean():.1%}",
         "Flagged with length-aware thresholds": f"{(nov_short > new_short).mean():.1%}"},
    ]
    cfg.update({"threshold_short": new_short, "short_max_words": SHORT_MAX_WORDS,
                "short_note": f"used for tickets of {SHORT_MAX_WORDS} words or fewer; calibrated on "
                              f"{SHORT_WORDS}-word versions of {N} normal test tickets at a "
                              f"{FALSE_ALARM:.0%} false-alarm budget"})
    cfg_path.write_text(json.dumps(cfg, indent=2))
    report = (
        "# Day 6 (B) - Length-aware new-issue threshold\n\n"
        f"Every ticket here is a normal test ticket (its topic exists in the library), so every flag is a false alarm.\n\n"
        + pd.DataFrame(rows).to_markdown(index=False) + "\n\n"
        f"- Full-ticket threshold (unchanged): {old:.4f}\n"
        f"- New short-ticket threshold (tickets of {SHORT_MAX_WORDS} words or fewer): {new_short:.4f}\n\n"
        "The D12 simulation results were measured on full tickets and still apply to them.\n"
    )
    Path("reports/novelty_short.md").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
