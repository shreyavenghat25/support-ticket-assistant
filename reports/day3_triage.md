# Day 3 - Ticket triage on 200 unseen test tickets

| Method                       |   Dept macro-F1 | Dept top-2 accuracy   |   Type macro-F1 |   Priority macro-F1 |   Failed outputs | Avg tokens (in/out)   | Median ms   |
|:-----------------------------|----------------:|:----------------------|----------------:|--------------------:|-----------------:|:----------------------|:------------|
| Baseline (TF-IDF + LogReg)   |           0.379 | 59.0%                 |           0.848 |               0.557 |                0 | -                     | -           |
| kNN vote (5 similar tickets) |           0.474 | 77.5%                 |           0.821 |               0.637 |                0 | -                     | -           |
| LLM (ticket only)            |           0.18  | 41.0%                 |           0.577 |               0.362 |                1 | 324/94                | 1034        |
| LLM + 5 similar tickets      |           0.381 | 63.5%                 |           0.781 |               0.605 |                2 | 851/98                | 1075        |

- Model: `gemini-3.5-flash-lite`, temperature 0, JSON output.
- **Dept top-2 accuracy**: the dataset's department is one of the two suggested (see decision D4).
- Dataset department labels are often questionable (Day 1 hand check), so exact-match scores understate quality.
- Sentiment has no ground truth in the dataset; distribution from LLM + 5 similar tickets: {'calm': 83, 'concerned': 108, 'frustrated': 7}.
- Small sample (200 tickets): differences of a few points may be noise.
