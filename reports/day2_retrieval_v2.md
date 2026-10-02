# Day 2 (v2) - Search quality on 1,000 unseen test tickets

## Full ticket (subject + body)

| Method               | Topic precision@5   |   Topic MRR | Same dept precision@5   |   Answer similarity (avg of 5) |   Median ms |
|:---------------------|:--------------------|------------:|:------------------------|-------------------------------:|------------:|
| Random               | 12.2%               |       0.225 | 17.7%                   |                          0.844 |           0 |
| Keyword (BM25)       | 63.6%               |       0.821 | 43.4%                   |                          0.919 |         215 |
| Meaning (embeddings) | 59.5%               |       0.794 | 44.2%                   |                          0.923 |           3 |
| Hybrid               | 64.1%               |       0.831 | 44.1%                   |                          0.923 |         217 |

### Is the difference real?

- Meaning (embeddings) vs Keyword (BM25), topic precision: -4.2% (95% range -6.0% to -2.3%) -> **real drop**
- Meaning (embeddings) vs Keyword (BM25), answer similarity: +0.004 (95% range +0.003 to +0.005) -> **real improvement**
- Hybrid vs Meaning (embeddings), topic precision: +4.6% (95% range +3.1% to +6.1%) -> **real improvement**
- Hybrid vs Meaning (embeddings), answer similarity: +0.001 (95% range -0.000 to +0.002) -> **could be luck**
- Meaning (embeddings) vs Random, topic precision: +47.3% (95% range +45.2% to +49.3%) -> **real improvement**
- Meaning (embeddings) vs Random, answer similarity: +0.078 (95% range +0.076 to +0.080) -> **real improvement**

## Short message (first 15 words)

| Method               | Topic precision@5   |   Topic MRR | Same dept precision@5   |   Answer similarity (avg of 5) |   Median ms |
|:---------------------|:--------------------|------------:|:------------------------|-------------------------------:|------------:|
| Random               | 11.5%               |       0.222 | 17.2%                   |                          0.845 |           0 |
| Keyword (BM25)       | 53.8%               |       0.734 | 36.3%                   |                          0.908 |          61 |
| Meaning (embeddings) | 50.0%               |       0.68  | 36.0%                   |                          0.911 |           3 |
| Hybrid               | 55.2%               |       0.737 | 38.1%                   |                          0.914 |          64 |

### Is the difference real?

- Meaning (embeddings) vs Keyword (BM25), topic precision: -3.7% (95% range -5.7% to -1.8%) -> **real drop**
- Meaning (embeddings) vs Keyword (BM25), answer similarity: +0.003 (95% range +0.002 to +0.005) -> **real improvement**
- Hybrid vs Meaning (embeddings), topic precision: +5.2% (95% range +3.5% to +6.8%) -> **real improvement**
- Hybrid vs Meaning (embeddings), answer similarity: +0.003 (95% range +0.002 to +0.004) -> **real improvement**
- Meaning (embeddings) vs Random, topic precision: +38.5% (95% range +36.3% to +40.8%) -> **real improvement**
- Meaning (embeddings) vs Random, answer similarity: +0.067 (95% range +0.065 to +0.068) -> **real improvement**

## How to read this

- **Topic precision@5**: share of the top 5 that share 2+ specific tags with the test ticket (only the 910 test tickets with 2+ specific tags).
- **Topic MRR**: 1.0 if the first result matches, 0.5 if the second, ... 0 if none of the 5 match.
- **Same dept precision@5**: share of the top 5 in the same department (labels imperfect, see D4).
- **Answer similarity (avg of 5)**: average closeness of the 5 retrieved answers to the real answer. Compare against the Random row, not against 1.0.
- **95% range**: if the range includes 0, the difference could be luck.
- Knowledge base: 32,206 tickets. Keyword latency uses the pure-Python rank_bm25 library.
