# Day 2 - Search quality on 1,000 unseen test tickets

| Method               | Topic match@5   | Same dept@5   |   Best answer similarity | Other-language results   |   Median ms/query |
|:---------------------|:----------------|:--------------|-------------------------:|:-------------------------|------------------:|
| Keyword (BM25)       | 84.0%           | 86.3%         |                    0.963 | 13.4%                    |               232 |
| Meaning (embeddings) | 83.0%           | 87.3%         |                    0.964 | 14.7%                    |                 3 |
| Hybrid               | 85.2%           | 88.0%         |                    0.965 | 14.1%                    |               219 |

- **Topic match@5**: at least one of the top 5 shares 2+ specific tags with the test ticket.
- **Same dept@5**: at least one of the top 5 is in the same department (labels are imperfect, see D4).
- **Best answer similarity**: how close the best retrieved ticket's answer is to the real answer (cosine, higher is better).
- **Other-language results**: share of results in the other language (cross-lingual matching).
- Model: `intfloat/multilingual-e5-small`. Knowledge base: 32,206 tickets after removing exact copies.
