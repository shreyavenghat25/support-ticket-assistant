# Day 4 - Resolution drafting on 50 unseen test tickets

| Method                   |   Similarity to real answer | Steps supported by sources   | Helpfulness (1-5)   | Escalated   | Invalid citations removed   | Failed outputs   | Avg tokens (in/out)   |
|:-------------------------|----------------------------:|:-----------------------------|:--------------------|:------------|:----------------------------|:-----------------|:----------------------|
| Random past answer       |                       0.851 | -                            | -                   | -           | -                           | -                | free                  |
| Copy most similar answer |                       0.954 | -                            | -                   | -           | -                           | -                | free                  |
| LLM without sources      |                       0.904 | 59%                          | 3.30                | 56%         | n/a                         | 0                | 178/183               |
| LLM with sources         |                       0.922 | 100%                         | 4.65                | 66%         | 0/39                        | 0                | 1054/152              |

- Model: `gemini-3.5-flash-lite`, temperature 0, JSON output; sources = top 5 hybrid-search tickets.
- **Similarity to real answer**: embedding similarity between the draft and the hidden real agent answer (compare with the Random row). Real answers often only ask for details, so this is a rough signal.
- **Steps supported by sources** and **Helpfulness**: LLM judge (same model) given the same 5 past tickets. Self-judging can be lenient; validate on a hand-checked sample before trusting it.
- **Invalid citations removed**: steps citing a ticket that was not provided, dropped automatically.
- Small sample (50 tickets).
