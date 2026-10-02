# Day 1 - Dataset exploration

**Rows:** 61,765  **Columns:** 16

## Missing values

- subject: 8.6%
- answer: 21.4%
- type: 21.3%
- version: 53.7%
- tag_1: 21.3%
- tag_2: 21.4%
- tag_3: 21.7%
- tag_4: 28.8%
- tag_5: 55.3%
- tag_6: 78.6%
- tag_7: 90.3%
- tag_8: 96.0%

Rows kept after dropping empty body/answer: **48,574**


## type
| value | count | share |
|---|---|---|
| Incident | 19442 | 40.0% |
| Request | 13948 | 28.7% |
| Problem | 10187 | 21.0% |
| Change | 4997 | 10.3% |

## queue
| value | count | share |
|---|---|---|
| Technical Support | 14184 | 29.2% |
| Product Support | 8953 | 18.4% |
| Customer Service | 7419 | 15.3% |
| IT Support | 5725 | 11.8% |
| Billing and Payments | 4871 | 10.0% |
| Returns and Exchanges | 2438 | 5.0% |
| Service Outages and Maintenance | 1912 | 3.9% |
| Sales and Pre-Sales | 1490 | 3.1% |
| Human Resources | 914 | 1.9% |
| General Inquiry | 668 | 1.4% |

## priority
| value | count | share |
|---|---|---|
| medium | 19651 | 40.5% |
| high | 18975 | 39.1% |
| low | 9948 | 20.5% |

## language
| value | count | share |
|---|---|---|
| en | 28254 | 58.2% |
| de | 20320 | 41.8% |

## version (possible proxy for data batches)
| value | count | share |
|---|---|---|
| nan | 19994 | 41.2% |
| 400.0 | 18594 | 38.3% |
| 52.0 | 9117 | 18.8% |
| 51.0 | 869 | 1.8% |

## Top 25 tags
| value | count | share |
|---|---|---|
| nan | 149895 | 38.6% |
| Tech Support | 24552 | 6.3% |
| IT | 24325 | 6.3% |
| Performance | 19995 | 5.1% |
| Feedback | 14699 | 3.8% |
| Bug | 14060 | 3.6% |
| Documentation | 13224 | 3.4% |
| Security | 11799 | 3.0% |
| Feature | 10123 | 2.6% |
| Disruption | 7104 | 1.8% |
| Outage | 6261 | 1.6% |
| Technical | 5804 | 1.5% |
| Network | 5582 | 1.4% |
| Product | 5099 | 1.3% |
| Resolution | 4173 | 1.1% |
| Sales | 3959 | 1.0% |
| Guidance | 3031 | 0.8% |
| Crash | 2961 | 0.8% |
| Recovery | 2826 | 0.7% |
| Billing | 2783 | 0.7% |
| Hardware | 2493 | 0.6% |
| Integration | 2356 | 0.6% |
| Customer | 2307 | 0.6% |
| Maintenance | 2297 | 0.6% |
| Payment | 2127 | 0.5% |

Tickets with at least one telecom-relevant tag: **30.6%**


## Text length (words)

|       |   body_words |   answer_words |
|:------|-------------:|---------------:|
| count |        48574 |          48574 |
| mean  |           55 |             58 |
| std   |           32 |             29 |
| min   |            1 |              1 |
| 25%   |           29 |             34 |
| 50%   |           53 |             58 |
| 75%   |           79 |             81 |
| max   |          281 |            226 |

## Duplicates

- Exact duplicates: **8,307**
- Near-duplicate groups (cosine >= 0.9): **39,740** groups for 48,574 tickets (18.2% of tickets are near-copies)

## What do agent answers contain? (heuristic)
| value | count | share |
|---|---|---|
| info_request_only | 25113 | 51.7% |
| other | 19236 | 39.6% |
| resolution | 4225 | 8.7% |

Spot-check 30 of each kind by hand and report the heuristic's precision.


## Baseline classifiers (TF-IDF + LogReg, group-aware 80/20 split)

- **queue** macro-F1: **0.461**
- **type** macro-F1: **0.837**
- **priority** macro-F1: **0.565**

These are the numbers your LLM/embedding approach must beat.


## Split
- Knowledge base: 38,875 tickets
- Test: 9,699 tickets
- Near-duplicate groups never cross the split (prevents leakage).
