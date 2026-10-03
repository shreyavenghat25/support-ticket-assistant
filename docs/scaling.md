# Scaling to production

What it costs today, what breaks first as volume grows, and what I would change.
Measured numbers come from this repository; estimates are labelled as such.

## 1. Cost per ticket

| Item | Measured |
|---|---|
| LLM input tokens per ticket (rules + 5 sources + ticket) | ~1,050-1,150 |
| LLM output tokens per ticket | ~120-150 |
| LLM calls per ticket | 1 (routing, priority and type use the free kNN vote) |

Estimate at paid prices, using Claude Haiku 4.5 ($1 / $5 per million input / output tokens) as a reference:

| Volume | Cost (estimate) |
|---|---|
| 1 ticket | ~$0.002 |
| 10,000 tickets / day | ~$18 / day, ~$540 / month |

Development and evaluation used the free Gemini tier. The LLM wrapper is one class, so switching providers is a configuration change.

**Ways to cut cost:**
- **Cache drafts for near-duplicate tickets.** About 17% of tickets in this dataset were exact copies (Day 1), so a cache keyed on the nearest-neighbour match could skip many LLM calls (estimate; real duplicate rates must be measured).
- **Shorter prompts:** truncate sources harder, or send 3 instead of 5 when agreement is 5 of 5.
- **Route by need:** routing never needs the LLM (D6); only drafting does.

## 2. Speed

| Step | Measured |
|---|---|
| Embedding search (32,206 tickets, in-memory) | ~3 ms |
| BM25 keyword search (pure-Python rank_bm25) | ~200 ms |
| LLM draft | ~1-1.3 s |
| Total per ticket | ~2-2.5 s |

The LLM call dominates and is network-bound, so throughput scales with concurrent requests rather than CPU.
10,000 tickets/day is ~0.12 tickets/second on average; a single instance with a few concurrent LLM calls covers that, and peaks (estimate: 10x) need ~3-4 concurrent calls plus headroom in the provider's rate limits.

## 3. What breaks first, and the fix

| Component today | Breaks when | Production replacement |
|---|---|---|
| Embeddings in a NumPy array (47 MB for 32k tickets) | ~1M+ tickets (~1.5 GB, slower brute-force search) | Vector database with approximate search (pgvector, Qdrant, or OpenSearch) |
| Pure-Python BM25, rebuilt in memory | Large KB or frequent updates: query time grows linearly, rebuilds take seconds | A search engine with incremental indexing and built-in hybrid search (OpenSearch / Elasticsearch) |
| `add_ticket` updates memory only | Restart loses added tickets | Persist resolved tickets to a database; index incrementally |
| Model and index cached in one process | Code or secret changes need a restart (seen during deployment) | Containerised, versioned deployments with blue-green rollout |
| One process serving UI and logic | Concurrent users | Separate API service behind a load balancer, autoscaled (e.g. Cloud Run or Kubernetes); UI scales independently |
| Daily request counter in memory | Multiple instances | Shared rate limiting (e.g. Redis) or an API gateway |

## 4. Reliability

- **LLM failure fallback:** if the draft fails or returns invalid JSON, agents still get routing and similar tickets (built).
- **Timeouts and retries:** short retries in the API path (built); add a circuit breaker so a provider outage doesn't slow every request.
- **Citation validation in code:** steps citing unseen tickets are removed (built).

## 5. Privacy and security

- API keys live in platform secrets, never in the repository (built).
- Ticket text is not logged; logs hold only length, lane, latency, tokens and outcome (built).
- Before production: redact personal data (names, account and phone numbers) before sending text to an external LLM, agree data-retention rules, and review the LLM provider's data-use terms.
- Adding tickets to the library requires an admin token (built).

## 6. Monitoring

| Metric | Why | Source |
|---|---|---|
| Accuracy per confidence lane | Is auto-routing still safe? Agent re-routes act as free labels | Agent overrides |
| Escalation rate | Rising = KB coverage gap or prompt regression | Request logs |
| New-issue rate | Rising = emerging problem (D12) | Request logs |
| Thumbs-up rate | Draft usefulness as judged by agents | Feedback endpoint |
| Groundedness (LLM judge on a sample) | Catch hallucination drift; re-calibrate the judge against human review monthly (D9) | Weekly sampled evaluation |
| Latency p50/p95, cost per day, LLM failure rate | Operations and budget | Request logs |

## 7. Safe changes

Every prompt or model change runs the existing evaluation sets first (retrieval, triage, drafting, and the 39 hand-labelled steps) and must not regress beyond agreed thresholds before deployment. The evaluation scripts in `scripts/` are the starting point for that regression suite.
