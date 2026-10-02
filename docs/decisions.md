# Design decisions

Each decision records what I chose, the evidence behind it, and the alternatives I considered.
Hand checks were done by me, with an AI assistant helping on German translations and borderline cases.

## D1. Dataset

**Decision:** Use Tobi-Bueck/customer-support-tickets (English + German). Supplement it later with a small set of telecom knowledge-base articles that I write myself.

**Evidence:**
- 61,765 tickets; 13,191 had no body or agent answer, leaving **48,574 usable tickets** (58% English, 42% German).
- Labelled with queue (10 departments), type (4), priority (3), language and up to 8 tags.
- Tickets are short (median 53 words), so each ticket can be stored whole, without chunking.

**Known gaps:**
- Synthetic, general IT/business data, not telecom-specific; 30.6% of tickets carry telecom-relevant tags (network, outage, billing, etc.).
- No timestamps, so a time-based split is impossible.
- No knowledge-base articles.
- 8.6% of tickets have no subject; in a 30-ticket sample, 3 had the agent's reply in place of the customer's message.

**Alternatives considered:**
- Fully synthetic telecom tickets: closer to the brief's domain, but I would be grading the system on data I created myself.
- santhoshmishra/Ticket_data: not yet evaluated.

**License:** CC BY-NC 4.0, suitable for a non-commercial assignment.

## D2. Knowledge/test split

**Decision:** Group-aware 80/20 split. Tickets with cosine similarity >= 0.90 form one group, and a group never crosses the split. Result: **38,875 knowledge-base tickets, 9,699 test tickets.**

**Evidence:**
- 8,307 exact duplicates, and 18.2% of tickets are exact or near-copies (39,740 groups for 48,574 tickets).
- 466 groups (994 tickets) are reworded copies that an exact-match check would miss.
- A random split would let a test ticket's copy sit in the knowledge base, inflating retrieval scores.

**Alternatives considered:**
- Time-based split: best practice in production, but the data has no timestamps.
- Random split: simple, but leaks copies across the split.

## D3. Agent answers that don't resolve anything

**Decision:** Prefer past tickets whose answers contain real help when drafting resolutions. Replace the word-pattern answer labeller with an LLM-based labeller, validated against my 30 hand-labelled answers.

**Evidence:**
- A word-pattern labeller marked 51.7% of answers as info-request-only, 39.6% as other and 8.7% as resolution.
- Hand check of 30 answers (10 per label): the label was right 60% (info-request), 90% (other) and only **30% (resolution)** of the time. It was fooled by words like "update" and "steps" that described the problem rather than a fix, and it missed useful advice that came before a closing "please provide details".
- Answers that actually helped the customer: 40% / 30% / 30% by label. Weighted by real label shares, about **35% of agent answers genuinely help** (small sample, approximate).

**Alternatives considered:**
- Treat every answer as a resolution: the system would copy "please send your logs" as a fix.
- Drop non-resolution tickets entirely: loses the signal about which details agents need, which can become suggested clarifying questions.

## D4. Department (queue) prediction and how to score it

**Decision:** Predict the **top 2 departments with a confidence score** rather than forcing one. Report both exact-match macro-F1 and a top-2 score.

**Evidence:**
- Baseline (TF-IDF + logistic regression): queue macro-F1 **0.46**, type **0.84**, priority **0.57**. Accuracy (0.47) is close to macro-F1, so small departments are not the main cause.
- Billing and Payments scores 0.71 (distinctive vocabulary); Technical, IT, Product Support and Customer Service score 0.41-0.50 (overlapping language).
- Labels are consistent across copies (100% agreement in 8,306 duplicated texts and 466 reworded groups), but **consistent is not the same as correct**.
- Hand check of 30 baseline errors: in **14 the dataset label did not fit the text** (e.g. a software bug under Human Resources), in **11 both departments were reasonable**, and only **5 were real model mistakes**.

**Alternatives considered:**
- Single-label prediction scored on exact match only: understates real usefulness, given how often two departments fit.
- Merging overlapping departments: simpler, but changes the task the brief defines.

## D5. Retrieval method

**Decision:** Hybrid search (keyword BM25 + multilingual embeddings, merged with Reciprocal Rank Fusion). Replace the pure-Python BM25 library with a faster one before production.

**Evidence (1,000 unseen test tickets, knowledge base of 32,206 tickets after removing 6,669 exact copies):**
- First evaluation ("any of top 5 matches") put all three methods within 1-2 points; that metric saturated, so I rebuilt it.
- Sharper evaluation, topic precision@5 on full tickets: Random 12.2%, Meaning 59.5%, Keyword 63.6%, **Hybrid 64.1%**.
- Hybrid vs Meaning: +4.6 points (95% CI +3.1 to +6.1) on full tickets, +5.2 points on short 15-word messages: a real improvement.
- Keyword vs Meaning: Keyword ahead by about 4 points (CI excludes zero), likely because this synthetic data reuses vocabulary.
- Short messages lower every method by about 10 points equally; my prediction that keyword search would suffer more was wrong.
- Hand test: for "internet keeps disconnecting every evening", keyword search returned unrelated tickets sharing the word "evening", while embeddings found real Wi-Fi disconnection tickets. Hybrid covers both exact-term and paraphrase cases.

**Alternatives considered:**
- Embeddings only: fastest (3 ms) and best on paraphrases, but about 5 points lower topic precision here. This was my first choice before the sharper evaluation reversed it.
- Keyword only: strong on this data, but fails on paraphrased real-world messages.

**Known limitation:** latency is 217 ms per query because rank_bm25 is pure Python; an optimised BM25 implementation should bring this to a few milliseconds.

## D6. Ticket triage: who labels what

**Decision:** Route department (top 2) and priority with a kNN vote over the 5 most similar past tickets; predict type with the TF-IDF baseline; use the LLM only for sentiment, a short explanation for the agent, and (Day 4) drafting resolutions. If the LLM output is invalid, fall back to kNN.

**Evidence (200 unseen test tickets, gemini-3.5-flash-lite, temperature 0):**
| Method | Dept macro-F1 | Dept top-2 | Type macro-F1 | Priority macro-F1 |
|---|---|---|---|---|
| Baseline (TF-IDF + LogReg) | 0.379 | 59.0% | 0.848 | 0.557 |
| kNN vote (5 similar tickets) | 0.474 | 77.5% | 0.821 | 0.637 |
| LLM (ticket only) | 0.180 | 41.0% | 0.577 | 0.362 |
| LLM + 5 similar tickets | 0.381 | 63.5% | 0.781 | 0.605 |

- The LLM alone applies common sense, but the dataset has its own labelling conventions (Day 1: many labels don't fit the text), so it scores below the baseline.
- Adding 5 retrieved tickets roughly doubles LLM department and priority scores, showing retrieval transfers the conventions.
- The free kNN vote still wins on department and priority, at zero LLM cost.
- LLM returned invalid JSON in 1-2 of 200 calls, handled by fallback.
- Sentiment (no ground truth): mostly calm or concerned, 7 frustrated, 0 angry, consistent with politely written synthetic tickets.

**Caveats:** small sample (200); scores are measured against imperfect labels, so some LLM "errors" may be label errors.

**Alternatives considered:**
- LLM for all labels: costlier and less accurate on this data.
- Fine-tuning a model on the labels: possible, but kNN already works and updates as new tickets arrive, without retraining.
