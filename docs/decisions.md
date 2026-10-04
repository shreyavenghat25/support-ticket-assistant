# Design decisions

Each decision records what I chose, the evidence behind it, and the alternatives I considered.
Hand checks were done by me, with an AI assistant helping on German translations and borderline cases.

## D1. Dataset

**Decision:** Use Tobi-Bueck/customer-support-tickets (English + German). Knowledge-base articles were planned but not added; the system retrieves from past tickets only (see README limitations).

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

**Decision:** Don't treat every past answer as a fix: the drafting prompt tells the model to ignore answers that only ask the customer for details. An LLM-based answer labeller, validated against my 30 hand-labelled answers, was planned but not built; it is listed as a next step.

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

**Decision:** Route department (top 2), priority and type with a kNN vote over the 5 most similar past tickets; use the LLM only for sentiment and (Day 4) drafting resolutions. Routing never depends on the LLM, so it still works if the LLM fails. The TF-IDF baseline is slightly better on type (0.848 vs 0.821), but I kept kNN for one consistent, model-free routing path.

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

## D7. Resolution drafting

**Decision:** Draft resolutions with the LLM grounded in the top 5 retrieved tickets. Every step must cite a provided source ID; code removes steps with missing or invalid citations; if no real fix exists in the sources, the status is "escalate" with clarifying questions. A human agent reviews every draft.

**Evidence (50 unseen test tickets, gemini-3.5-flash-lite, temperature 0):**
| Method | Similarity to real answer | Steps supported | Helpfulness (1-5) | Escalated |
|---|---|---|---|---|
| Random past answer | 0.851 | - | - | - |
| Copy most similar answer | 0.954 | - | - | - |
| LLM without sources | 0.904 | 59% | 3.30 | 56% |
| LLM with sources | 0.922 | 100% | 4.65 | 66% |

- Sources raise supported steps from 59% to 100% (LLM judge).
- 0 of 39 proposed steps had invalid citations; the code check is kept as a safety net.
- Escalation (66%) reflects the data: about half of historical answers contain no fix.
- Copying the top answer scores highest similarity because real answers mostly request details; similarity is used only as an on-topic check, not to choose methods.
- Cost: about 1,054 input / 152 output tokens per ticket with sources.

**Caveat:** the judge is the same model and may be lenient; validated with a hand check (Day 5).

**Alternatives considered:**
- Copy the most similar answer: free, but reproduces non-answers and can't combine fixes or adapt language.
- LLM without sources: cheaper, but 41% of steps unsupported.

## D8. Smart escalation (confidence-based routing)

**Decision:** Use agreement among the 5 retrieved tickets as a confidence signal, with three lanes:
- Auto-route when 4+ of 5 agree on the department.
- Suggest the top 2 (agent confirms with one click) when 3 of 5 agree.
- Manual routing when 2 or fewer agree.
Priority is auto-set only when all 5 agree; otherwise it is suggested for confirmation.

**Evidence (1,000 unseen test tickets, no LLM calls):**
- Department accuracy by agreement: 5/5 = 90%, 4/5 = 75%, 3/5 = 57%, 2/5 = 50%; agreement predicts correctness.
- Auto-route at 4+ agreement: 24% of tickets, 81% accuracy (91% top-2). At 5/5 only: 9% at 90%. Routing everything: 60%.
- Suggest lane (3 of 5): 34% of tickets, 84% top-2 accuracy.
- Priority is not monotonic (2/5 = 65% vs 3/5 = 54%); only 5/5 (85%) is reliable.

**Caveat:** accuracy is measured against imperfect labels (Day 1: in 14 of 30 baseline errors the label did not fit the text), so true accuracy is likely higher.

**Alternatives considered:**
- Route everything automatically: 60% accuracy, too many misroutes.
- Use the LLM's self-reported confidence: not calibrated (confidences did not sum to 1 in testing) and costs tokens.

## D9. Validating the LLM judge

**Decision:** Don't trust LLM-as-judge groundedness scores without human validation. Report the human-checked figure (84%), not the judge's (100%).

**Evidence (blind hand check of all 39 drafted steps from Day 4, judge hidden during labelling):**
- Human: 32 supported, 6 not supported, 1 unsure, so 84% supported (excluding unsure). LLM judge: 100%.
- Per ticket, human and judge agreed on 13 of 17 drafts.
- All 6 unsupported steps share one failure mode: recommending actions the customer or source ticket had already tried without success (restarts, patches, password resets, firewall updates, query optimisation, reducing server load). The judge missed these because the words appear in the sources.
- In 3 of 17 drafts (18%), every step was this kind of advice; these should have been escalations.
- Other observations: supported steps can still be unhelpful (e.g. "review the billing discrepancies"); one step mixed supported and unsupported actions; dataset placeholders (<tel_num>, <link>) were copied into drafts; near-duplicate sources occupied several retrieval slots.

**Caveats:** one reviewer, 39 steps.

**Next fix:** add rules to the drafting prompt ("never recommend actions marked as already tried") and to the judge ("check whether the source recommends the action or only reports it as tried"), then re-measure.

## D10. Fix: never recommend already-tried actions

**Decision:** Add a drafting rule: don't recommend actions the new ticket or a source says were already tried; escalate if only those remain. Add the same check to the judge.

**Evidence (same 50 tickets and sources as Day 4):**
- All 3 drafts I had marked fully unsupported now escalate, with follow-up questions that acknowledge what was already tried.
- 10 of 13 drafts I had marked fully supported still draft; 3 became escalations (tickets 14, 24, 30), a coverage cost.
- Overall: drafted fixes fell from 17 to 12 of 50 (escalation 66% to 76%); improved-judge support rose from 85% to 100%.
- Judge calibration: on the old drafts the improved judge gives 85% versus my hand check of 84% (original judge: 100%).

**Remaining limitation:** steps that mix a supported and an unsupported action (ticket 38) are not addressed by this rule.

**Trade-off:** fewer drafts, more trustworthy ones; wrong advice costs more than an escalation in support work.

## D11. Deployment

**Decision:** Live demo on Streamlit Community Cloud, with the Streamlit app calling the `Assistant` class in-process. The FastAPI service and Dockerfile remain the production design (separate backend and frontend in one container, deployable to any container platform).

**Why:**
- Hugging Face Docker Spaces now require a paid plan on the free CPU tier (HTTP 402 at creation).
- Streamlit Community Cloud is free and has enough memory for the embedding model and index; CPU-only PyTorch keeps memory down.
- The same `Assistant` code runs behind both the API and the standalone app, so behaviour is identical.

**Operational notes:**
- The Gemini key is stored in platform secrets, never in the repository.
- The model connection is cached in memory, so changing a secret requires an app reboot.
- A daily request limit protects the free LLM quota on a public demo; the ticket text itself is not logged.
- Free-tier apps sleep after inactivity; the first request after waking takes about a minute.

## D12. Evolving data and new issue types

**Decision:** Flag tickets whose 5 closest library matches are unusually dissimilar (novelty threshold set for about a 5% false-alarm rate), group flagged tickets by meaning, and surface groups with their top keywords as an emerging-issue alert for humans. Newly resolved tickets are added to the library, so known issues stop being flagged without retraining.

**Evidence (simulation: one whole department removed from the library, its tickets streamed in among normal traffic):**
| New issue | Flagged | False alarms | AUC | Flagged after adding 75 resolved |
|---|---|---|---|---|
| Billing and Payments | 35% | 6% | 0.74 | 11% |
| Service Outages and Maintenance | 15% | 4% | 0.70 | 5% |
| Returns and Exchanges | 12% | 4% | 0.56 | 8% |

- For billing, two groups of flagged tickets were 92% and 85% from the new issue, with keywords "billing, charges, account, payment": a clear alert without ever having seen a billing ticket.
- Outages and Returns were weak because those department labels don't correspond to distinct topics (Day 1 finding); their flagged groups had unrelated keywords.
- Adding resolved examples reduced flags in every scenario, so the library adapts to new data.

**Caveat:** holding out a department label is a pessimistic proxy for a new topic; genuinely new issues usually bring new vocabulary (product names, error codes), as billing does here.
