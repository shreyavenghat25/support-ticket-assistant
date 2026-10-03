# Day 5 (C) - Fixing the 'already tried' failure

## Did the fix work on the drafts you hand-checked?

|   Ticket | Your verdict (old draft)   | Old prompt   | New prompt   |   New steps |
|---------:|:---------------------------|:-------------|:-------------|------------:|
|        0 | bad                        | draft        | escalate     |           0 |
|        1 | good                       | draft        | draft        |           1 |
|       10 | good                       | draft        | draft        |           2 |
|       11 | good                       | draft        | draft        |           3 |
|       14 | good                       | draft        | escalate     |           0 |
|       15 | bad                        | draft        | escalate     |           0 |
|       22 | good                       | draft        | draft        |           4 |
|       24 | good                       | draft        | escalate     |           0 |
|       27 | good                       | draft        | draft        |           2 |
|       29 | bad                        | draft        | escalate     |           0 |
|       30 | good                       | draft        | escalate     |           0 |
|       32 | good                       | draft        | draft        |           2 |
|       34 | good                       | draft        | draft        |           5 |
|       35 | good                       | draft        | draft        |           1 |
|       38 | mixed                      | draft        | draft        |           3 |
|       42 | good                       | draft        | draft        |           3 |
|       49 | good                       | draft        | draft        |           2 |

- Bad drafts (all steps already-tried) now escalated: **3 of 3**
- Good drafts still drafted (not over-escalated): **10 of 13**

## Overall on the 50 tickets

| Prompt                   |   Drafted (of 50) | Escalated   |   Steps | Improved judge: supported   |
|:-------------------------|------------------:|:------------|--------:|:----------------------------|
| Old (Day 4)              |                17 | 66%         |      39 | 85%                         |
| New (already-tried rule) |                12 | 76%         |      29 | 100%                        |

## Does the improved judge agree with you?

| Who checked the old drafts   | Supported   |
|:-----------------------------|:------------|
| Original LLM judge           | 100%        |
| Improved LLM judge           | 85%         |
| You (hand check)             | 84%         |

- Same 50 tickets and sources as Day 4; only the instructions changed.
- Small sample; new drafts for the previously bad tickets are printed below the table in the terminal.
