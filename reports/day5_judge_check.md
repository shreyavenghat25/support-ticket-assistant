# Day 5 (B) - Checking the LLM judge by hand

- Steps checked: 39 across 17 tickets (yes 32, no 6, unsure 1).
- **Human: supported** 84% of steps (excluding unsure).
- **LLM judge: supported** 100% of steps.
- Tickets where human and judge counted the same number of supported steps: 13 of 17.

|   Ticket |   Steps |   You: supported |   Judge: supported | Agree   |
|---------:|--------:|-----------------:|-------------------:|:--------|
|        0 |       2 |                0 |                  2 | no      |
|        1 |       2 |                2 |                  2 | yes     |
|       10 |       3 |                3 |                  3 | yes     |
|       11 |       3 |                3 |                  3 | yes     |
|       14 |       2 |                2 |                  2 | yes     |
|       15 |       2 |                0 |                  2 | no      |
|       22 |       2 |                2 |                  2 | yes     |
|       24 |       3 |                3 |                  3 | yes     |
|       27 |       1 |                1 |                  1 | yes     |
|       29 |       2 |                0 |                  2 | no      |
|       30 |       2 |                2 |                  2 | yes     |
|       32 |       2 |                2 |                  2 | yes     |
|       34 |       4 |                4 |                  4 | yes     |
|       35 |       2 |                2 |                  2 | yes     |
|       38 |       2 |                1 |                  2 | no      |
|       42 |       3 |                3 |                  3 | yes     |
|       49 |       2 |                2 |                  2 | yes     |

The judge was hidden during labelling. One reviewer, small sample.
