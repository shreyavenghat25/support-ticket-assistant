# Day 6 (B) - Length-aware new-issue threshold

Every ticket here is a normal test ticket (its topic exists in the library), so every flag is a false alarm.

| Normal test tickets      | Flagged with current threshold   | Flagged with length-aware thresholds   |
|:-------------------------|:---------------------------------|:---------------------------------------|
| Full tickets             | 4.4%                             | 4.4%                                   |
| Short (20-word) versions | 23.0%                            | 5.0%                                   |

- Full-ticket threshold (unchanged): 0.0999
- New short-ticket threshold (tickets of 30 words or fewer): 0.1155

The D12 simulation results were measured on full tickets and still apply to them.
