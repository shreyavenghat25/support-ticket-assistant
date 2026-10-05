# Day 5 (E) - Near-duplicate neighbours and routing confidence (1,000 test tickets)

- Neighbour slots taken by a near-copy (cosine >= 0.9) of another neighbour: **1348 of 5000 (27.0%)**.

## Current neighbours (as deployed)

| Lane             | Share   | Top-1 correct   | Top-2 correct   |
|:-----------------|:--------|:----------------|:----------------|
| Auto (4-5 agree) | 24%     | 81.1%           | 90.9%           |
| Confirm (3)      | 34%     | 57.4%           | 83.8%           |
| Manual (0-2)     | 42%     | 49.9%           | 71.7%           |
| of which 5 of 5  | 9%      | 90.3%           | -               |

## De-duplicated neighbours

| Lane             | Share   | Top-1 correct   | Top-2 correct   |
|:-----------------|:--------|:----------------|:----------------|
| Auto (4-5 agree) | 20%     | 78.0%           | 89.8%           |
| Confirm (3)      | 30%     | 61.2%           | 86.8%           |
| Manual (0-2)     | 49%     | 49.7%           | 72.1%           |
| of which 5 of 5  | 7%      | 94.0%           | -               |

Switch to de-duplicated neighbours only if Auto-lane accuracy holds or improves.
