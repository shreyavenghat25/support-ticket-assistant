# Day 5 - Smart escalation on 1,000 unseen test tickets

The 5 most similar past tickets vote on the label. Strong agreement = confident = route automatically; weak agreement = send to a human.

### Department: accuracy by agreement level

| Agreement (of 5)   |   Tickets | Share   | Top-1 correct   | Top-2 correct   |
|:-------------------|----------:|:--------|:----------------|:----------------|
| 5 of 5             |        93 | 9%      | 90%             | 90%             |
| 4 of 5             |       150 | 15%     | 75%             | 91%             |
| 3 of 5             |       340 | 34%     | 57%             | 84%             |
| 2 of 5             |       398 | 40%     | 50%             | 72%             |
| 1 of 5             |        19 | 2%      | 47%             | 63%             |

### Department: automation rules

| Rule: auto-route if   | Automated (coverage)   | Accuracy on automated   | Sent to humans   | Accuracy if humans' share were auto-routed   |
|:----------------------|:-----------------------|:------------------------|:-----------------|:---------------------------------------------|
| 5+ of 5 agree         | 9%                     | 90%                     | 91%              | 57%                                          |
| 4+ of 5 agree         | 24%                    | 81%                     | 76%              | 53%                                          |
| 3+ of 5 agree         | 58%                    | 67%                     | 42%              | 50%                                          |
| 2+ of 5 agree         | 98%                    | 60%                     | 2%               | 47%                                          |
| 1+ of 5 agree         | 100%                   | 60%                     | 0%               | -                                            |

### Priority: accuracy by agreement level

| Agreement (of 5)   |   Tickets | Share   | Top-1 correct   | Top-2 correct   |
|:-------------------|----------:|:--------|:----------------|:----------------|
| 5 of 5             |       102 | 10%     | 85%             | 85%             |
| 4 of 5             |       270 | 27%     | 69%             | 92%             |
| 3 of 5             |       445 | 44%     | 54%             | 91%             |
| 2 of 5             |       183 | 18%     | 65%             | 78%             |

### Priority: automation rules

| Rule: auto-route if   | Automated (coverage)   | Accuracy on automated   | Sent to humans   | Accuracy if humans' share were auto-routed   |
|:----------------------|:-----------------------|:------------------------|:-----------------|:---------------------------------------------|
| 5+ of 5 agree         | 10%                    | 85%                     | 90%              | 61%                                          |
| 4+ of 5 agree         | 37%                    | 73%                     | 63%              | 57%                                          |
| 3+ of 5 agree         | 82%                    | 63%                     | 18%              | 65%                                          |
| 2+ of 5 agree         | 100%                   | 63%                     | 0%               | -                                            |
| 1+ of 5 agree         | 100%                   | 63%                     | 0%               | -                                            |

- **Coverage**: share of tickets the rule routes automatically.
- **Accuracy on automated**: how often the automatic label matches the dataset label.
- Dataset labels are imperfect (Day 1 hand check: many department labels don't fit the text), so true accuracy is likely higher than shown.
