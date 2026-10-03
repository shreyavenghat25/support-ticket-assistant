# New-issue detection (simulation)

Each row removes one whole department from the library, then streams its tickets in among normal traffic. A ticket is flagged when its 5 closest library matches are unusually dissimilar (threshold set so about 5% of normal tickets are flagged).

| New issue (held-out department)   | New tickets flagged   | False alarms (normal)   |   AUC | Flagged after adding 75 resolved   | Biggest flagged group: % new issue   |
|:----------------------------------|:----------------------|:------------------------|------:|:-----------------------------------|:-------------------------------------|
| Billing and Payments              | 35%                   | 6%                      |  0.74 | 11%                                | 3%                                   |
| Service Outages and Maintenance   | 15%                   | 4%                      |  0.7  | 5%                                 | 36%                                  |
| Returns and Exchanges             | 12%                   | 4%                      |  0.56 | 8%                                 | 28%                                  |

- **Flagged after adding 75 resolved**: once a few resolved tickets of the new kind join the library, new arrivals of that kind should stop being flagged (evolving data).
- **AUC**: 0.5 = no better than chance, 1.0 = perfect separation of new vs normal tickets.
- **Biggest flagged group**: flagged tickets are grouped by meaning; this is the share of the largest group that belongs to the new issue.

## Groups of flagged tickets

### Billing and Payments

|   Flagged tickets | Share from new issue   | Top keywords                                                   |
|------------------:|:-----------------------|:---------------------------------------------------------------|
|                39 | 3%                     | data, security, breach, performance, problems, synchronization |
|                26 | 92%                    | billing, charges, account, payment, recent, error              |
|                26 | 85%                    | billing, payment, options, information, details, plans         |
|                13 | 38%                    | mich, daten, datenverletzung, wurden, beim, systemausfall      |

### Service Outages and Maintenance

|   Flagged tickets | Share from new issue   | Top keywords                                               |
|------------------:|:-----------------------|:-----------------------------------------------------------|
|                47 | 36%                    | service, security, problems, data, network, investment     |
|                16 | 38%                    | alteryx, strategies, optimization, macos, windows, provide |

### Returns and Exchanges

|   Flagged tickets | Share from new issue   | Top keywords                                          |
|------------------:|:-----------------------|:------------------------------------------------------|
|                39 | 28%                    | platform, data, investment, breach, incident, problem |
|                16 | 44%                    | strategies, return, data, process, marketing, provide |

Production threshold (full library, 5% false-alarm budget): 0.0999, saved to `data/index/novelty.json`.

Caveats: department labels are imperfect (Day 1), and departments overlap in vocabulary, so some 'new' tickets genuinely resemble tickets from other departments.
