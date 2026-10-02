import pandas as pd, textwrap
path = "reports/queue_errors.csv"
s = pd.read_csv(path)
s["who_is_right"] = s["who_is_right"].astype("object")
keys = {"l": "label", "m": "model", "b": "both"}
for i, r in s.iterrows():
    if isinstance(r.who_is_right, str) and r.who_is_right in keys.values():
        continue
    print("\n" + "=" * 70)
    print(f"Ticket {i+1} of {len(s)}")
    print(f"SUBJECT: {r.subject}")
    print("BODY:", textwrap.fill(str(r.body), 70))
    print(f"\n  Dataset says: {r.queue}")
    print(f"  Model says:   {r.predicted}")
    a = ""
    while a not in keys and a != "q":
        a = input("\n  [l] label right  [m] model right  [b] both fine  [q] quit > ").strip().lower()
    if a == "q":
        break
    s.at[i, "who_is_right"] = keys[a]
    s.to_csv(path, index=False)
print("\nCounts so far:")
print(s.who_is_right.value_counts())
