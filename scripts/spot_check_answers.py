import pandas as pd, textwrap, os
path = "reports/spot_check.csv"
if not os.path.exists(path):
    df = pd.read_parquet("data/kb_tickets.parquet")
    s = df.groupby("answer_kind").sample(10, random_state=1)
    s[["answer_kind", "subject", "answer"]].assign(correct="", is_real_answer="").to_csv(path, index=False)
s = pd.read_csv(path)
for c in ["correct", "is_real_answer"]:
    s[c] = s[c].astype("object")
meaning = {"resolution": "gives real fix steps",
           "info_request_only": "ONLY asks the customer for more details",
           "other": "neither of the above"}
def ask(q):
    a = ""
    while a not in ("y", "n", "q"):
        a = input(q).strip().lower()
    return a
for i, r in s.iterrows():
    if r.correct in ("Y", "N") and r.is_real_answer in ("Y", "N"):
        continue
    print("\n" + "=" * 70)
    print(f"Ticket {i+1} of {len(s)}   |   SUBJECT: {r.subject}")
    print("AGENT ANSWER:", textwrap.fill(str(r.answer), 70))
    print(f"\n  Script's label: {r.answer_kind}  (= {meaning[r.answer_kind]})")
    a = ask("\n  1) Is the script's label right?  [y/n, q=quit] > ")
    if a == "q": break
    b = ask("  2) Does the answer actually help the customer (a fix OR the info they asked for)?  [y/n, q=quit] > ")
    if b == "q": break
    s.at[i, "correct"], s.at[i, "is_real_answer"] = a.upper(), b.upper()
    s.to_csv(path, index=False)
done = s[s.correct.isin(["Y", "N"])]
print(f"\nDone: {len(done)} of {len(s)}")
if len(done):
    print((done.assign(correct=done.correct.eq("Y"), is_real_answer=done.is_real_answer.eq("Y"))
              .groupby("answer_kind")[["correct", "is_real_answer"]].mean().mul(100).round(0)))
