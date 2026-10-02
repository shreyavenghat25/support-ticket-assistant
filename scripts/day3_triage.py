"""
Day 3 - Label each ticket: department (top 2 + confidence), type, priority, sentiment.
Usage:
    python scripts/day3_triage.py try "I was charged twice this month"
    python scripts/day3_triage.py eval            # 200 test tickets
    python scripts/day3_triage.py eval 100        # smaller run
LLM answers are cached in data/day3_cache.jsonl, so an interrupted run resumes.
Output: reports/day3_triage.md
"""
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from day2_search import Searcher, ticket_text  # noqa: E402

logging.getLogger("google_genai").setLevel(logging.ERROR)

DEPARTMENTS = ["Technical Support", "Product Support", "Customer Service", "IT Support",
               "Billing and Payments", "Returns and Exchanges", "Service Outages and Maintenance",
               "Sales and Pre-Sales", "Human Resources", "General Inquiry"]
TYPES = ["Incident", "Request", "Problem", "Change"]
PRIORITIES = ["low", "medium", "high"]
SENTIMENTS = ["calm", "concerned", "frustrated", "angry"]
CACHE = Path("data/day3_cache.jsonl")
SEED = 0
PAUSE_SECONDS = 4
MAX_RETRIES = 6


class LLM:
    def __init__(self):
        from dotenv import load_dotenv
        from google import genai
        from google.genai import types
        load_dotenv()
        self.model = os.getenv("GEMINI_MODEL")
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.config = types.GenerateContentConfig(temperature=0, response_mime_type="application/json")

    def json(self, prompt):
        for attempt in range(MAX_RETRIES):
            try:
                start = time.perf_counter()
                r = self.client.models.generate_content(model=self.model, contents=prompt, config=self.config)
                ms = (time.perf_counter() - start) * 1000
                u = r.usage_metadata
                return json.loads(r.text), {"ms": ms, "in": getattr(u, "prompt_token_count", 0) or 0,
                                            "out": getattr(u, "candidates_token_count", 0) or 0}
            except json.JSONDecodeError:
                return None, {"ms": 0, "in": 0, "out": 0}
            except Exception as e:
                wait = 15 * (attempt + 1)
                print(f"    API error ({str(e)[:80]}...), waiting {wait}s")
                time.sleep(wait)
        return None, {"ms": 0, "in": 0, "out": 0}


INSTRUCTIONS = f"""You triage customer support tickets. Read the ticket and return JSON only:
{{"departments": [{{"name": <dept>, "confidence": <0-1>}}, {{"name": <dept>, "confidence": <0-1>}}],
  "type": <type>, "priority": <priority>, "sentiment": <sentiment>, "reason": <one short sentence>}}
- departments: the 2 most suitable, best first, from: {DEPARTMENTS}
- type: one of {TYPES} (Incident = something broke; Request = asking for something or information;
  Problem = recurring underlying issue; Change = asking to modify something)
- priority: one of {PRIORITIES}
- sentiment: one of {SENTIMENTS}
The ticket may be in English or German. Use the exact label spellings above."""


def prompt_plain(text):
    return f"{INSTRUCTIONS}\n\nTICKET:\n{text[:2000]}"


def prompt_rag(text, examples):
    ex = "\n\n".join(
        f"Example {i}: department={r['queue']}, type={r['type']}, priority={r['priority']}\n{str(r['text'])[:600]}"
        for i, (_, r) in enumerate(examples.iterrows(), 1))
    return (f"{INSTRUCTIONS}\n\nHere are similar past tickets with the labels agents gave them. "
            f"Use them as hints, but judge the new ticket on its own content.\n\n{ex}\n\nNEW TICKET:\n{text[:2000]}")


def clean(pred):
    if not isinstance(pred, dict):
        return None
    depts = [d.get("name") for d in pred.get("departments", []) if isinstance(d, dict)]
    depts = [d for d in depts if d in DEPARTMENTS][:2]
    if not depts:
        return None
    pr = str(pred.get("priority", "")).lower()
    return {"top2": depts, "type": pred.get("type") if pred.get("type") in TYPES else None,
            "priority": pr if pr in PRIORITIES else None,
            "sentiment": pred.get("sentiment") if pred.get("sentiment") in SENTIMENTS else None,
            "reason": pred.get("reason", "")}


def load_cache():
    if not CACHE.exists():
        return {}
    out = {}
    for line in CACHE.read_text().splitlines():
        rec = json.loads(line)
        out[rec["key"]] = rec
    return out


def save_cache(rec):
    with CACHE.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def macro_f1(y_true, y_pred, labels):
    from sklearn.metrics import f1_score
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    if not pairs:
        return float("nan")
    t, p = zip(*pairs)
    return f1_score(t, p, labels=labels, average="macro", zero_division=0)


def top2_acc(y_true, top2s):
    return float(np.mean([t in (p or []) for t, p in zip(y_true, top2s)]))


def try_one(text):
    s = Searcher()
    llm = LLM()
    ex = s.kb.iloc[s.hybrid_top(text, 5)]
    for name, prompt in [("LLM", prompt_plain(text)), ("LLM + similar tickets", prompt_rag(text, ex))]:
        pred, usage = llm.json(prompt)
        print(f"\n===== {name} ({usage['ms']:.0f} ms, {usage['in']} in / {usage['out']} out tokens) =====")
        print(json.dumps(pred, indent=2, ensure_ascii=False))


def evaluate(n):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    s = Searcher()
    llm = LLM()
    cache = load_cache()
    test = pd.read_parquet("data/test_tickets.parquet").sample(n, random_state=SEED).reset_index(drop=True)
    test["text"] = ticket_text(test)

    print("Training the Day 1 baseline on the knowledge base...")
    kb_full = pd.read_parquet("data/kb_tickets.parquet")
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)
    Xkb = vec.fit_transform(ticket_text(kb_full))
    Xte = vec.transform(test["text"])
    base = {}
    for col in ["queue", "type", "priority"]:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xkb, kb_full[col])
        order = np.argsort(-clf.predict_proba(Xte), axis=1)
        base[col] = [list(clf.classes_[o[:2]]) for o in order]

    preds = {m: {"top2": [], "type": [], "priority": [], "sentiment": []} for m in ["kNN", "LLM", "LLM+RAG"]}
    usage = {"LLM": [], "LLM+RAG": []}
    failures = Counter()
    for i, t in test.iterrows():
        print(f"Ticket {i + 1}/{n}")
        ex = s.kb.iloc[s.hybrid_top(t["text"], 5)]
        preds["kNN"]["top2"].append([v for v, _ in Counter(ex["queue"]).most_common(2)])
        preds["kNN"]["type"].append(Counter(ex["type"]).most_common(1)[0][0])
        preds["kNN"]["priority"].append(Counter(ex["priority"]).most_common(1)[0][0])
        preds["kNN"]["sentiment"].append(None)
        for m, prompt in [("LLM", prompt_plain(t["text"])), ("LLM+RAG", prompt_rag(t["text"], ex))]:
            key = f"{m}|{SEED}|{i}|{os.getenv('GEMINI_MODEL')}"
            if key in cache:
                rec = cache[key]
            else:
                raw, u = llm.json(prompt)
                rec = {"key": key, "pred": clean(raw), "usage": u}
                save_cache(rec)
                time.sleep(PAUSE_SECONDS)
            p = rec["pred"]
            if p is None:
                failures[m] += 1
            preds[m]["top2"].append(p["top2"] if p else None)
            for col in ["type", "priority", "sentiment"]:
                preds[m][col].append(p[col] if p else None)
            usage[m].append(rec["usage"])

    rows = [{
        "Method": "Baseline (TF-IDF + LogReg)",
        "Dept macro-F1": macro_f1(test["queue"], [p[0] for p in base["queue"]], DEPARTMENTS),
        "Dept top-2 accuracy": top2_acc(test["queue"], base["queue"]),
        "Type macro-F1": macro_f1(test["type"], [p[0] for p in base["type"]], TYPES),
        "Priority macro-F1": macro_f1(test["priority"], [p[0] for p in base["priority"]], PRIORITIES),
        "Failed outputs": 0, "Avg tokens (in/out)": "-", "Median ms": "-"}]
    names = {"kNN": "kNN vote (5 similar tickets)", "LLM": "LLM (ticket only)", "LLM+RAG": "LLM + 5 similar tickets"}
    for m in ["kNN", "LLM", "LLM+RAG"]:
        p, u = preds[m], usage.get(m, [])
        rows.append({
            "Method": names[m],
            "Dept macro-F1": macro_f1(test["queue"], [x[0] if x else None for x in p["top2"]], DEPARTMENTS),
            "Dept top-2 accuracy": top2_acc(test["queue"], p["top2"]),
            "Type macro-F1": macro_f1(test["type"], p["type"], TYPES),
            "Priority macro-F1": macro_f1(test["priority"], p["priority"], PRIORITIES),
            "Failed outputs": failures[m],
            "Avg tokens (in/out)": f"{np.mean([x['in'] for x in u]):.0f}/{np.mean([x['out'] for x in u]):.0f}" if u else "-",
            "Median ms": f"{np.median([x['ms'] for x in u if x['ms']]):.0f}" if u and any(x['ms'] for x in u) else "-"})
    table = pd.DataFrame(rows)
    for c in ["Dept macro-F1", "Type macro-F1", "Priority macro-F1"]:
        table[c] = table[c].map(lambda v: f"{v:.3f}")
    table["Dept top-2 accuracy"] = table["Dept top-2 accuracy"].map(lambda v: f"{v:.1%}")

    sent = Counter(x for x in preds["LLM+RAG"]["sentiment"] if x)
    report = (
        f"# Day 3 - Ticket triage on {n} unseen test tickets\n\n" + table.to_markdown(index=False) + "\n\n"
        f"- Model: `{os.getenv('GEMINI_MODEL')}`, temperature 0, JSON output.\n"
        "- **Dept top-2 accuracy**: the dataset's department is one of the two suggested (see decision D4).\n"
        "- Dataset department labels are often questionable (Day 1 hand check), so exact-match scores understate quality.\n"
        f"- Sentiment has no ground truth in the dataset; distribution from LLM + 5 similar tickets: {dict(sent)}.\n"
        f"- Small sample ({n} tickets): differences of a few points may be noise.\n"
    )
    Path("reports/day3_triage.md").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "try" and len(sys.argv) > 2:
        try_one(" ".join(sys.argv[2:]))
    elif cmd == "eval":
        evaluate(int(sys.argv[2]) if len(sys.argv) > 2 else 200)
    else:
        print(__doc__)
