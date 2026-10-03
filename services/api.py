"""FastAPI backend. Run: python -m uvicorn services.api:app --port 8000"""
import json
import os
import statistics
import time
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from services.assistant import Assistant

LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(exist_ok=True)
DAILY_LIMIT = int(os.getenv("DAILY_LIMIT", "200"))  # protects the free LLM quota on a public demo

app = FastAPI(title="Support Ticket Resolution Assistant", version="1.0")
_assistant = None
_usage = {"day": date.today(), "count": 0}
_recent = []


def assistant():
    global _assistant
    if _assistant is None:
        _assistant = Assistant()
    return _assistant


def log(name, record):
    with (LOG_DIR / name).open("a") as f:
        f.write(json.dumps(record) + "\n")


class TicketIn(BaseModel):
    text: str = Field(min_length=5, max_length=5000)


class FeedbackIn(BaseModel):
    request_id: str
    helpful: bool
    comment: str | None = Field(default=None, max_length=1000)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _assistant is not None}


@app.post("/analyse")
def analyse(ticket: TicketIn):
    if _usage["day"] != date.today():
        _usage.update(day=date.today(), count=0)
    if _usage["count"] >= DAILY_LIMIT:
        raise HTTPException(429, "Daily demo limit reached. Please try again tomorrow.")
    _usage["count"] += 1

    request_id = uuid.uuid4().hex[:12]
    start = time.perf_counter()
    try:
        result = assistant().analyse(ticket.text)
    except Exception as e:
        log("errors.jsonl", {"request_id": request_id, "error": str(e)[:300],
                             "ts": datetime.now(timezone.utc).isoformat()})
        raise HTTPException(500, "Analysis failed. The error was logged.")
    d = result["diagnostics"]
    record = {"request_id": request_id, "ts": datetime.now(timezone.utc).isoformat(),
              "chars": len(ticket.text),
              "lane": result["routing"]["lane"], "reply_status": result["reply"]["status"],
              "total_ms": round((time.perf_counter() - start) * 1000), "tokens_in": d["tokens_in"],
              "tokens_out": d["tokens_out"], "llm_ok": d["llm_ok"],
              "invalid_citations_removed": d["invalid_citations_removed"]}
    log("requests.jsonl", record)
    _recent.append(record)
    return {"request_id": request_id, **result}


@app.post("/feedback")
def feedback(fb: FeedbackIn):
    log("feedback.jsonl", {**fb.model_dump(), "ts": datetime.now(timezone.utc).isoformat()})
    return {"status": "thanks"}


@app.get("/metrics")
def metrics():
    fb_path = LOG_DIR / "feedback.jsonl"
    fb = [json.loads(l) for l in fb_path.read_text().splitlines()] if fb_path.exists() else []
    n = len(_recent)
    count = lambda key, val: sum(r[key] == val for r in _recent)  # noqa: E731
    return {
        "requests": n,
        "median_latency_ms": statistics.median([r["total_ms"] for r in _recent]) if n else None,
        "lanes": {l: count("lane", l) for l in ["auto", "suggest", "manual"]},
        "escalation_rate": round(count("reply_status", "escalate") / n, 3) if n else None,
        "llm_failure_rate": round(count("llm_ok", False) / n, 3) if n else None,
        "avg_tokens_in": round(sum(r["tokens_in"] for r in _recent) / n) if n else None,
        "feedback_count": len(fb),
        "helpful_rate": round(sum(f["helpful"] for f in fb) / len(fb), 3) if fb else None,
    }
