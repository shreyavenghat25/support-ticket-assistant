"""Safety rules that must never silently break. No models, no API keys, no network needed."""
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from day4_resolve import validate
from services.api import app
from services.assistant import Assistant, lane_for

ALLOWED = {"KB-1", "KB-2"}


# --- Citation guardrail (decision D7) ---------------------------------------------------
def test_step_citing_unseen_ticket_is_removed():
    draft = {"status": "draft", "steps": [
        {"text": "Restart the router", "sources": ["KB-1"]},
        {"text": "Invented fix", "sources": ["KB-999"]},
    ]}
    cleaned, removed, proposed = validate(draft, ALLOWED)
    assert [s["text"] for s in cleaned["steps"]] == ["Restart the router"]
    assert (removed, proposed) == (1, 2)


def test_step_without_any_citation_is_removed():
    cleaned, removed, _ = validate({"status": "draft", "steps": [{"text": "Trust me", "sources": []}]}, ALLOWED)
    assert cleaned["steps"] == [] and removed == 1


def test_draft_with_no_supported_steps_becomes_escalation():
    cleaned, _, _ = validate({"status": "draft", "steps": [{"text": "Invented", "sources": ["KB-9"]}]}, ALLOWED)
    assert cleaned["status"] == "escalate"


def test_invalid_llm_output_is_rejected():
    assert validate("not json", ALLOWED)[0] is None


# --- Confidence lanes (decision D8) -----------------------------------------------------
def test_confidence_lanes():
    assert lane_for(5) == "auto"
    assert lane_for(4) == "auto"
    assert lane_for(3) == "suggest"
    assert lane_for(2) == "manual"
    assert lane_for(1) == "manual"


# --- New-issue detection (decision D12) -------------------------------------------------
def _assistant_with_library(vectors, threshold):
    a = Assistant.__new__(Assistant)  # skip loading models
    a.search = SimpleNamespace(emb=np.array(vectors, dtype=np.float32))
    a.novelty_threshold = threshold
    return a


def test_familiar_ticket_is_not_flagged():
    a = _assistant_with_library([[1, 0, 0]] * 5 + [[0, 1, 0]] * 5, threshold=0.1)
    assert a.novelty(np.array([1, 0, 0], dtype=np.float32))["new_issue_suspected"] is False


def test_unfamiliar_ticket_is_flagged():
    a = _assistant_with_library([[1, 0, 0]] * 5 + [[0, 1, 0]] * 5, threshold=0.1)
    assert a.novelty(np.array([0, 0, 1], dtype=np.float32))["new_issue_suspected"] is True


def test_novelty_feature_off_without_threshold():
    a = _assistant_with_library([[1, 0, 0]] * 5, threshold=None)
    assert a.novelty(np.array([0, 0, 1], dtype=np.float32))["new_issue_suspected"] is False


# --- API protection ---------------------------------------------------------------------
client = TestClient(app)
TICKET = {"body": "Router keeps dropping", "answer": "Change the WiFi channel", "queue": "Technical Support",
          "priority": "low", "type": "Incident", "language": "en"}


def test_adding_tickets_requires_admin_token():
    assert client.post("/tickets", json=TICKET).status_code == 403
    assert client.post("/tickets", json=TICKET, headers={"x-admin-token": "wrong"}).status_code == 403


def test_too_short_ticket_is_rejected():
    assert client.post("/analyse", json={"text": "hi"}).status_code == 422


def test_health_endpoint():
    assert client.get("/health").json()["status"] == "ok"
