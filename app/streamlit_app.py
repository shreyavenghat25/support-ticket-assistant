"""Agent screen.
Local with backend:  API_URL=http://localhost:8000 streamlit run app/streamlit_app.py
Standalone / cloud:  streamlit run app/streamlit_app.py   (runs the assistant in-process)
"""
import json
import os
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ui  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
API = os.getenv("API_URL")  # unset = embedded mode (no separate backend needed)
DAILY_LIMIT = int(os.getenv("DAILY_LIMIT", "200"))
LOG_DIR = Path(os.getenv("LOG_DIR", "/tmp/ticket-assistant-logs"))
LANES = {"auto": ("🟢 Auto-routed", "The 5 most similar past tickets strongly agree."),
         "suggest": ("🟡 Please confirm", "The similar tickets partly agree. Check the top 2."),
         "manual": ("🔴 Manual routing", "The similar tickets disagree. Please decide.")}
EXAMPLES = ["", "I was charged twice for my monthly subscription. Please refund the extra charge.",
            "I want to return the headphones I bought last week, they stopped charging after two days.",
            "Ich wurde zweimal für mein Abonnement belastet. Bitte erstatten Sie den Betrag.",
            "My broadband drops every evening around 8, I work from home and this is costing me money.",
            "My smart fridge keeps ordering 40 litres of milk every night through the Alexa integration."]


@st.cache_resource(show_spinner="Loading the search index and models (first time takes about a minute)...")
def get_assistant():
    sys.path.insert(0, str(ROOT))
    from services.assistant import Assistant
    return Assistant()


@st.cache_resource
def usage_counter():
    return {"day": date.today(), "count": 0}


def log(name, record):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / name).open("a") as f:
        f.write(json.dumps(record) + "\n")


def analyse(text):
    if API:
        r = requests.post(f"{API}/analyse", json={"text": text}, timeout=90)
        if r.status_code != 200:
            raise RuntimeError(r.json().get("detail", "Something went wrong."))
        return r.json()
    u = usage_counter()
    if u["day"] != date.today():
        u.update(day=date.today(), count=0)
    if u["count"] >= DAILY_LIMIT:
        raise RuntimeError("The demo has reached today's limit of AI drafts. Please try again tomorrow.")
    u["count"] += 1
    res = get_assistant().analyse(text)
    res["request_id"] = uuid.uuid4().hex[:12]
    d = res["diagnostics"]
    log("requests.jsonl", {"request_id": res["request_id"], "ts": datetime.now(timezone.utc).isoformat(),
                           "chars": len(text), "lane": res["routing"]["lane"], "reply_status": res["reply"]["status"],
                           "total_ms": d["total_ms"], "tokens_in": d["tokens_in"], "tokens_out": d["tokens_out"],
                           "llm_ok": d["llm_ok"], "new_issue": res["novelty"]["new_issue_suspected"]})
    return res


def send_feedback(request_id, helpful):
    if API:
        requests.post(f"{API}/feedback", json={"request_id": request_id, "helpful": helpful}, timeout=10)
    else:
        log("feedback.jsonl", {"request_id": request_id, "helpful": helpful,
                               "ts": datetime.now(timezone.utc).isoformat()})


st.set_page_config(page_title="Support Ticket Assistant", page_icon="🎫", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)
st.markdown(ui.hero(), unsafe_allow_html=True)
st.markdown(ui.note("If the app was asleep, the first analysis takes about a minute while the models load. After that, "
                    "most tickets take a few seconds; when the free AI service is busy, a draft can take up to about "
                    "15 seconds."), unsafe_allow_html=True)

example = st.selectbox("Try an example (optional)", EXAMPLES, format_func=lambda x: x[:90] or "Choose...")
text = st.text_area("Customer ticket", value=example, height=140)

if st.button("Analyse", type="primary"):
    if len(text.strip()) < 5:
        st.error("Please paste a ticket first.")
    elif len(text) > 5000:
        st.error("Please keep the ticket under 5,000 characters.")
    else:
        with st.spinner("Searching past tickets and drafting a reply..."):
            try:
                st.session_state["result"] = analyse(text)
            except requests.RequestException:
                st.error("The backend is not reachable. Is the API running?")
            except Exception as e:
                st.error(str(e))

res = st.session_state.get("result")
if res:
    rt, reply = res["routing"], res["reply"]
    nov = res.get("novelty") or {}
    new_issue = bool(nov.get("new_issue_suspected"))
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(ui.label("Routing"), unsafe_allow_html=True)
        if new_issue:
            st.markdown(ui.callout("alert", "Possible new issue: this ticket doesn't closely match any past ticket. "
                                            "Routing is set to manual; please flag it for review."),
                        unsafe_allow_html=True)
        st.markdown(ui.routing_card(rt, res.get("sentiment"), new_issue), unsafe_allow_html=True)
    with right:
        st.markdown(ui.label("Suggested reply"), unsafe_allow_html=True)
        status = reply["status"]
        if status == "draft":
            msg = ("ok", "Past tickets contain a proven fix. Review the reply, then send.")
        elif status == "escalate":
            msg = ("info", "No proven fix in past tickets, so the reply asks for details and the ticket goes to a "
                           "specialist. This is deliberate: the assistant doesn't guess.")
        else:
            msg = ("warn", "The AI draft is paused right now (busy or over the demo quota). "
                           "Routing and similar tickets below still work.")
        st.markdown(ui.callout(*msg), unsafe_allow_html=True)
        if reply.get("customer_reply"):
            st.text_area("Reply to the customer (edit before sending)", reply["customer_reply"], height=170,
                         key=f"reply-{res['request_id']}")
        if status == "draft" and reply["steps"]:
            st.markdown(ui.label("Steps · each cites its past ticket"), unsafe_allow_html=True)
            st.markdown(ui.steps_list(reply["steps"]), unsafe_allow_html=True)
        if reply.get("clarifying_questions"):
            st.markdown(ui.label("Questions to ask the customer"), unsafe_allow_html=True)
            st.markdown(ui.questions_list(reply["clarifying_questions"]), unsafe_allow_html=True)

    with st.expander("Similar past tickets used as sources"):
        st.markdown(ui.sources_html(res["similar_tickets"]), unsafe_allow_html=True)

    st.markdown(ui.label("Was this helpful?"), unsafe_allow_html=True)
    c1, c2, _ = st.columns([1, 1, 6])
    for col, lbl, val in [(c1, "👍 Yes", True), (c2, "👎 No", False)]:
        if col.button(lbl):
            send_feedback(res["request_id"], val)
            st.success("Thanks for the feedback!")
    st.markdown(ui.footer(res["request_id"], res["diagnostics"]), unsafe_allow_html=True)
