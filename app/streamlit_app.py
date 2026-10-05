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
st.title("Support Ticket Assistant")
st.caption("Finds similar past tickets, suggests routing, and drafts a reply grounded in past resolutions. "
           "A human agent reviews everything before it reaches the customer.")
st.info("If the app was asleep, the first analysis takes about a minute while the models load. "
        "After that, most tickets take a few seconds; when the free AI service is busy, a draft can take up to about 15 seconds.", icon="⏱️")

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
    left, right = st.columns(2)
    with left:
        st.subheader("Routing")
        nov = res.get("novelty") or {}
        if nov.get("new_issue_suspected"):
            st.warning("⚠️ Possible new issue: this ticket doesn't closely match any past ticket. "
                       "Routing is set to manual; please flag it for review.")
        title, why = LANES[rt["lane"]]
        if nov.get("new_issue_suspected"):
            why = ("Set to manual because the ticket looks unlike past tickets, so their vote may not apply. "
                   "Use the suggestion below as a hint only.")
        st.markdown(f"**{title}** ({rt['agreement']} similar tickets agree)  \n{why}")
        for d in rt["departments"]:
            st.write(f"- {d['name']} ({d['votes']} of 5 votes)")
        st.write(f"**Priority:** {rt['priority']}" + ("" if rt["priority_confirmed"] else " (please confirm)"))
        st.write(f"**Type:** {rt['type']}   **Customer mood:** {res.get('sentiment') or 'unknown'}")
    with right:
        st.subheader("Suggested reply")
        status = reply["status"]
        if status == "draft":
            st.success("Past tickets contain a proven fix. Review the reply, then send.")
        elif status == "escalate":
            st.info("No proven fix in past tickets, so the reply asks for details and the ticket goes to a "
                    "specialist. This is deliberate: the assistant doesn't guess.")
        else:
            st.warning("The AI draft is paused right now (busy or over the demo quota). "
                       "Routing and similar tickets below still work.")
        if reply.get("customer_reply"):
            st.text_area("Reply to the customer (edit before sending)", reply["customer_reply"], height=180,
                         key=f"reply-{res['request_id']}")
        if status == "draft" and reply["steps"]:
            st.markdown("**Steps, with the past ticket each one comes from:**")
            for n, s in enumerate(reply["steps"], 1):
                st.markdown(f"{n}. {s['text']}  `{', '.join(s['sources'])}`")
        if reply.get("clarifying_questions"):
            st.markdown("**Questions to ask the customer:**")
            for q in reply["clarifying_questions"]:
                st.write(f"- {q}")

    with st.expander("Similar past tickets used as sources"):
        for t in res["similar_tickets"]:
            st.markdown(f"**{t['id']}** [{t['language']}] [{t['department']}] {t['subject']}")
            st.caption(t["answer"])

    st.write("Was this helpful?")
    c1, c2, _ = st.columns([1, 1, 6])
    for col, label, val in [(c1, "👍 Yes", True), (c2, "👎 No", False)]:
        if col.button(label):
            send_feedback(res["request_id"], val)
            st.success("Thanks for the feedback!")
    d = res["diagnostics"]
    st.caption(f"Request {res['request_id']} · {d['total_ms']} ms · {d['tokens_in']}/{d['tokens_out']} tokens")
