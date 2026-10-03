"""Agent screen. Run: streamlit run app/streamlit_app.py"""
import os

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
LANES = {"auto": ("🟢 Auto-routed", "The 5 most similar past tickets strongly agree."),
         "suggest": ("🟡 Please confirm", "The similar tickets partly agree. Check the top 2."),
         "manual": ("🔴 Manual routing", "The similar tickets disagree. Please decide.")}
EXAMPLES = ["", "My broadband drops every evening around 8, I work from home and this is costing me money.",
            "I was charged twice for my monthly subscription. Please refund the extra charge.",
            "Ich wurde zweimal für mein Abonnement belastet. Bitte erstatten Sie den Betrag."]

st.set_page_config(page_title="Support Ticket Assistant", page_icon="🎫", layout="wide")
st.title("Support Ticket Assistant")
st.caption("Finds similar past tickets, suggests routing, and drafts a reply grounded in past resolutions. "
           "A human agent reviews everything before it reaches the customer.")

example = st.selectbox("Try an example (optional)", EXAMPLES, format_func=lambda x: x[:90] or "Choose...")
text = st.text_area("Customer ticket", value=example, height=140)

if st.button("Analyse", type="primary"):
    if len(text.strip()) < 5:
        st.error("Please paste a ticket first.")
    else:
        with st.spinner("Searching past tickets and drafting a reply..."):
            try:
                r = requests.post(f"{API}/analyse", json={"text": text}, timeout=90)
                r.raise_for_status()
                st.session_state["result"] = r.json()
            except requests.HTTPError:
                st.error(r.json().get("detail", "Something went wrong."))
            except requests.RequestException:
                st.error("The backend is not reachable. Is the API running?")

res = st.session_state.get("result")
if res:
    rt, reply = res["routing"], res["reply"]
    left, right = st.columns(2)
    with left:
        st.subheader("Routing")
        title, why = LANES[rt["lane"]]
        st.markdown(f"**{title}** ({rt['agreement']} similar tickets agree)  \n{why}")
        for d in rt["departments"]:
            st.write(f"- {d['name']} ({d['votes']} of 5 votes)")
        st.write(f"**Priority:** {rt['priority']}" + ("" if rt["priority_confirmed"] else " (please confirm)"))
        st.write(f"**Type:** {rt['type']}   **Customer mood:** {res.get('sentiment') or 'unknown'}")
    with right:
        st.subheader("Suggested reply")
        if reply["status"] == "draft":
            st.write(reply["summary"])
            for n, s in enumerate(reply["steps"], 1):
                st.markdown(f"{n}. {s['text']}  `{', '.join(s['sources'])}`")
        elif reply["status"] == "escalate":
            st.warning("No proven fix in past tickets. Escalate and ask the customer:")
        else:
            st.error("The AI draft is unavailable right now. Use the routing and similar tickets below.")
        for q in reply.get("clarifying_questions") or []:
            st.write(f"- {q}")

    with st.expander("Similar past tickets used as sources"):
        for t in res["similar_tickets"]:
            st.markdown(f"**{t['id']}** [{t['language']}] [{t['department']}] {t['subject']}")
            st.caption(t["answer"])

    st.write("Was this helpful?")
    c1, c2, _ = st.columns([1, 1, 6])
    for col, label, val in [(c1, "👍 Yes", True), (c2, "👎 No", False)]:
        if col.button(label):
            requests.post(f"{API}/feedback", json={"request_id": res["request_id"], "helpful": val}, timeout=10)
            st.success("Thanks for the feedback!")
    d = res["diagnostics"]
    st.caption(f"Request {res['request_id']} · {d['total_ms']} ms · {d['tokens_in']}/{d['tokens_out']} tokens")
