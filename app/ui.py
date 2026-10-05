"""Neon 'agent console' look for the Streamlit app: one stylesheet plus small HTML helpers.

Everything shown from tickets or the AI goes through esc() before it is placed in HTML,
so customer text can never inject markup into the page.
Colours follow Tailwind's palette (slate, cyan, fuchsia, lime, amber, rose)."""
from html import escape

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800&family=JetBrains+Mono:wght@500&family=Inter:wght@400;500;600&display=swap');
:root{
  --bg:#070B14; --panel:#0D1424; --panel-2:#111A2E; --line:rgba(34,211,238,.22); --line-soft:rgba(148,163,184,.14);
  --ink:#E2E8F0; --muted:#94A3B8; --cyan:#22D3EE; --fuchsia:#E879F9; --lime:#A3E635; --amber:#FBBF24; --rose:#FB7185;
  --display:'Orbitron',system-ui,sans-serif; --mono:'JetBrains Mono',ui-monospace,Menlo,monospace; --body:'Inter',system-ui,sans-serif;
}
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(900px 420px at 12% -10%, rgba(34,211,238,.10), transparent 60%),
    radial-gradient(800px 380px at 95% 0%, rgba(232,121,249,.09), transparent 60%),
    repeating-linear-gradient(0deg, rgba(255,255,255,.012) 0 1px, transparent 1px 3px),
    var(--bg);
}
[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:2.2rem; max-width:1200px}
html, body, [data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"], .stTextArea textarea{font-family:var(--body)}
[data-testid="stMarkdownContainer"] p{color:var(--ink)}

/* hero */
.cx-hero{border:1px solid var(--line); border-radius:14px; padding:22px 26px; margin-bottom:18px;
  background:linear-gradient(135deg, rgba(13,20,36,.92), rgba(17,26,46,.78)); position:relative; overflow:hidden}
.cx-hero:after{content:""; position:absolute; inset:0 0 auto 0; height:2px;
  background:linear-gradient(90deg, var(--cyan), var(--fuchsia), transparent)}
.cx-eyebrow{font-family:var(--mono)!important; font-size:12px; letter-spacing:.18em; color:var(--cyan); text-transform:uppercase}
.cx-title{font-family:var(--display)!important; font-weight:800; font-size:clamp(26px,4vw,40px); line-height:1.15; margin:6px 0 8px;
  background:linear-gradient(90deg,#67E8F9,#E879F9); -webkit-background-clip:text; background-clip:text; color:transparent;
  text-shadow:0 0 28px rgba(34,211,238,.18)}
.cx-sub{color:var(--muted); font-size:15px; max-width:70ch; margin:0}
.cx-chips{display:flex; flex-wrap:wrap; gap:8px; margin-top:14px}
.cx-chip{font-family:var(--mono)!important; font-size:12px; color:var(--ink); border:1px solid var(--line-soft);
  background:rgba(148,163,184,.06); padding:4px 10px; border-radius:999px}
.cx-chip b{color:var(--cyan); font-weight:500}
.cx-note{font-family:var(--mono)!important; font-size:12px; color:var(--muted); margin:-6px 0 14px}

/* section labels and cards */
.cx-label{font-family:var(--mono)!important; font-size:12px; letter-spacing:.16em; text-transform:uppercase; color:var(--cyan);
  margin:6px 0 10px; display:flex; align-items:center; gap:10px}
.cx-label:after{content:""; flex:1; height:1px; background:linear-gradient(90deg,var(--line),transparent)}
.cx-card{background:var(--panel); border:1px solid var(--line-soft); border-radius:12px; padding:16px 18px; margin-bottom:12px}

/* lane badge and vote meter */
.cx-badge{display:inline-flex; align-items:center; gap:8px; font-family:var(--mono)!important; font-size:13px; font-weight:500;
  padding:5px 12px; border-radius:8px; border:1px solid currentColor}
.cx-badge i{width:8px; height:8px; border-radius:50%; background:currentColor; box-shadow:0 0 10px currentColor}
.cx-auto{color:var(--lime)} .cx-suggest{color:var(--amber)} .cx-manual{color:var(--rose)}
.cx-why{color:var(--muted); font-size:14px; margin:10px 0 14px}
.cx-meter{display:flex; gap:5px; margin:4px 0 2px}
.cx-meter span{flex:1; height:8px; border-radius:2px; background:rgba(148,163,184,.14)}
.cx-meter span.on{background:var(--cyan); box-shadow:0 0 8px rgba(34,211,238,.6)}
.cx-dept{display:flex; justify-content:space-between; gap:12px; font-size:14px; padding:8px 0; border-top:1px solid var(--line-soft)}
.cx-dept:first-of-type{border-top:none}
.cx-dept em{font-style:normal; font-family:var(--mono)!important; color:var(--muted); font-size:12px; white-space:nowrap}
.cx-meta{display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; margin-top:12px}
.cx-meta div{background:var(--panel-2); border:1px solid var(--line-soft); border-radius:8px; padding:8px 10px; min-width:0}
.cx-meta small{display:block; font-family:var(--mono)!important; font-size:11px; color:var(--muted); letter-spacing:.08em; text-transform:uppercase}
.cx-meta span{font-size:14px; color:var(--ink)}
.cx-meta span.warn{color:var(--amber)}
@media (max-width:640px){.cx-meta{grid-template-columns:1fr 1fr}}

/* callouts */
.cx-callout{border-radius:10px; padding:10px 14px; font-size:14px; margin-bottom:12px; border:1px solid; line-height:1.5}
.cx-ok{color:#D9F99D; border-color:rgba(163,230,53,.45); background:rgba(163,230,53,.07)}
.cx-info{color:#A5F3FC; border-color:rgba(34,211,238,.45); background:rgba(34,211,238,.07)}
.cx-warn{color:#FDE68A; border-color:rgba(251,191,36,.45); background:rgba(251,191,36,.07)}
.cx-alert{color:#FECDD3; border-color:rgba(251,113,133,.5); background:rgba(251,113,133,.08)}

/* steps, questions, sources */
.cx-list{list-style:none; padding:0; margin:0; counter-reset:s}
.cx-list li{position:relative; padding:9px 0 9px 34px; border-top:1px solid var(--line-soft); font-size:14px; color:var(--ink)}
.cx-list li:first-child{border-top:none}
.cx-list.num li:before{counter-increment:s; content:counter(s, decimal-leading-zero); position:absolute; left:0; top:9px;
  font-family:var(--mono); font-size:12px; color:var(--fuchsia)}
.cx-list.q li:before{content:"?"; position:absolute; left:4px; top:9px; font-family:var(--mono); color:var(--cyan)}
.cx-kb{font-family:var(--mono)!important; font-size:11px; color:var(--cyan); border:1px solid var(--line); border-radius:4px;
  padding:1px 6px; margin-left:6px; white-space:nowrap}
.cx-src{border-top:1px solid var(--line-soft); padding:10px 0}
.cx-src:first-child{border-top:none}
.cx-src-head{display:flex; flex-wrap:wrap; gap:6px; align-items:center; font-size:14px; color:var(--ink); font-weight:600}
.cx-src p{color:var(--muted)!important; font-size:13px; margin:6px 0 0}
.cx-tag{font-family:var(--mono)!important; font-size:11px; font-weight:500; color:var(--muted); border:1px solid var(--line-soft);
  border-radius:4px; padding:1px 6px}
.cx-foot{font-family:var(--mono)!important; font-size:12px; color:var(--muted); margin-top:6px}

/* Streamlit widgets */
.stTextArea textarea, [data-baseweb="select"] > div{background:var(--panel)!important; border:1px solid var(--line-soft)!important;
  border-radius:10px!important; color:var(--ink)!important}
.stTextArea textarea:focus{border-color:var(--cyan)!important; box-shadow:0 0 0 1px var(--cyan), 0 0 18px rgba(34,211,238,.2)!important}
[data-testid="stWidgetLabel"] p{font-family:var(--mono)!important; font-size:12px!important; letter-spacing:.12em;
  text-transform:uppercase; color:var(--muted)!important}
button[kind="primary"], [data-testid="stBaseButton-primary"]{
  background:linear-gradient(90deg,#22D3EE,#E879F9)!important; color:#070B14!important; border:none!important;
  font-family:var(--mono)!important; font-weight:600!important; letter-spacing:.08em; border-radius:10px!important;
  box-shadow:0 0 22px rgba(34,211,238,.25)}
button[kind="primary"] p, [data-testid="stBaseButton-primary"] p{color:#070B14!important; font-family:var(--mono)!important}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover{box-shadow:0 0 30px rgba(232,121,249,.4)}
button[kind="secondary"], [data-testid="stBaseButton-secondary"]{background:var(--panel)!important;
  border:1px solid var(--line-soft)!important; border-radius:10px!important}
[data-testid="stExpander"] details{background:var(--panel); border:1px solid var(--line-soft)!important; border-radius:12px}
[data-testid="stExpander"] summary p{font-family:var(--mono)!important; font-size:13px; color:var(--cyan)!important}
</style>
"""
# Markdown ends an HTML block at a blank line, so the stylesheet must not contain any.
CSS = "\n".join(line for line in _CSS.splitlines() if line.strip())

LANE = {
    "auto": ("cx-auto", "Auto-routed", "The 5 most similar past tickets strongly agree."),
    "suggest": ("cx-suggest", "Please confirm", "The similar tickets partly agree. Check the top 2."),
    "manual": ("cx-manual", "Manual routing", "The similar tickets disagree. Please decide."),
}


def esc(value):
    return escape(str(value if value is not None else ""))


def hero():
    return (
        '<div class="cx-hero"><div class="cx-eyebrow">Agent console · English + German</div>'
        '<div class="cx-title">Support Ticket Assistant</div>'
        '<p class="cx-sub">Finds similar past tickets, suggests which team should handle the ticket and how sure it is, '
        'and drafts a reply grounded in how those tickets were solved. A human agent reviews everything before it '
        'reaches the customer.</p><div class="cx-chips">'
        '<span class="cx-chip"><b>90%</b> routing accuracy when all 5 similar tickets agree</span>'
        '<span class="cx-chip"><b>32,206</b> past tickets</span>'
        '<span class="cx-chip"><b>every step</b> cites its source</span></div></div>'
    )


def note(text):
    return f'<div class="cx-note">{esc(text)}</div>'


def label(text):
    return f'<div class="cx-label">{esc(text)}</div>'


def callout(kind, text):
    return f'<div class="cx-callout cx-{kind}">{esc(text)}</div>'


def routing_card(rt, sentiment, new_issue):
    cls, title, why = LANE.get(rt["lane"], LANE["manual"])
    if new_issue:
        why = ("Set to manual because the ticket looks unlike past tickets, so their vote may not apply. "
               "Use the suggestion below as a hint only.")
    top_votes = rt["departments"][0]["votes"] if rt["departments"] else 0
    meter = "".join(f'<span class="{"on" if i < top_votes else ""}"></span>' for i in range(5))
    depts = "".join(f'<div class="cx-dept"><span>{esc(d["name"])}</span><em>{esc(d["votes"])} of 5 votes</em></div>'
                    for d in rt["departments"])
    prio_cls = "" if rt["priority_confirmed"] else ' class="warn"'
    prio = esc(rt["priority"]) + ("" if rt["priority_confirmed"] else " · confirm")
    return (
        f'<div class="cx-card"><span class="cx-badge {cls}"><i></i>{title} · {esc(rt["agreement"])} agree</span>'
        f'<div class="cx-why">{esc(why)}</div><div class="cx-meter">{meter}</div>{depts}'
        f'<div class="cx-meta"><div><small>Priority</small><span{prio_cls}>{prio}</span></div>'
        f'<div><small>Type</small><span>{esc(rt["type"])}</span></div>'
        f'<div><small>Mood</small><span>{esc(sentiment or "unknown")}</span></div></div></div>'
    )


def steps_list(steps):
    items = "".join(f'<li>{esc(s["text"])}' + "".join(f'<span class="cx-kb">{esc(c)}</span>' for c in s["sources"])
                    + "</li>" for s in steps)
    return f'<div class="cx-card"><ol class="cx-list num">{items}</ol></div>'


def questions_list(questions):
    items = "".join(f"<li>{esc(q)}</li>" for q in questions)
    return f'<div class="cx-card"><ul class="cx-list q">{items}</ul></div>'


def sources_html(tickets):
    rows = "".join(
        f'<div class="cx-src"><div class="cx-src-head"><span class="cx-kb">{esc(t["id"])}</span>'
        f'<span class="cx-tag">{esc(t["language"])}</span><span class="cx-tag">{esc(t["department"])}</span>'
        f'<span>{esc(t["subject"])}</span></div><p>{esc(t["answer"])}</p></div>'
        for t in tickets)
    return f'<div>{rows}</div>'


def footer(request_id, d):
    return (f'<div class="cx-foot">request {esc(request_id)} · {esc(d["total_ms"])} ms · '
            f'{esc(d["tokens_in"])}/{esc(d["tokens_out"])} tokens</div>')
