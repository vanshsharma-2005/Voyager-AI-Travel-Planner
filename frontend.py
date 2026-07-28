"""
frontend.py
-----------
Drop this file next to your existing `tools/` folder and `.env`:

    travel_planner/
    ├── frontend.py      <-- this file
    ├── tools/
    │   ├── flight_tool.py
    │   └── tavily_tool.py
    └── .env

Run locally:      streamlit run frontend.py
Deploy:            push to GitHub, then on Streamlit Cloud set these secrets
                    (Settings -> Secrets):

                        GROQ_API_KEY = "..."
                        TAVILY_API_KEY = "..."
                        AVIATIONSTACK_API_KEY = "..."
                        DATABASE_URL = "postgresql://..."   # optional

Note on secrets: your original notebook had the Groq key and DB password
hardcoded in plaintext. Neither is hardcoded here -- both are read from
`.env` locally / Streamlit secrets when deployed. If that key was ever
written in cleartext, it's worth rotating in the Groq console regardless.

Note on DATABASE_URL: if it's unset or unreachable (e.g. it points at
localhost, which won't exist on Streamlit Cloud), this falls back to an
in-memory checkpointer automatically -- the app still works, conversation
memory just won't survive a restart.
"""

import os
import operator
import uuid
from typing import TypedDict, Annotated

import streamlit as st
import streamlit.components.v1 as components

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq

from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights

st.set_page_config(
    page_title="Voyager — AI Travel Planner",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Secrets -> env (Streamlit secrets locally fall back to .env via os.getenv)
# ---------------------------------------------------------------------------
for key in ("GROQ_API_KEY", "TAVILY_API_KEY", "AVIATIONSTACK_API_KEY", "DATABASE_URL"):
    if key in st.secrets:
        os.environ[key] = st.secrets[key]


def _get_key(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


# ---------------------------------------------------------------------------
# Graph: flight_agent -> hotel_agent -> itinerary_agent -> final_agent
# ---------------------------------------------------------------------------
class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int


@st.cache_resource(show_spinner=False)
def build_pipeline():
    """Builds the LLM, graph, and checkpointer once per app process."""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=_get_key("GROQ_API_KEY"),
    )

    def flight_agent(state: TravelState):
        flight_data = search_flights(state["user_query"])
        return {
            "flight_results": flight_data,
            "messages": [AIMessage(content="Flight results fetched")],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    def hotel_agent(state: TravelState):
        hotel_results = tavily_search(f"Best hotels for {state['user_query']}")
        return {
            "hotel_results": hotel_results,
            "messages": [AIMessage(content="Hotel information fetched")],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    def itinerary_agent(state: TravelState):
        prompt = f"""
        Create a travel itinerary.
        User Query:
        {state['user_query']}

        Flight Results:
        {state['flight_results']}

        Hotel Results:
        {state['hotel_results']}
        """
        response = llm.invoke([
            SystemMessage(content="You are an expert travel planner"),
            HumanMessage(content=prompt),
        ])
        return {
            "itinerary": response.content,
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    def final_agent(state: TravelState):
        final_prompt = f"""
        Generate final travel response.

        Flights:
        {state['flight_results']}

        Hotels:
        {state['hotel_results']}

        Itinerary:
        {state['itinerary']}
        """
        response = llm.invoke([HumanMessage(content=final_prompt)])
        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    graph = StateGraph(TravelState)
    graph.add_node("flight_agent", flight_agent)
    graph.add_node("hotel_agent", hotel_agent)
    graph.add_node("itinerary_agent", itinerary_agent)
    graph.add_node("final_agent", final_agent)
    graph.add_edge(START, "flight_agent")
    graph.add_edge("flight_agent", "hotel_agent")
    graph.add_edge("hotel_agent", "itinerary_agent")
    graph.add_edge("itinerary_agent", "final_agent")
    graph.add_edge("final_agent", END)

    using_postgres = False
    db_url = _get_key("DATABASE_URL")
    checkpointer = None
    if db_url:
        try:
            import psycopg
            from psycopg.rows import dict_row
            from langgraph.checkpoint.postgres import PostgresSaver

            conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
            checkpointer = PostgresSaver(conn)
            checkpointer.setup()
            using_postgres = True
        except Exception as e:
            print(f"[frontend] Postgres checkpointer unavailable ({e}); using in-memory instead.")
    if checkpointer is None:
        checkpointer = MemorySaver()

    compiled = graph.compile(checkpointer=checkpointer)
    return compiled, using_postgres


def run_agent(query: str, thread_id: str) -> str:
    app, _ = build_pipeline()
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "user_query": query,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0,
        },
        config=config,
    )
    return result["messages"][-1].content


# ---------------------------------------------------------------------------
# Design tokens / CSS
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #090D1A;
    --surface: #121A2E;
    --surface-alt: #182140;
    --gold: #E8A33D;
    --sky: #5EC8D8;
    --cream: #F3EEE2;
    --muted: #7C89A8;
    --border: #232F4D;
    --danger: #E0645A;
}

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(94,200,216,0.07), transparent 40%),
        radial-gradient(circle at 85% 100%, rgba(232,163,61,0.06), transparent 45%),
        var(--bg);
    color: var(--cream);
}

#MainMenu, header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }

section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--cream); }

.board-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.18em;
    color: var(--gold);
    text-transform: uppercase;
    margin: 18px 0 10px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.board-label::after { content: ""; flex: 1; height: 1px; background: var(--border); }

.status-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    padding: 6px 0;
    color: var(--muted);
}
.dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; margin-right: 8px; }
.dot-on { background: #58C27D; box-shadow: 0 0 6px #58C27D; }
.dot-off { background: var(--danger); box-shadow: 0 0 6px var(--danger); }

section[data-testid="stSidebar"] .stButton>button {
    width: 100%;
    text-align: left;
    background: var(--surface-alt);
    border: 1px solid var(--border);
    color: var(--cream);
    font-family: 'JetBrains Mono', monospace;
    font-size: 12.5px;
    padding: 10px 12px;
    border-radius: 6px;
    margin-bottom: 8px;
    transition: border-color 0.15s ease, transform 0.15s ease;
}
section[data-testid="stSidebar"] .stButton>button:hover {
    border-color: var(--gold);
    color: var(--gold);
    transform: translateX(2px);
}

.hero-wrap { padding: 6px 0 18px 0; border-bottom: 1px solid var(--border); margin-bottom: 22px; }
.hero-tag {
    font-family: 'JetBrains Mono', monospace;
    color: var(--sky);
    font-size: 12px;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    margin-top: 6px;
}

.chat-scroll { padding-bottom: 8px; }
.msg-row { display: flex; margin: 14px 0; }
.msg-row.user { justify-content: flex-end; }
.msg-row.assistant { justify-content: flex-start; }

.bubble { max-width: 72%; padding: 14px 16px; border-radius: 10px; font-size: 14.5px; line-height: 1.55; position: relative; }
.bubble.user { background: var(--gold); color: #1B1305; border-radius: 10px 10px 2px 10px; font-weight: 500; }
.bubble.assistant { background: var(--surface); border: 1px solid var(--border); border-radius: 2px 10px 10px 10px; color: var(--cream); }
.bubble.assistant::before {
    content: ""; position: absolute; left: -1px; top: 14px; width: 10px; height: 10px;
    background: var(--bg); border-left: 1px dashed var(--border); border-radius: 50%; transform: translateX(-50%);
}
.bubble .tag {
    font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 0.15em;
    text-transform: uppercase; opacity: 0.55; margin-bottom: 6px; display: block;
}

.empty-state { text-align: center; padding: 60px 20px; color: var(--muted); }
.empty-state h2 { font-family: 'Space Mono', monospace; color: var(--cream); font-size: 22px; margin-bottom: 8px; }
.empty-state p { font-size: 13.5px; }

div[data-testid="stChatInput"] { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; }
div[data-testid="stChatInput"] textarea { color: var(--cream) !important; font-family: 'Plus Jakarta Sans', sans-serif !important; }

::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

SPLIT_FLAP_HTML = """
<div style="font-family:'JetBrains Mono',monospace;">
  <style>
    body { margin:0; background:transparent; }
    .flap-board { display:flex; gap:6px; flex-wrap:wrap; }
    .flap {
      width:34px; height:44px; background:#182140; border:1px solid #2A3760;
      border-radius:4px; display:flex; align-items:center; justify-content:center;
      font-family:'Space Mono',monospace; font-size:24px; font-weight:700;
      color:#E8A33D; position:relative; overflow:hidden;
    }
    .flap.space { background:transparent; border:none; width:14px; }
  </style>
  <div class="flap-board" id="board"></div>
  <script>
    const text = "VOYAGER";
    const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    const board = document.getElementById("board");
    const cells = [];
    for (const ch of text) {
      const el = document.createElement("div");
      el.className = ch === " " ? "flap space" : "flap";
      el.textContent = "";
      board.appendChild(el);
      cells.push({el, target: ch});
    }
    cells.forEach((c, i) => {
      if (c.target === " ") return;
      let ticks = 8 + i * 2;
      const interval = setInterval(() => {
        c.el.textContent = chars[Math.floor(Math.random() * chars.length)];
        ticks--;
        if (ticks <= 0) {
          clearInterval(interval);
          c.el.textContent = c.target;
        }
      }, 45);
    });
  </script>
</div>
"""

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div style='font-family:Space Mono,monospace;font-size:18px;color:#E8A33D;"
        "letter-spacing:0.05em;'>✈ VOYAGER</div>"
        "<div style='font-family:JetBrains Mono,monospace;font-size:10.5px;color:#7C89A8;"
        "letter-spacing:0.1em;margin-top:2px;'>AI TRAVEL PLANNING AGENT</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="board-label">System Status</div>', unsafe_allow_html=True)
    checks = [
        ("Groq · Llama 3.3 70B", bool(_get_key("GROQ_API_KEY"))),
        ("Tavily Search", bool(_get_key("TAVILY_API_KEY"))),
        ("AviationStack", bool(_get_key("AVIATIONSTACK_API_KEY"))),
    ]
    for label, ok in checks:
        dot = "dot-on" if ok else "dot-off"
        state = "READY" if ok else "MISSING KEY"
        st.markdown(
            f'<div class="status-row"><span><span class="dot {dot}"></span>{label}</span>'
            f'<span>{state}</span></div>',
            unsafe_allow_html=True,
        )

    try:
        _, using_postgres = build_pipeline()
    except Exception:
        using_postgres = False
    mem_dot = "dot-on" if using_postgres else "dot-off"
    mem_label = "Postgres (persistent)" if using_postgres else "In-memory (session only)"
    st.markdown(
        f'<div class="status-row"><span><span class="dot {mem_dot}"></span>Conversation memory</span>'
        f'<span>{mem_label}</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="board-label">Quick Queries</div>', unsafe_allow_html=True)
    quick_prompts = [
        "✈  Flights from Delhi to Dubai this week",
        "🏨  Best budget hotels in Bali for a week",
        "🌤  Weather in Bangkok next month",
        "📄  Visa requirements for Thailand (India)",
    ]
    picked_prompt = None
    for qp in quick_prompts:
        if st.button(qp, key=qp):
            picked_prompt = qp.split("  ", 1)[1]

    st.markdown('<div class="board-label">Session</div>', unsafe_allow_html=True)
    if st.button("🗑  Clear conversation", key="clear"):
        st.session_state.messages = []
        st.session_state.thread_id = f"user_{uuid.uuid4().hex[:12]}"
        st.rerun()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="hero-wrap">', unsafe_allow_html=True)
components.html(SPLIT_FLAP_HTML, height=60)
st.markdown(
    '<div class="hero-tag">Gate open · flights · stays · itineraries · one chat</div>',
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"user_{uuid.uuid4().hex[:12]}"


def render_message(role: str, content: str):
    tag = "You" if role == "user" else "Voyager"
    st.markdown(
        f'<div class="msg-row {role}"><div class="bubble {role}">'
        f'<span class="tag">{tag}</span>{content}</div></div>',
        unsafe_allow_html=True,
    )


st.markdown('<div class="chat-scroll">', unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-state">
            <svg width="180" height="90" viewBox="0 0 180 90" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-bottom:14px;">
                <path d="M10 70 Q90 10 170 70" stroke="#232F4D" stroke-width="1.5" stroke-dasharray="4 5" fill="none"/>
                <circle cx="10" cy="70" r="3.5" fill="#5EC8D8"/>
                <circle cx="170" cy="70" r="3.5" fill="#E8A33D"/>
                <g transform="translate(84,28) rotate(35)">
                    <path d="M0 8 L18 0 L20 3 L8 9 L10 15 L15 17 L15 19 L8 18 L6 22 L3 22 L4 17 L0 15 Z" fill="#E8A33D"/>
                </g>
            </svg>
            <h2>Where are we flying today?</h2>
            <p>Tell me your trip — I'll pull flights, find hotels, and build a full itinerary in one go.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    for m in st.session_state.messages:
        render_message(m["role"], m["content"])

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------
user_query = st.chat_input("Ask about flights, hotels, weather, visas…")
query = picked_prompt or user_query

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.spinner("Fetching flights, hotels, and building your itinerary…"):
        try:
            reply = run_agent(query, thread_id=st.session_state.thread_id)
        except Exception as e:
            reply = (
                "Something went wrong reaching the agent. "
                f"Details: `{e}`\n\nCheck that your API keys are set "
                "(GROQ_API_KEY, TAVILY_API_KEY, AVIATIONSTACK_API_KEY)."
            )
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()
