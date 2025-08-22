import os, sys, requests, streamlit as st
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_URL

st.set_page_config(page_title="Medical Chatbot (RAG)", page_icon="🧠", layout="wide")
st.title("🧠 Medical Chatbot (RAG)")
st.write("💬 Ask medical questions answered using the knowledge base with citations.")

# Auth
if "HF_TOKEN" not in st.session_state or not st.session_state["HF_TOKEN"]:
    st.warning("Please enter your Hugging Face token on the Home page sidebar.")
    st.stop()

API_BASE = BASE_URL.rstrip("/")
HEADERS  = {"Authorization": f"Bearer {st.session_state['HF_TOKEN']}"}

# Sidebar
st.sidebar.header("Chat Settings")
show_refs = st.sidebar.checkbox("Show citations", value=True)
examples = [
    "what is Antigens tell in 100 words?",
    "Tell about Cirrhosis disease?",
    "Tell about Ultrasound?",
]
st.sidebar.markdown("**Try an example:**")
if st.sidebar.button("Antigens"):  st.session_state["rag_prefill"] = examples[0]
if st.sidebar.button("Cirrhosis"): st.session_state["rag_prefill"] = examples[1]
if st.sidebar.button("Ultrasound"):  st.session_state["rag_prefill"] = examples[2]

# State
if "rag_messages" not in st.session_state:
    st.session_state["rag_messages"] = []
if "rag_last_sent" not in st.session_state:
    st.session_state["rag_last_sent"] = None

def add_message(role, content, refs=None):
    st.session_state["rag_messages"].append({"role": role, "content": content, "refs": refs})

def call_backend(q: str):
    r = requests.get(f"{API_BASE}/query", params={"q": q}, headers=HEADERS, timeout=60)
    if not r.ok:
        raise RuntimeError(f"{r.status_code}: {r.text}")
    return r.json()

# -------- 1) INPUT: handle send FIRST --------
pending_prefill = st.session_state.pop("rag_prefill", None)
prompt = st.chat_input("Ask a medical question…", key="rag_input")
if prompt is None and pending_prefill:
    prompt = pending_prefill

if prompt and prompt != st.session_state["rag_last_sent"]:
    st.session_state["rag_last_sent"] = prompt

    # store user message so it renders this run
    add_message("user", prompt)

    # call backend and store assistant message
    with st.spinner("Thinking…"):
        try:
            data   = call_backend(prompt)
            answer = (data.get("answer") or "").strip() or "_No answer returned_"
            refs   = data.get("references") or []
            add_message("assistant", answer, refs=refs)
        except Exception as e:
            add_message("assistant", f"❌ Backend error: {e}")

# -------- 2) RENDER: full history (now includes the just-added messages) --------
for m in st.session_state["rag_messages"]:
    with st.chat_message("user" if m["role"] == "user" else "assistant"):
        st.markdown(m["content"])
        if m["role"] == "assistant" and show_refs and m.get("refs"):
            with st.expander("📚 References"):
                for ref in m["refs"]:
                    page = (ref or {}).get("page", "N/A")
                    link = (ref or {}).get("link") or (ref or {}).get("url") or "#"
                    st.markdown(f"- Page **{page}** — [{link}]({link})")

# Actions
c1, c2, _ = st.columns([1,1,4])
with c1:
    if st.button("🧹 Clear Chat"):
        st.session_state["rag_messages"] = []
        st.session_state["rag_last_sent"] = None
        st.rerun()
with c2:
    if st.button("🔁 Re-run last"):
        last_user = next((m for m in reversed(st.session_state["rag_messages"]) if m["role"] == "user"), None)
        if last_user:
            st.session_state["rag_prefill"] = last_user["content"]
            st.session_state["rag_last_sent"] = None
            st.rerun()