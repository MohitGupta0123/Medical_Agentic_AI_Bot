import os
import sys
import requests
import streamlit as st

# Local import for BASE_URL
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_URL

st.set_page_config(page_title="Agent Bot", page_icon="🤖", layout="wide")
st.markdown("## 🤖 Agent Bot – Orchestrated Assistant")
st.write("Ask free-form questions and let the intelligent assistant decide the right action.")

# ---- Auth guard ----
if "HF_TOKEN" not in st.session_state or not st.session_state["HF_TOKEN"]:
    st.warning("Please enter your Hugging Face token in the sidebar on the Home page.")
    st.stop()

API_BASE = BASE_URL.rstrip("/")
HEADERS = {"Authorization": f"Bearer {st.session_state['HF_TOKEN']}"}

# ---- API helper ----
def call_orchestrator(q: str, timeout: int = 90) -> dict:
    url = f"{API_BASE}/orchestrator_query"
    r = requests.get(url, params={"q": q}, headers=HEADERS, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"{r.status_code}: {r.text}")
    return r.json()

# ---- Examples in sidebar (prefill once) ----
st.sidebar.subheader("Examples")
examples = [
    "Register patient John, age 35, reason: chest pain",
    "Check appointment for Vikas",
    "Is Aspirin available?",
    "Summarize case for patient id 3",
    "What is atherosclerosis?",
]
for i, ex in enumerate(examples, 1):
    if st.sidebar.button(f"Example {i}"):
        st.session_state["agent_prefill"] = ex

# ---- Chat state ----
if "agent_messages" not in st.session_state:
    st.session_state["agent_messages"] = []  # [{role, content, payload}]
if "agent_last_sent" not in st.session_state:
    st.session_state["agent_last_sent"] = None  # prevents double-send on rerun

def add_msg(role: str, content: str, payload=None):
    if payload is not None and not isinstance(payload, dict):
        payload = {"raw": payload}
    st.session_state["agent_messages"].append({"role": role, "content": content, "payload": payload})

# ---- Pretty renderer for structured payloads (uses payload['type']) ----
def _render_structured(payload):
    if not isinstance(payload, dict):
        return
    t = payload.get("type")

    # --- RAG ---
    if t == "rag":
        st.markdown("### 🧠 RAG Answer")
        if isinstance(payload.get("answer"), str) and payload["answer"].strip():
            st.write(payload["answer"])
        refs = payload.get("references") or []
        if refs:
            with st.expander("📚 References"):
                for ref in refs:
                    page = (ref or {}).get("page", "N/A")
                    link = (ref or {}).get("link") or (ref or {}).get("url") or "#"
                    st.markdown(f"- Page **{page}** — [{link}]({link})")
        return

    # --- Register patient ---
    if t == "register_patient":
        d = payload.get("assigned_doctor") or {}
        st.markdown("### 🩺 Assigned Doctor")
        st.markdown(f"**👨‍⚕️ Name:** {d.get('name','N/A')}")
        st.markdown(f"**🧪 Specialization:** {d.get('specialization','N/A')}")
        if payload.get("reasoning"):
            with st.expander("🧠 Model Reasoning"):
                st.write(payload["reasoning"])
        return

    # --- Confirm appointment ---
    if t == "confirm_appointment":
        doc, spec = payload.get("doctor_name"), payload.get("specialization")
        if doc and spec:
            st.success(f"📅 Appointment with **{doc}** (*{spec}*) confirmed.")
        elif payload.get("message"):
            st.info(payload["message"])
        return

    # --- Medicine availability ---
    if t == "medicine_availability":
        if payload.get("medicine_message"):
            st.info(payload["medicine_message"])
        elif payload.get("message"):
            st.info(payload["message"])
        return

    # --- Summarize case ---
    if t == "summarize_case":
        if isinstance(payload.get("summary"), str) and payload["summary"].strip():
            st.markdown("### 📝 Case Summary")
            st.write(payload["summary"])
            st.code(payload["summary"])
            st.download_button(
                "⬇️ Download Summary",
                data=payload["summary"].encode("utf-8"),
                file_name="case_summary.txt",
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.info(payload.get("message", "No summary returned."))
        return

    # --- Fallbacks / legacy keys ---
    if payload.get("redirect_to_rag") and payload.get("rag_url"):
        st.info(f"🔁 This looks like a RAG query. Open: {payload['rag_url']}")
    if "assigned_doctor" in payload:
        d = payload["assigned_doctor"] or {}
        st.markdown("### 🩺 Assigned Doctor")
        st.markdown(f"**👨‍⚕️ Name:** {d.get('name','N/A')}")
        st.markdown(f"**🧪 Specialization:** {d.get('specialization','N/A')}")

# -------------------------------
# 1) INPUT: handle send FIRST
# -------------------------------
pending_prefill = st.session_state.pop("agent_prefill", None)
user_q = st.chat_input(
    "Type your instruction (e.g., 'Register patient John', 'Is Aspirin available?')",
    key="agent_input"
)
if user_q is None and pending_prefill:
    user_q = pending_prefill

if user_q and user_q != st.session_state["agent_last_sent"]:
    st.session_state["agent_last_sent"] = user_q

    # store user message immediately so it renders this run
    add_msg("user", user_q)

    # call backend and store assistant message
    with st.spinner("Thinking…"):
        try:
            data = call_orchestrator(user_q)          # {"result": {...}, "references": [...]}
            result = data.get("result", {})

            if isinstance(result, dict):
                # pick a nice bubble message (type-aware)
                t = result.get("type")
                bubble = (
                    result.get("message")
                    or ((result.get("answer")[:200] + "…")
                        if t == "rag" and isinstance(result.get("answer"), str) and result.get("answer") else None)
                    or result.get("medicine_message")
                    or ("Summary ready." if t == "summarize_case" and result.get("summary") else None)
                    or "Done."
                )
                add_msg("assistant", bubble, payload=result)
            else:
                text_answer = str(result) if result is not None else "_No result returned_"
                add_msg("assistant", text_answer)
        except Exception as e:
            add_msg("assistant", f"❌ {e}")

# -------------------------------
# 2) RENDER: show full history
# -------------------------------
for m in st.session_state["agent_messages"]:
    with st.chat_message("user" if m["role"] == "user" else "assistant"):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            _render_structured(m.get("payload"))
            if isinstance(m.get("payload"), dict):
                with st.expander("🔎 Raw result"):
                    st.json(m["payload"])

# ---- Utilities row ----
c1, c2 = st.columns([1,1])
with c1:
    if st.button("🧹 Clear Chat"):
        st.session_state["agent_messages"] = []
        st.session_state["agent_last_sent"] = None
        st.rerun()
with c2:
    if st.button("🔁 Re-run last"):
        last_user = next((m for m in reversed(st.session_state["agent_messages"]) if m["role"] == "user"), None)
        if last_user:
            st.session_state["agent_prefill"] = last_user["content"]
            st.session_state["agent_last_sent"] = None  # allow resend
            st.rerun()
