import os
import requests
import streamlit as st

st.set_page_config(page_title="Medical Assistant", page_icon="💊", layout="wide")

# -------------------------------
# Sidebar: Auth + Backend config
# -------------------------------
st.sidebar.header("🔐 Authentication")

# Prefer an env var default if you deploy multiple envs
default_base = os.getenv("BASE_URL", "https://mohitg012-medical-bot-agentic-ai.hf.space")
BASE_URL = st.sidebar.text_input("Backend URL", value=default_base, help="Your FastAPI Space URL")

hf_token = st.sidebar.text_input("Hugging Face Token", type="password", placeholder="hf_xxx...")
remember = st.sidebar.checkbox("Remember token for this browser", value=False)

if hf_token:
    st.session_state["HF_TOKEN"] = hf_token
    if remember:
        # Warning: session_state is per-session; for true persistence, use st.session_state + st.experimental_set_query_params or a cookie/lib
        st.session_state["REMEMBER"] = True
    st.sidebar.success("Token saved.")
else:
    st.sidebar.info("Enter your HF token to enable authenticated calls.")

# -------------------------------
# Header
# -------------------------------
left, right = st.columns([0.8, 0.2])
with left:
    st.title("💊 Medical Assistant Portal")
    st.write("🏠 Home – Enter your Hugging Face token and navigate to the Medical Assistant tools.")
    st.write("RAG answers, agent workflows, and admin dashboard all in one place.")
with right:
    st.image("https://img.icons8.com/color/96/medical-doctor.png", use_container_width=True)

st.markdown("---")
# -------------------------------
# Backend health check
# -------------------------------
st.markdown("### 🩺 Backend Status")

def check_backend(base_url: str, token: str | None):
    try:
        r = requests.get(f"{base_url.rstrip('/')}/", timeout=10)
        ok_root = r.ok
        r2 = requests.get(f"{base_url.rstrip('/')}/docs", timeout=10)
        ok_docs = r2.ok
        auth_ok = None
        if token:
            # Small authenticated call that doesn't mutate state
            rq = requests.get(
                f"{base_url.rstrip('/')}/medicine_availability",
                params={"name": "Paracetamol"},
                headers={"Authorization": f"Bearer {token}"},
                timeout=15
            )
            auth_ok = rq.ok
        return ok_root, ok_docs, auth_ok, r.json() if ok_root else None
    except Exception as e:
        return False, False, False, {"error": str(e)}

col1, col2, col3 = st.columns(3)
ok_root, ok_docs, auth_ok, root_payload = check_backend(BASE_URL, st.session_state.get("HF_TOKEN"))

with col1:
    st.metric("Root", "OK" if ok_root else "Down")
with col2:
    st.metric("Docs", "OK" if ok_docs else "Down")
with col3:
    st.metric("Auth Test", ("OK" if auth_ok else ("Needs Token" if st.session_state.get("HF_TOKEN") is None else "Failed")))

if ok_root and isinstance(root_payload, dict):
    with st.expander("ℹ️ Server Info"):
        st.json(root_payload)

st.markdown("---")

# -------------------------------
# Navigation (one-click)
# -------------------------------

st.markdown("### 🧭 Navigate")
nav1, nav2, nav3, nav4 = st.columns(4)
with nav1:
    st.page_link("pages/1_Medical_Chatbot.py", label="Medical Chatbot", icon="🧠")
with nav2:
    st.page_link("pages/2_Registration_And_Operations.py", label="Registration And Operations", icon="📝")
with nav3:
    st.page_link("pages/3_Agent_Bot.py", label="Agent Bot", icon="🤖")
with nav4:
    st.page_link("pages/4_Dashboard.py", label="Dashboard", icon="📊")

st.markdown("---")

# -------------------------------
# Quick Actions (optional)
# -------------------------------
st.markdown("### ⚡ Quick Actions")
qa1, qa2 = st.columns(2)

with qa1:
    st.subheader("Ask a sample question")
    sample_q = st.text_input("Try a quick RAG query", value="What are common symptoms of jaundice?")
    if st.button("Run sample RAG"):
        if not st.session_state.get("HF_TOKEN"):
            st.warning("Enter your HF token first (left sidebar).")
        else:
            with st.spinner("Contacting backend…"):
                try:
                    r = requests.get(
                        f"{BASE_URL.rstrip('/')}/query",
                        params={"q": sample_q},
                        headers={"Authorization": f"Bearer {st.session_state['HF_TOKEN']}"},
                        timeout=30
                    )
                    if r.ok:
                        st.success("RAG response:")
                        st.json(r.json())
                    else:
                        st.error(f"{r.status_code}: {r.text}")
                except Exception as e:
                    st.error(f"Error: {e}")

with qa2:
    st.subheader("Release stale doctors")
    st.caption("Frees any bookings older than 20 minutes.")
    if st.button("Release"):
        if not st.session_state.get("HF_TOKEN"):
            st.warning("Enter your HF token first.")
        else:
            try:
                r = requests.post(
                    f"{BASE_URL.rstrip('/')}/release_stale_doctors",
                    headers={"Authorization": f"Bearer {st.session_state['HF_TOKEN']}"},
                    timeout=20
                )
                st.success(r.json() if r.ok else f"Failed: {r.status_code} {r.text}")
            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")

# -------------------------------
# Help / Tips
# -------------------------------
with st.expander("🧩 How this works", expanded=True):
    st.markdown(
        """
- **Backend URL**: Your FastAPI service (on Spaces) that exposes `/query`, `/orchestrator_query`, etc.  
- **HF Token**: Used to call HF models from the backend. We pass it in the `Authorization` header.
- **Pages**:
    - *Medical Chatbot*: RAG QA over your medical corpus.
    - *Agent Bot*: Patient registration, appointment confirmation, medicines, summaries.
    - *Dashboard*: Live tables & charts via backend admin endpoints.
- **Troubleshooting**:
    - If **Docs** is down, check your backend is running and exporting `/docs`.
    - If **Auth Test** fails, verify your token and CORS.
"""
    )

st.caption("© 2025 Medical Assistant • Built with Streamlit + FastAPI - By Mohit Gupta")