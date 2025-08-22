import os
import sys
import requests
from datetime import datetime, timezone
import streamlit as st

# Local import for BASE_URL
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_URL

st.set_page_config(page_title="Registration & Ops", page_icon="🩺", layout="wide")
st.markdown("## 📝 Registration & Operations")
st.write("📚 Register patients, check appointments, view medicine availability, and summarize cases.")
# ---- Auth guard ----
if "HF_TOKEN" not in st.session_state or not st.session_state["HF_TOKEN"]:
    st.warning("Please enter your Hugging Face token in the sidebar on the Home page.")
    st.stop()

API_BASE = BASE_URL.rstrip("/")
HEADERS = {"Authorization": f"Bearer {st.session_state['HF_TOKEN']}"}

# ---- Small API helper ----
def api_call(method: str, path: str, *, params=None, json=None, timeout=40):
    url = f"{API_BASE}/{path.lstrip('/')}"
    try:
        r = requests.request(method, url, headers=HEADERS, params=params, json=json, timeout=timeout)
        if not r.ok:
            raise RuntimeError(f"{r.status_code}: {r.text}")
        try:
            return r.json()
        except Exception:
            return {"_raw": r.text}
    except Exception as e:
        raise RuntimeError(f"Request to {path} failed: {e}")

# ---- Tabs ----
tab = st.radio(
    "Select Action",
    ["Register Patient", "Check Appointment", "Medicine Availability", "Summarize Case"],
    horizontal=True
)
st.markdown("---")

# =====================================================================================
# 1) Register Patient
# =====================================================================================
if tab == "Register Patient":
    st.caption("Assigns a suitable doctor (Gemma-powered) and creates a patient record.")

    # Quick examples OUTSIDE the form (allowed)
    st.markdown("**Quick Examples:**")
    ex1, ex2, ex3 = st.columns(3)
    if ex1.button("Fever example"):
        st.session_state["example_reason"] = "High fever and body ache for 2 days"
    if ex2.button("Jaundice example"):
        st.session_state["example_reason"] = "Yellowing of eyes and dark urine in my child"
    if ex3.button("Chest pain example"):
        st.session_state["example_reason"] = "Chest pain on exertion"

    with st.form("register_patient_form", clear_on_submit=False):
        c1, c2 = st.columns([2, 1])
        with c1:
            name = st.text_input("🧑 Patient Name", placeholder="e.g., Vikas Gupta")
        with c2:
            age = st.number_input("🎂 Age", min_value=1, max_value=120, value=30, step=1)

        reason = st.text_area(
            "📋 Reason for Visit",
            value=st.session_state.get("example_reason", ""),
            placeholder="e.g., My 2-year-old has jaundice…"
        )

        submit = st.form_submit_button("📝 Register", use_container_width=True)
        if submit:
            if not name.strip() or not reason.strip():
                st.warning("Name and reason cannot be empty.")
            else:
                with st.spinner("Registering and assigning a doctor…"):
                    try:
                        data = api_call(
                            "POST", "/register_patient",
                            json={"name": name, "age": int(age), "reason": reason}
                        )
                        st.success("✅ Patient Registered!")
                        st.markdown(
                            f"""
                            **Patient ID:** `{data.get('patient_id')}`  
                            **Assigned Doctor:** **{data.get('assigned_doctor', {}).get('name','N/A')}**  
                            **Specialization:** {data.get('assigned_doctor', {}).get('specialization','N/A')}
                            """
                        )
                        if data.get("reasoning"):
                            with st.expander("🧠 Model Reasoning"):
                                st.write(data["reasoning"])
                    except Exception as e:
                        st.error(str(e))

# =====================================================================================
# 2) Check Appointment
# =====================================================================================
elif tab == "Check Appointment":
    st.caption("Checks registration and confirms appointment with assigned doctor (books if available).")

    with st.form("check_appt_form"):
        name = st.text_input("🧑 Patient Name", placeholder="Exact name used during registration")
        submit = st.form_submit_button("📅 Check Appointment", use_container_width=True)

        if submit:
            if not name.strip():
                st.warning("Patient name cannot be empty.")
            else:
                with st.spinner("Fetching appointment…"):
                    try:
                        data = api_call("POST", "/check_registration_status", json={"name": name})
                        if "message" in data:
                            st.success(data["message"])

                        if "doctor_name" in data:
                            st.markdown("### 🩺 Doctor Info")
                            st.markdown(f"**👨‍⚕️ Doctor:** {data['doctor_name']}")
                            st.markdown(f"**🧪 Specialization:** {data.get('specialization', 'N/A')}")

                        # elapsed time if backend includes last_booked_at later
                        last = data.get("last_booked_at")
                        if last:
                            try:
                                dt = datetime.fromisoformat(last)
                                if dt.tzinfo is None:
                                    dt = dt.replace(tzinfo=timezone.utc)
                                mins = round((datetime.now(timezone.utc) - dt).total_seconds() / 60)
                                st.caption(f"⏳ Booked ~ {mins} minutes ago")
                            except Exception:
                                pass
                    except Exception as e:
                        st.error(str(e))

# =====================================================================================
# 3) Medicine Availability
# =====================================================================================
elif tab == "Medicine Availability":
    st.caption("Search is case-insensitive and supports partial names.")

    # Quick picks OUTSIDE the form
    st.markdown("**Quick Picks:**")
    mq1, mq2, mq3, mq4 = st.columns(4)
    if mq1.button("Paracetamol"):
        st.session_state["last_med_prefill"] = "Paracetamol"
    if mq2.button("Ibuprofen"):
        st.session_state["last_med_prefill"] = "Ibuprofen"
    if mq3.button("Amlodipine"):
        st.session_state["last_med_prefill"] = "Amlodipine"
    if mq4.button("Omeprazole"):
        st.session_state["last_med_prefill"] = "Omeprazole"

    with st.form("med_avail_form"):
        med_name = st.text_input(
            "💊 Medicine Name",
            value=st.session_state.get("last_med_prefill", ""),
            placeholder="e.g., paracetamol, metoprolol"
        )
        submit = st.form_submit_button("🔎 Check Availability", use_container_width=True)

        if submit:
            if not med_name.strip():
                st.warning("Medicine name cannot be empty.")
            else:
                with st.spinner("Checking stock…"):
                    try:
                        data = api_call("GET", "/medicine_availability", params={"name": med_name})
                        st.info(f"ℹ️ {data.get('message','No message')}")
                    except Exception as e:
                        st.error(str(e))

# =====================================================================================
# 4) Summarize Case
# =====================================================================================
elif tab == "Summarize Case":
    st.caption("Generates a concise summary of the patient’s case (uses your LLM).")

    # keep last result in session so we can render it outside the form
    if "last_summary" not in st.session_state:
        st.session_state["last_summary"] = None
        st.session_state["last_summary_pid"] = None

    with st.form("summarize_form"):
        patient_id = st.number_input("🔢 Patient ID", min_value=1, step=1)
        submit = st.form_submit_button("🧠 Summarize Case", use_container_width=True)

        if submit:
            with st.spinner("Summarizing…"):
                try:
                    data = api_call("GET", f"/summarize_case/{int(patient_id)}")
                    summary = (data.get("summary") or "").strip()
                    if summary:
                        st.session_state["last_summary"] = summary
                        st.session_state["last_summary_pid"] = int(patient_id)
                    else:
                        st.session_state["last_summary"] = None
                        st.session_state["last_summary_pid"] = None
                        st.warning("No summary returned.")
                except Exception as e:
                    st.session_state["last_summary"] = None
                    st.session_state["last_summary_pid"] = None
                    st.error(str(e))

    # 🔽 Render result OUTSIDE the form (download_button allowed here)
    if st.session_state.get("last_summary"):
        pid = st.session_state.get("last_summary_pid")
        summary = st.session_state["last_summary"]

        st.markdown("### 📝 Case Summary")
        st.write(summary)
        st.code(summary)
        st.download_button(
            "⬇️ Download Summary",
            data=summary.encode("utf-8"),
            file_name=f"patient_{pid}_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )
        # optional: clear button
        if st.button("Clear Summary"):
            st.session_state["last_summary"] = None
            st.session_state["last_summary_pid"] = None
            st.rerun()