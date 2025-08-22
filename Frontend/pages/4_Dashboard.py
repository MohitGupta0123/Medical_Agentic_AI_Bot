import os
import sys
import io
import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta, timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_URL

st.set_page_config(page_title="Medical Assistant Dashboard", page_icon="📊", layout="wide")
st.title("📊 Medical Assistant Dashboard")
st.write("View real-time stats, charts, and detailed tables for patients, doctors, and medicines.")

# -------- Auth guard --------
if "HF_TOKEN" not in st.session_state or not st.session_state["HF_TOKEN"]:
    st.warning("Please enter your Hugging Face token on the Home page.")
    st.stop()

HEADERS = {"Authorization": f"Bearer {st.session_state['HF_TOKEN']}"}
API = BASE_URL.rstrip("/")

@st.cache_data(ttl=30)
def fetch_table(path: str) -> pd.DataFrame:
    url = f"{API}/{path.lstrip('/')}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    if not r.ok:
        raise RuntimeError(f"GET {path} failed: {r.status_code} {r.text}")
    data = r.json()
    rows = data.get("items", data)
    return pd.DataFrame(rows)

def load_data():
    patients = fetch_table("/admin/patients")
    doctors = fetch_table("/admin/doctors")
    medicines = fetch_table("/admin/medicines")
    return patients, doctors, medicines

def to_csv_download(df: pd.DataFrame, filename: str, label: str):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button(
        f"⬇️ Download {label} CSV",
        data=buf.getvalue().encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=True
    )

# -------- Top actions --------
left, mid, right = st.columns([1,1,4])
with left:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
with mid:
    if st.button("🧹 Release Stale Doctors"):
        try:
            r = requests.post(f"{API}/release_stale_doctors", headers=HEADERS, timeout=30)
            if r.ok:
                st.success("Released stale bookings.")
                st.cache_data.clear(); st.rerun()
            else:
                st.error(f"Failed: {r.status_code} {r.text}")
        except Exception as e:
            st.error(f"Error: {e}")

# -------- Load --------
try:
    patients_df, doctors_df, medicines_df = load_data()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# Ensure columns exist
for df, cols in [
    (patients_df, ["id","name","age","reason","registered_at","doctor_id"]),
    (doctors_df, ["id","name","specialization","available","last_booked_at"]),
    (medicines_df, ["id","name","stock"]),
]:
    for c in cols:
        if c not in df.columns:
            df[c] = pd.Series(dtype="object")

# Normalize types
if not patients_df.empty:
    # parse registered_at to datetime (UTC fallback)
    def _parse_dt(x):
        try:
            dt = pd.to_datetime(x, utc=True)
            return dt
        except Exception:
            return pd.NaT
    patients_df["registered_at_dt"] = patients_df["registered_at"].apply(_parse_dt)

if not doctors_df.empty:
    doctors_df["available"] = pd.to_numeric(doctors_df["available"], errors="coerce").fillna(0).astype(int)

if not medicines_df.empty:
    medicines_df["stock"] = pd.to_numeric(medicines_df["stock"], errors="coerce").fillna(0).astype(int)

# -------- Sidebar filters --------
st.sidebar.header("🔎 Filters")

# Patients filters
st.sidebar.subheader("Patients")
name_query = st.sidebar.text_input("Search name contains")
reason_query = st.sidebar.text_input("Search reason contains")
date_range = st.sidebar.selectbox("Date range", ["All", "Last 24h", "Last 7 days", "Last 30 days"])
start_dt = None
if date_range != "All" and not patients_df.empty:
    now = datetime.now(timezone.utc)
    delta = {"Last 24h": 1, "Last 7 days": 7, "Last 30 days": 30}[date_range]
    start_dt = now - timedelta(days=delta)

# Doctors filters
st.sidebar.subheader("Doctors")
spec_options = sorted(doctors_df["specialization"].dropna().unique().tolist()) if not doctors_df.empty else []
spec_sel = st.sidebar.multiselect("Specializations", spec_options)
availability = st.sidebar.selectbox("Availability", ["All", "Available", "Unavailable"])

# Medicines filters
st.sidebar.subheader("Medicines")
stock_filter = st.sidebar.selectbox("Stock filter", ["All", "In stock (>0)", "Low stock (<50)", "Out of stock (0)"])

# -------- Apply filters --------
p_df = patients_df.copy()
if not p_df.empty:
    if name_query:
        p_df = p_df[p_df["name"].str.contains(name_query, case=False, na=False)]
    if reason_query:
        p_df = p_df[p_df["reason"].str.contains(reason_query, case=False, na=False)]
    if start_dt is not None:
        p_df = p_df[p_df["registered_at_dt"] >= start_dt]
    p_df = p_df.sort_values(by="registered_at_dt", ascending=False, na_position="last")

d_df = doctors_df.copy()
if not d_df.empty:
    if spec_sel:
        d_df = d_df[d_df["specialization"].isin(spec_sel)]
    if availability == "Available":
        d_df = d_df[d_df["available"] == 1]
    elif availability == "Unavailable":
        d_df = d_df[d_df["available"] == 0]

m_df = medicines_df.copy()
if not m_df.empty:
    if stock_filter == "In stock (>0)":
        m_df = m_df[m_df["stock"] > 0]
    elif stock_filter == "Low stock (<50)":
        m_df = m_df[m_df["stock"] < 50]
    elif stock_filter == "Out of stock (0)":
        m_df = m_df[m_df["stock"] == 0]

st.markdown("---")

# -------- KPIs --------
st.markdown("### Key Metrics")
k1, k2, k3, k4 = st.columns(4)
with k1: st.metric("Total Patients", int(len(patients_df)))
with k2: st.metric("Total Doctors", int(len(doctors_df)))
with k3:
    available_doctors = int((doctors_df["available"] == 1).sum()) if not doctors_df.empty else 0
    st.metric("Available Doctors", available_doctors)
with k4:
    total_meds_available = int((medicines_df["stock"] > 0).sum()) if not medicines_df.empty else 0
    st.metric("Medicines > 0 stock", total_meds_available)

st.markdown("---")

# -------- Patients --------
st.markdown("## 🧑‍⚕️ Registered Patients")
st.dataframe(
    p_df.drop(columns=["registered_at_dt"], errors="ignore"),
    use_container_width=True,
    # height=min(400, 40 + 28 * min(len(p_df), 10))
)
cpa1, cpa2 = st.columns(2)
with cpa1:
    if not p_df.empty:
        st.markdown("#### Patients by Reason")
        reason_counts = p_df["reason"].value_counts().sort_values(ascending=False)
        st.bar_chart(reason_counts)
with cpa2:
    if not p_df.empty:
        st.markdown("#### Patients Assigned per Doctor")
        assigned_counts = p_df["doctor_id"].astype("string").value_counts().sort_values(ascending=False)
        st.bar_chart(assigned_counts)

to_csv_download(p_df.drop(columns=["registered_at_dt"], errors="ignore"), "patients.csv", "Patients")

st.markdown("---")

# -------- Doctors --------
st.markdown("## 🩺 Doctors")
st.dataframe(
    d_df,
    use_container_width=True,
    height=min(400, 40 + 28 * min(len(d_df), 10))
)

cd1, cd2 = st.columns(2)
with cd1:
    if not d_df.empty:
        st.markdown("#### Availability")
        status_counts = (
            d_df["available"].map({1: "Available", 0: "Unavailable"}).value_counts().sort_index()
        )
        st.bar_chart(status_counts)
with cd2:
    if not d_df.empty:
        st.markdown("#### By Specialization")
        spec_counts = d_df["specialization"].value_counts().sort_values(ascending=False)
        st.bar_chart(spec_counts)

to_csv_download(d_df, "doctors.csv", "Doctors")

st.markdown("---")

# -------- Medicines --------
st.markdown("## 💊 Medicines")
# highlight low stock
def _color_low(val):
    try:
        v = int(val)
        if v == 0: return "background-color: #ffcccc"   # red-ish
        if v < 50: return "background-color: #fff3cd"   # amber
    except: pass
    return ""

styled_meds = m_df.style.map(_color_low, subset=["stock"])
st.dataframe(
    m_df, use_container_width=True,
    height=min(400, 40 + 28 * min(len(m_df), 12))
)

if not m_df.empty:
    st.markdown("#### Stock Levels")
    stock_chart = m_df.set_index("name")["stock"].sort_values(ascending=False)
    st.bar_chart(stock_chart)

    low_stock = m_df[m_df["stock"] < 50]
    if not low_stock.empty:
        st.warning("⚠️ Low Stock Medicines (Below 50 units)")
        st.dataframe(low_stock, use_container_width=True)

to_csv_download(m_df, "medicines.csv", "Medicines")

# -------- Footer --------
st.success(f"Last updated: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S UTC')}")