# Src/api/fastapi_app.py
import os
from datetime import datetime

from fastapi import FastAPI, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import text, select

# DB and services
from ..services.db import (
    init_db, seed_data, get_session,
    Patient, Doctor, Medicine,  # ORM models for admin endpoints
)
from ..services.patient_service import register_patient as save_patient, get_patient_full_case
from ..services.medicine_service import check_medicine_availability
from ..services.doctor_service import release_stale_doctors
from ..services.doctor_assignment import assign_doctor_with_gemma
from ..services.summarizer import summarize_patient_case
# RAG
from ..rag.rag_pipeline import rag_query_multimodal
# Agent system
from ..agent.orchestrator import orchestrate_query
from ..agent.agent_executor import get_agent_executor

app = FastAPI(title="Medical Agentic Bot Backend FastAPI", version="1.0.0")

# ----------------------------
# CORS (Streamlit Space + local dev)
# ----------------------------
frontend_origin = os.getenv("FRONTEND_ORIGIN")
origins = [o for o in [frontend_origin, "http://localhost:8501"] if o]
# fallback (dev): if nothing set, allow all (you can tighten later)
allow_origins = origins if origins else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# On Startup: DB Init + Seed
# ----------------------------
@app.on_event("startup")
def startup_event():
    init_db()
    seed_data()

# ----------------------------
# Root
# ----------------------------
# class Message(BaseModel):
#     message: str

# @app.get("/", response_model=Message, summary="Root Endpoint", tags=["Root"])
# def read_root():
#     return {"message": "Welcome to My Awesome API!"}

class Message(BaseModel):
    message: str
    OWNER: str
    Purpose: str

@app.get("/", response_model=Message, summary="Root Endpoint", tags=["Root"])
def read_root():
    """
    Root endpoint for checking the API status.

    Returns a welcome message.
    """
    return {"message": "Welcome to My Awesome Backend FastAPI Application!",
            "OWNER": "MOHIT GUPTA",
            "Purpose": "THIS API IS BACKEND FASTAPI FOR AGENTIC MEDICAL AI BOT."}

# ----------------------------
# 1. Medical RAG Chatbot
# ----------------------------
@app.get("/query")
def query_bot(q: str = Query(...), authorization: str = Header(...)):
    hf_token = authorization.replace("Bearer ", "")
    answer, references = rag_query_multimodal(q, k=10, hf_token=hf_token)
    return {"answer": answer, "references": references}

# ----------------------------
# 2. Register Patient + Assign Doctor
# ----------------------------
class PatientData(BaseModel):
    name: str
    age: int
    reason: str

@app.post("/register_patient")
def register_patient_api(data: PatientData, authorization: str = Header(...)):
    hf_token = authorization.replace("Bearer ", "")
    release_stale_doctors()
    doctor, reasoning = assign_doctor_with_gemma(data.reason, hf_token=hf_token)

    if not doctor:
        return {"message": "No suitable doctor found. Please try again later."}

    patient_record = save_patient(
        {"name": data.name, "age": data.age, "reason": data.reason},
        doctor["id"]
    )

    return {
        "patient_id": patient_record["id"],
        "assigned_doctor": doctor,
        "reasoning": reasoning
    }

# ----------------------------
# 3. Check Registration & Confirm Appointment
# ----------------------------
class AppointmentData(BaseModel):
    name: str

@app.post("/check_registration_status")
def check_registration_status(data: AppointmentData, authorization: str = Header(...)):
    """
    Looks up the patient by name, verifies/sets doctor availability,
    and confirms appointment—same behavior as before.
    """
    with get_session() as s:
        record = s.execute(
            text("""
                SELECT p.id AS patient_id, p.name AS patient_name, d.id AS doctor_id,
                       d.name AS doctor_name, d.specialization, d.available
                FROM patients p
                LEFT JOIN doctors d ON p.doctor_id = d.id
                WHERE p.name = :name
                LIMIT 1
            """),
            {"name": data.name}
        ).mappings().first()

        if not record:
            return {"message": "Patient not registered. Please register first."}

        if not record["doctor_id"]:
            return {"message": "No doctor assigned yet. Please register again."}

        if record["available"] == 1:
            s.execute(
                text("""
                    UPDATE doctors
                    SET available = 0, last_booked_at = :ts
                    WHERE id = :doc_id
                """),
                {"ts": datetime.utcnow().isoformat(), "doc_id": record["doctor_id"]}
            )

        return {
            "message": (
                f"Appointment confirmed with {record['doctor_name']} "
                f"({record['specialization']}) for {record['patient_name']}."
            ),
            "doctor_name": record["doctor_name"],
            "specialization": record["specialization"],
        }

# ----------------------------
# 4. Medicine Availability
# ----------------------------
@app.get("/medicine_availability")
def medicine_availability_api(name: str = Query(...), authorization: str = Header(...)):
    result = check_medicine_availability(name)
    return {"message": result}

# ----------------------------
# 5. Summarize Patient Case
# ----------------------------
@app.get("/summarize_case/{patient_id}")
def summarize_case_api(patient_id: int, authorization: str = Header(...)):
    hf_token = authorization.replace("Bearer ", "")
    patient_data = get_patient_full_case(patient_id)
    if not patient_data:
        return {"message": "Patient not found"}
    summary = summarize_patient_case(patient_data, hf_token=hf_token)
    return {"summary": summary}

# ----------------------------
# 6. LangChain Agent Endpoint
# ----------------------------
@app.get("/agent_query")
def agent_query(q: str = Query(...), authorization: str = Header(...)):
    hf_token = authorization.replace("Bearer ", "")
    agent = get_agent_executor(hf_token=hf_token)
    response = agent.run(q)
    return {"response": response}

# ----------------------------
# 7. Lightweight Orchestrator Endpoint
# ----------------------------
@app.get("/orchestrator_query")
def orchestrator_query(q: str = Query(...), authorization: str = Header(...)):
    hf_token = authorization.replace("Bearer ", "")
    result, references = orchestrate_query(q, hf_token=hf_token)
    return {"result": result, "references": references}

@app.post("/release_stale_doctors")
def release_stale_doctors_api():
    release_stale_doctors()
    return {"status": "success"}

# ----------------------------
# 8. Admin list endpoints (for Streamlit dashboard)
# ----------------------------
@app.get("/admin/patients")
def admin_list_patients(authorization: str = Header(...)):
    with get_session() as s:
        rows = s.execute(select(Patient)).scalars().all()
        return {"items": [
            {
                "id": p.id,
                "name": p.name,
                "age": p.age,
                "reason": p.reason,
                "registered_at": p.registered_at,
                "doctor_id": p.doctor_id,
            } for p in rows
        ]}

@app.get("/admin/doctors")
def admin_list_doctors(authorization: str = Header(...)):
    with get_session() as s:
        rows = s.execute(select(Doctor)).scalars().all()
        return {"items": [
            {
                "id": d.id,
                "name": d.name,
                "specialization": d.specialization,
                "available": d.available,
                "last_booked_at": d.last_booked_at,
            } for d in rows
        ]}

@app.get("/admin/medicines")
def admin_list_medicines(authorization: str = Header(...)):
    with get_session() as s:
        rows = s.execute(select(Medicine)).scalars().all()
        return {"items": [
            {"id": m.id, "name": m.name, "stock": m.stock} for m in rows
        ]}
