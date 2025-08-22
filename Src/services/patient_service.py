# src/services/patient_service.py
from datetime import datetime
from typing import Optional, Dict

from sqlalchemy import select, desc, text

from .db import get_session, row_to_dict
from .db import Patient, Doctor  # ORM models


# -------------------------
# Register Patient
# -------------------------
def register_patient(data: dict, doctor_id: int | None = None) -> Dict:
    """
    Register a new patient into the database.
    Optionally assign a doctor_id during registration (if doctor assigned via LLM).
    Returns full patient record (dict).
    """
    name = data["name"]
    age = int(data["age"])
    reason = data["reason"]

    # Basic validation
    if not name.strip():
        raise ValueError("Patient name cannot be empty.")
    if age <= 0:
        raise ValueError("Invalid age provided.")
    if not reason.strip():
        raise ValueError("Reason for visit cannot be empty.")

    ts = datetime.utcnow().isoformat()

    with get_session() as s:
        p = Patient(
            name=name,
            age=age,
            reason=reason,
            registered_at=ts,
            doctor_id=doctor_id,
        )
        s.add(p)
        s.flush()   # get auto id
        s.refresh(p)

        return {
            "id": p.id,
            "name": p.name,
            "age": p.age,
            "reason": p.reason,
            "doctor_id": p.doctor_id,
            "registered_at": p.registered_at,
        }


# -------------------------
# Fetch Patient by Name
# -------------------------
def get_patient_by_name(name: str) -> Optional[Dict]:
    """
    Retrieve patient details by name (latest entry if multiple exist).
    """
    with get_session() as s:
        # Order by registered_at DESC to get the latest
        q = (
            select(Patient)
            .where(Patient.name == name)
            .order_by(desc(Patient.registered_at))
            .limit(1)
        )
        p = s.execute(q).scalar_one_or_none()
        return row_to_dict(p) if p else None


# -------------------------
# Update Assigned Doctor
# -------------------------
def update_patient_doctor(patient_id: int, doctor_id: int) -> bool:
    """
    Update the assigned doctor for a patient.
    Useful if doctor is assigned after initial registration.
    """
    with get_session() as s:
        p = s.get(Patient, patient_id)
        if not p:
            return False
        p.doctor_id = doctor_id
        s.add(p)
        return True


# -------------------------
# Fetch Patient Full Case (with doctor info)
# -------------------------
def get_patient_full_case(patient_id: int) -> Optional[Dict]:
    """
    Fetch patient's case details including assigned doctor info.
    """
    with get_session() as s:
        # Join to doctor to fetch name + specialization
        q = (
            select(
                Patient.id.label("patient_id"),
                Patient.name.label("patient_name"),
                Patient.age,
                Patient.reason,
                Patient.registered_at,
                Doctor.name.label("doctor_name"),
                Doctor.specialization,
            )
            .select_from(Patient)
            .join(Doctor, Patient.doctor_id == Doctor.id, isouter=True)
            .where(Patient.id == patient_id)
            .limit(1)
        )
        row = s.execute(q).mappings().first()
        return dict(row) if row else None
