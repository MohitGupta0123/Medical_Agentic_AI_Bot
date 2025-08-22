# Src/services/doctor_service.py
from datetime import datetime, timedelta
from sqlalchemy import select, update, text
from .db import get_session, row_to_dict
from .db import Doctor, Patient  # ORM models from services/db.py

STALE_MINUTES = 20  # same as before


# --------------------------
# Utility to release stale bookings
# --------------------------
def release_stale_doctors():
    """Release doctors who were booked >20 minutes ago."""
    threshold_time = (datetime.utcnow() - timedelta(minutes=STALE_MINUTES)).isoformat()

    with get_session() as s:
        # Only flip those that are currently marked unavailable and past threshold
        s.execute(
            text("""
                UPDATE doctors
                SET available = 1
                WHERE available = 0
                  AND last_booked_at <= :ts
            """),
            {"ts": threshold_time},
        )
        # session commits on context exit


# --------------------------
# Fetch doctor (by specialization or fallback)
# --------------------------
def get_available_doctor(specialization: str | None = None):
    """
    Fetch an available doctor.
    - If specialization is provided → prioritize doctors with that specialization.
    - If none found → fallback to any available doctor.
    Also marks the returned doctor as unavailable and stamps last_booked_at.
    """
    release_stale_doctors()

    with get_session() as s:
        doc = None

        if specialization:
            # Try matching specialization first
            doc = s.execute(
                select(Doctor)
                .where(Doctor.specialization == specialization, Doctor.available == 1)
                .limit(1)
            ).scalar_one_or_none()

            # Fallback to any available doctor
            if doc is None:
                doc = s.execute(
                    select(Doctor).where(Doctor.available == 1).limit(1)
                ).scalar_one_or_none()
        else:
            doc = s.execute(
                select(Doctor).where(Doctor.available == 1).limit(1)
            ).scalar_one_or_none()

        if doc is None:
            return None

        # Mark unavailable + set last_booked_at
        doc.available = 0
        doc.last_booked_at = datetime.utcnow().isoformat()
        s.add(doc)
        s.flush()
        s.refresh(doc)

        return row_to_dict(doc)


# --------------------------
# Confirm appointment with already assigned doctor
# --------------------------
def confirm_assigned_doctor(patient_name: str):
    """
    Confirm appointment with doctor already assigned to the patient.
    - Fetch doctor_id from patients table
    - Mark doctor unavailable (if not already)
    - Return doctor details
    """
    release_stale_doctors()

    with get_session() as s:
        # Join patients -> doctors by patient name
        result = s.execute(
            select(Doctor)
            .join(Patient, Patient.doctor_id == Doctor.id)
            .where(Patient.name == patient_name)
            .limit(1)
        ).scalar_one_or_none()

        if result is None:
            return None  # No patient or doctor assigned

        # Mark doctor unavailable (confirm booking)
        result.available = 0
        result.last_booked_at = datetime.utcnow().isoformat()
        s.add(result)
        s.flush()
        s.refresh(result)

        return row_to_dict(result)
