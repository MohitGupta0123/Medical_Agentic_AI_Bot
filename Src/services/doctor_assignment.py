# Src/services/doctor_assignment.py
from datetime import datetime
from typing import List, Dict, Tuple, Optional

from huggingface_hub import InferenceClient
from sqlalchemy import select

from .db import get_session, row_to_dict, Doctor  # SQLAlchemy session + ORM model

MODEL_NAME = "google/gemma-3-27b-it"


# -----------------------
# Helper Functions
# -----------------------
def fetch_available_doctors() -> List[Dict]:
    """Fetch all available doctors from DB (as dicts)."""
    with get_session() as s:
        rows = s.execute(
            select(Doctor).where(Doctor.available == 1).order_by(Doctor.id.asc())
        ).scalars().all()
        return [row_to_dict(d) for d in rows]


def mark_doctor_unavailable(doctor_id: int) -> None:
    """Mark doctor as unavailable (appointment booked)."""
    ts = datetime.utcnow().isoformat()
    with get_session() as s:
        doc = s.get(Doctor, doctor_id)
        if doc:
            doc.available = 0
            doc.last_booked_at = ts
            s.add(doc)
            # commit handled by context manager


def _extract_text_from_hf_chat(response) -> str:
    """
    HF chat responses can vary a bit by version:
    - choices[0].message["content"] might be a string
    - or a list of chunks like [{"type":"text","text":"..."}]
    Normalize to a single string.
    """
    try:
        msg = response.choices[0].message
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # collect "text" fields
            parts = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    parts.append(c.get("text", ""))
            return "\n".join(parts).strip()
        # fallback: try choices[0].text if present on some SDKs
        txt = getattr(response.choices[0], "text", None)
        if isinstance(txt, str):
            return txt
    except Exception:
        pass
    return ""


# -----------------------
# Main Assignment Logic
# -----------------------
def assign_doctor_with_gemma(patient_reason: str, hf_token: Optional[str] = None) -> Tuple[Optional[Dict], str]:
    """
    Use Gemma LLM to assign the most suitable doctor based on patient's reason.
    Returns (doctor_dict, reasoning_text).
    """
    # Step 1: Fetch available doctors
    doctors = fetch_available_doctors()
    if not doctors:
        return None, "No doctors available at the moment."

    # Prepare doctor options for LLM
    doctor_list_str = "\n".join([f"- {doc['name']} ({doc['specialization']})" for doc in doctors])

    # Step 2: Construct prompt for Gemma
    system_prompt = (
        "You are an intelligent hospital assistant.\n"
        "Task: Assign the most suitable doctor for a patient based on their medical reason.\n"
        "You must choose ONE doctor from the provided list and explain why briefly.\n"
        "Return the doctor's name or specialization and a short reasoning."
    )

    user_prompt = (
        f"Available doctors:\n{doctor_list_str}\n\n"
        f"Patient's reason: {patient_reason}\n\n"
        "Which doctor should handle this case and why?"
    )

    # Step 3: Call Gemma model
    client = InferenceClient(token=hf_token, provider="auto", timeout=400)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ],
        max_tokens=300,
    )
    reasoning = _extract_text_from_hf_chat(response).strip()

    # Step 4: Match doctor by name or specialization from LLM response
    assigned_doctor = None
    lower_reasoning = reasoning.lower()
    for doc in doctors:
        if doc["name"].lower() in lower_reasoning or doc["specialization"].lower() in lower_reasoning:
            assigned_doctor = doc
            break

    # Step 5: Fallback - If LLM fails to match, choose first available doctor
    if not assigned_doctor:
        assigned_doctor = doctors[0]
        fallback_note = f"(Fallback: Assigned first available doctor {assigned_doctor['name']})"
        reasoning = f"{reasoning}\n{fallback_note}" if reasoning else fallback_note

    # Step 6: Mark doctor unavailable
    mark_doctor_unavailable(assigned_doctor["id"])

    return assigned_doctor, reasoning
