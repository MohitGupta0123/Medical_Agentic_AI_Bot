# src/agent/tools.py

from langchain.tools import tool
from ..services.patient_service import register_patient, get_patient_full_case
from ..services.doctor_service import confirm_assigned_doctor
from ..services.medicine_service import check_medicine_availability
from ..services.summarizer import summarize_patient_case
from ..rag.rag_pipeline import rag_query_multimodal

# -------------------------------
# Register Patient Tool
# -------------------------------
@tool("register_patient", return_direct=True)
def register_patient_tool(name: str, age: int, reason: str) -> str:
    """Register a new patient with their details and reason for visit."""
    patient = register_patient({"name": name, "age": age, "reason": reason})
    return (
        f"Patient {patient['name']}, {patient['age']} years old, "
        f"registered successfully with a complaint of {patient['reason']}. "
        f"Patient ID: {patient['id']}."
    )

# -------------------------------
# Confirm Appointment Tool
# -------------------------------
@tool("confirm_appointment", return_direct=True)
def confirm_appointment_tool(name: str) -> str:
    """
    Confirm appointment with the doctor already assigned to this patient.
    If no assigned doctor is found, notify the user.
    """
    doctor = confirm_assigned_doctor(name)
    if doctor:
        return (
            f"Appointment confirmed with {doctor['name']} "
            f"({doctor['specialization']}) for patient {name}."
        )
    else:
        return f"No assigned doctor found for patient {name}. Please register first."

# -------------------------------
# Medicine Availability Tool
# -------------------------------
@tool("medicine_availability", return_direct=True)
def medicine_availability_tool(medicine_name: str) -> str:
    """Check if a specific medicine is available in stock."""
    return check_medicine_availability(medicine_name)

# -------------------------------
# Summarize Case Tool
# -------------------------------
@tool("summarize_case", return_direct=True)
def summarize_case_tool(patient_id: int, hf_token: str = None) -> str:
    """
    Summarize a patient's case using stored data and LLM.
    Provide patient_id as input.
    """
    patient_data = get_patient_full_case(patient_id)
    summary = summarize_patient_case(patient_data, hf_token=hf_token)
    return summary

# -------------------------------
# RAG Medical Knowledge Tool (Optional if used separately)
# -------------------------------
@tool("medical_rag", return_direct=True)
def medical_rag_tool(query: str, hf_token: str = None) -> str:
    """Answer medical questions from the PDF knowledge base using RAG."""
    answer, refs = rag_query_multimodal(query, k=5, hf_token=hf_token)
    refs_str = "\n".join([f"Page {r['page']}: {r['link']}" for r in refs])
    return f"{answer}\n\nReferences:\n{refs_str}"
