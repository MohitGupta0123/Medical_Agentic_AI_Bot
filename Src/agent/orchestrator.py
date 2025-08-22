# src/agent/orchestrator.py
import json
import re
from typing import Tuple, Dict, Any, Optional

from ..rag.rag_pipeline import rag_query_multimodal
from .tools import (
    register_patient_tool,
    confirm_appointment_tool,
    medicine_availability_tool,
)
from ..services.patient_service import get_patient_full_case, register_patient as save_patient
from ..services.doctor_service import release_stale_doctors
from ..services.doctor_assignment import assign_doctor_with_gemma
from ..services.summarizer import summarize_patient_case
from .gemma_chat_llm import GemmaChatLLM2


# -------------------------
# Helpers
# -------------------------
def _preview(text: Optional[str], n: int = 160) -> Optional[str]:
    if not isinstance(text, str) or not text.strip():
        return None
    return (text[:n] + "…") if len(text) > n else text

def _safe_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


# -------------------------
# Classify Query Type
# -------------------------
def classify_query_with_llm(query: str, hf_token: str = None) -> str:
    llm = GemmaChatLLM2(hf_token=hf_token)

    prompt = (
        "You Just give answer in same words and in single word"
        "You are an intelligent medical assistant. Based on the user's question below, classify the action into one of the following:\n\n"
        "- register_patient\n"
        "- confirm_appointment\n"
        "- medicine_availability\n"
        "- summarize_case\n"
        "- rag\n\n"
        f"User Query: {query}\n\n"
        "Action:"
    )

    response = llm._call(prompt).strip().lower()
    
    return response

def extract_parameter(query: str, action: str, hf_token: str = None) -> dict:
    """
    Extracts structured parameters for a given action using the LLM.
    Returns {} on failure (we'll handle fallbacks in each branch).
    """
    llm = GemmaChatLLM2(hf_token=hf_token)

    prompt_map = {
        "register_patient": (
            "Extract the patient's name, age, and reason from the following query. "
            "Return only JSON: {\"name\": ..., \"age\": ..., \"reason\": ...}"
        ),
        "confirm_appointment": (
            "Extract the patient's name from the following query. "
            "Return only JSON: {\"name\": ...}"
        ),
        "medicine_availability": (
            "Extract the medicine name from the following query. "
            "Return only JSON: {\"medicine_name\": ...}"
        ),
        "summarize_case": (
            "Extract the patient ID from the following query. "
            "Return only JSON: {\"patient_id\": ...}"
        )
    }

    if action not in prompt_map:
        return {}

    prompt = prompt_map[action] + f"\n\nQuery: {query}\n\nAnswer:"
    response = llm._call(prompt).strip()

    # Strip fenced code blocks if any
    response = re.sub(r"```json|```", "", response).strip()

    try:
        return json.loads(response)
    except Exception:
        # log for debugging
        print(f"[extract_parameter] Failed to parse JSON: {response}")
        return {}


# -------------------------
# Orchestrate Query Handling
# -------------------------
def orchestrate_query(query: str, hf_token: str = None) -> Tuple[Dict[str, Any], list]:
    action = classify_query_with_llm(query, hf_token=hf_token)
    params = extract_parameter(query, action, hf_token=hf_token)

    print(f"[orchestrate] action={action}")
    print(f"[orchestrate] params={params}")

    # ------------------ register_patient ------------------
    if action == "register_patient":
        # free up stale bookings first
        release_stale_doctors()

        name = (params.get("name") or "").strip()
        age = _safe_int(params.get("age"))
        reason = (params.get("reason") or "").strip()

        if not (name and age > 0 and reason):
            return {
                "type": "register_patient",
                "ok": False,
                "message": "Missing registration details. Please provide name, age, and reason.",
                "missing": {"name": not bool(name), "age": age <= 0, "reason": not bool(reason)},
            }, []

        # LLM doctor assignment
        doctor, reasoning = assign_doctor_with_gemma(reason, hf_token=hf_token)
        if not doctor:
            return {
                "type": "register_patient",
                "ok": False,
                "message": "No suitable doctor found. Please try again later."
            }, []

        # Save patient
        patient_record = save_patient(
            {"name": name, "age": age, "reason": reason},
            doctor["id"]
        )

        return {
            "type": "register_patient",
            "ok": True,
            "patient_id": patient_record["id"],
            "assigned_doctor": doctor,
            "reasoning": reasoning,
            "message": f"Patient {name} registered (ID {patient_record['id']}) and assigned to {doctor.get('name')}."
        }, []

    # ------------------ confirm_appointment ------------------
    elif action == "confirm_appointment":
        result = confirm_appointment_tool.run(params)  # your tool returns a dict/string
        if isinstance(result, dict):
            result.setdefault("type", "confirm_appointment")
            result.setdefault("ok", True)
            result.setdefault("message", result.get("message") or "Appointment status updated.")
            return result, []
        else:
            return {
                "type": "confirm_appointment",
                "ok": True,
                "message": str(result)
            }, []

    # ------------------ medicine_availability ------------------
    elif action == "medicine_availability":
        result = medicine_availability_tool.run(params)
        if isinstance(result, dict):
            result.setdefault("type", "medicine_availability")
            result.setdefault("ok", True)
            result.setdefault("medicine_message", result.get("message"))
            result.setdefault("message", result.get("message") or "Checked medicine availability.")
            return result, []
        else:
            return {
                "type": "medicine_availability",
                "ok": True,
                "medicine_message": str(result),
                "message": str(result),
            }, []

    # ------------------ summarize_case ------------------
    elif action == "summarize_case":
        patient_id = _safe_int(params.get("patient_id"), default=0)
        if patient_id <= 0:
            # try a tiny fallback: detect digits in query
            m = re.search(r"\b(\d+)\b", query)
            if m:
                patient_id = int(m.group(1))

        patient_data = get_patient_full_case(patient_id) if patient_id > 0 else None
        if not patient_data:
            return {
                "type": "summarize_case",
                "ok": False,
                "message": f"Patient not found for id={patient_id}."
            }, []

        summary = summarize_patient_case(patient_data, hf_token=hf_token)
        return {
            "type": "summarize_case",
            "ok": True,
            "summary": summary,
            "message": "Summary ready."
        }, []

    # ------------------ rag (default) ------------------
    else:
        print("Answering using Medical Chatbot (RAG)")
        answer, references = rag_query_multimodal(query, k=10, hf_token=hf_token)
        return {
            "type": "rag",                # <--- FRONTEND FLAG
            "ok": True,
            "answer": answer,
            "references": references,
            "message": "Answer ready."
        }, []
