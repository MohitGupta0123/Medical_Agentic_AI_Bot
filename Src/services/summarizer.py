# src/services/summarizer.py
from huggingface_hub import InferenceClient

MODEL_NAME = "google/gemma-3-27b-it"

def summarize_patient_case(patient_data: dict, hf_token: str= None) -> str:
    """
    Summarize the patient's case using LLM (Gemma).
    """
    if not patient_data:
        return "No patient data found."

    # Construct context
    context = (
        f"Patient Name: {patient_data['patient_name']}\n"
        f"Age: {patient_data['age']}\n"
        f"Reason for Visit: {patient_data['reason']}\n"
        f"Registered At: {patient_data['registered_at']}\n"
        f"Assigned Doctor: {patient_data['doctor_name']} ({patient_data['specialization']})\n"
    )

    prompt = (
        "You are a hospital assistant. Summarize this patient’s case in 3-4 sentences, "
        "mentioning the patient's name, age, reason for visit, and assigned doctor:\n\n"
        f"{context}"
    )

    client = InferenceClient(provider="auto", timeout=400, token=hf_token)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful summarization assistant."}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ],
        max_tokens=300
    )

    return response.choices[0].message["content"]
