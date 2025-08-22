# src/agent/agent_executor.py
from .gemma_chat_llm import GemmaChatLLM
from langchain.agents import initialize_agent, Tool
from .tools import (
    register_patient_tool,
    confirm_appointment_tool,
    medicine_availability_tool,
)
from ..rag.rag_pipeline import rag_query_multimodal
from ..services.patient_service import get_patient_full_case
from ..services.summarizer import summarize_patient_case

# -----------------------------------
# RAG Tool Function
# -----------------------------------
def rag_tool_func(query: str, hf_token: str = None):
    answer, refs = rag_query_multimodal(query, k=10, hf_token=hf_token)
    refs_str = "\n".join([f"Page {r['page']}: {r['link']}" for r in refs])
    return f"{answer}\n\nReferences:\n{refs_str}"

# -----------------------------------
# Summarizer Tool Function
# -----------------------------------
def summarize_case_func(patient_id: int, hf_token: str = None) -> str:
    data = get_patient_full_case(patient_id)
    return summarize_patient_case(data, hf_token=hf_token)

# -----------------------------------
# Initialize Agent Executor
# -----------------------------------
def get_agent_executor(hf_token: str = None):
    # RAG Tool
    rag_tool = Tool(
        name="MedicalRAG",
        func=lambda q: rag_tool_func(q, hf_token=hf_token),
        description="Use this tool to answer medical queries from the PDF knowledge base."
    )

    # Summarizer Tool (Token-aware)
    summarizer_tool = Tool(
        name="SummarizePatientCase",
        func=lambda pid: summarize_case_func(pid, hf_token=hf_token),
        description="Summarize a patient's case using their patient ID."
    )

    tools = [
        register_patient_tool,
        confirm_appointment_tool,
        medicine_availability_tool,
        summarizer_tool,
        rag_tool
    ]

    llm = GemmaChatLLM(model="google/gemma-3-27b-it", temperature=0.2, hf_token=hf_token)

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="structured-chat-zero-shot-react-description",
        verbose=True
    )
    return agent