# Src/rag/rag_pipeline.py

from huggingface_hub import InferenceClient
from Src.rag.retriever import retrieve_top_k
import os

# Get project root dynamically (3 levels up from current file)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

DATA_DIR = os.path.join(BASE_DIR, "Artifacts")
RAW_PDF_DIR = os.path.join(DATA_DIR, "raw_pdf")
PROCESSED_TEXT_DIR = os.path.join(DATA_DIR, "processed_text")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
PAGE_IMAGES_DIR = os.path.join(DATA_DIR, "page_images")

def generate_answer_multimodal(query, retrieved_chunks, model="google/gemma-3-27b-it", hf_token=None):
    context_text = "\n\n".join([
        f"--- Page {c['page_num']} ---\nText:\n{c['content']}" for c in retrieved_chunks
    ])

    system_prompt = """
You are a knowledgeable medical assistant.

- Use ONLY the provided context (text) to answer the user query.
- Output strictly in English.
- If specific sections like Definition, Causes, Diagnosis, or Treatment are relevant, organize the answer using these sections.
- If the query does not fit these sections (e.g., a simple definition or general explanation), provide a concise, well-structured paragraph instead.
- If any requested information is missing in the context, explicitly state "Not mentioned in the document."
- Do NOT invent facts, add unrelated metadata, or include random symbols.
- Do NOT mention images unless explicitly described in the text.
- If you need to say about document then say According to my sources.

For structured answers (when applicable), use this format:

**Definition:**
<text>

**Causes:**
<text>

**Diagnosis:**
<text>

**Treatment:**
<text>

For general answers (when structured sections are irrelevant), provide a single detailed paragraph addressing the query clearly and factually and in detail.

For Example:
Query:
What is asthma and how is it diagnosed?

Answer:
**Definition:**
Asthma is a chronic inflammatory disease of the airways characterized by episodes of wheezing and breathlessness.

**Causes:**
- Allergic reactions
- Environmental triggers
- Respiratory infections

**Diagnosis:**
- Spirometry
- Peak flow measurement

**Treatment:**
- Inhaled corticosteroids
- Bronchodilators

For Example:
Query:
What is antigen?

Answer:
An Antigen (Ag) is a molecule, moiety, foreign particulate matter, or an allergen, such as pollen, that can bind to a specific antibody or T-cell receptor. The presence of antigens in the body may trigger an immune response.
Antigens can be proteins, peptides (amino acid chains), polysaccharides (chains of simple sugars), lipids, or nucleic acids.[3][4] Antigens exist on normal cells, cancer cells, parasites, viruses, fungi, and bacteria.
"""

    user_prompt = f"""
Context:
{context_text}

User Question:
{query}
"""

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
    ]

    client = InferenceClient(token=hf_token, provider="auto", timeout=400)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        max_tokens=1000
    )

    return response.choices[0].message["content"]

def rag_query_multimodal(query, k=5, hf_token=None):
    retrieved = retrieve_top_k(query, k=k)
    answer = generate_answer_multimodal(query, retrieved, hf_token=hf_token)

    references = []
    for r in retrieved:
        relevant_images = r.get("images", [])
        references.append({
            "page": r["page_num"],
            "link": r["link"],
            "snippet": r["snippet"],
            "page_snapshot": r.get("page_snapshot"),
            "images": relevant_images
        })

    return answer, references
