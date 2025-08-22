import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm

import os

# Get project root dynamically (3 levels up from current file)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

DATA_DIR = os.path.join(BASE_DIR, "Artifacts")
RAW_PDF_DIR = os.path.join(DATA_DIR, "raw_pdf")
PROCESSED_TEXT_DIR = os.path.join(DATA_DIR, "processed_text")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
PAGE_IMAGES_DIR = os.path.join(DATA_DIR, "page_images")
FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(EMBEDDINGS_DIR, "metadata.pkl")
PDF_DIR = RAW_PDF_DIR

# Load index & metadata
index = faiss.read_index(FAISS_INDEX_PATH)
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

embed_model = SentenceTransformer("/app/models/all-MiniLM-L6-v2")
# embed_model = SentenceTransformer("models/all-MiniLM-L6-v2")

def retrieve_top_k(query, k=5, similarity_threshold=0):
    query_vec = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, k)

    max_dist = np.max(distances)
    similarities = 1 - (distances / max_dist)

    raw_results = []
    for idx, dist, sim in zip(indices[0], distances[0], similarities[0]):
        if idx == -1: continue
        if sim < similarity_threshold: continue

        chunk_meta = metadata[idx]
        raw_results.append({
            "content": chunk_meta["content"],
            "page_num": chunk_meta["page_num"],
            "pdf_file": chunk_meta["pdf_file"],
            "page_snapshot": chunk_meta.get("page_snapshot"),
            "images": chunk_meta.get("images", []),
            # "captions": chunk_meta.get("captions", []),
            "distance": float(dist),
            "similarity": float(sim),
            "link": f"{PDF_DIR}/{chunk_meta['pdf_file']}#page={chunk_meta['page_num']}"
        })

    # Group by page
    grouped = {}
    for r in raw_results:
        page = r["page_num"]
        if page not in grouped:
            grouped[page] = {
                "page_num": page,
                "pdf_file": r["pdf_file"],
                "content": [],
                "images": [],
                # "captions": r.get("captions", []),
                "page_snapshot": r["page_snapshot"],
                "link": r["link"]
            }
        grouped[page]["content"].append(r["content"])
        grouped[page]["images"].extend(r["images"])

    results = []
    for page, data in grouped.items():
        merged_text = " ".join(data["content"])
        snippet = merged_text[:150] + "..." if len(merged_text) > 150 else merged_text
        results.append({
            "page_num": page,
            "pdf_file": data["pdf_file"],
            "content": merged_text,
            "page_snapshot": data["page_snapshot"],
            "images": list(set(data["images"])),
            # "captions": data.get("captions", []),
            "link": data["link"],
            "snippet": snippet
        })

    return sorted(results, key=lambda x: x["page_num"])

def filter_images_by_caption_similarity(query, captions, threshold=0.4):
    query_emb = embed_model.encode([query], convert_to_numpy=True)[0]
    relevant_images = []
    for cap in captions:
        if cap.get("embedding") is not None:
            cap_emb = np.array(cap["embedding"])
            sim = np.dot(query_emb, cap_emb) / (norm(query_emb) * norm(cap_emb))
            if sim >= threshold:
                relevant_images.append(cap["image_path"])
    return relevant_images
