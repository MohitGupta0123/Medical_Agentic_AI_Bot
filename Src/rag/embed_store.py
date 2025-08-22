import os
import json
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# Get project root dynamically (3 levels up from current file)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

DATA_DIR = os.path.join(BASE_DIR, "Artifacts")
RAW_PDF_DIR = os.path.join(DATA_DIR, "raw_pdf")
PROCESSED_TEXT_DIR = os.path.join(DATA_DIR, "processed_text")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
PAGE_IMAGES_DIR = os.path.join(DATA_DIR, "page_images")
PROCESSED_TEXT_PATH = os.path.join(PROCESSED_TEXT_DIR, "chunks_metadata.json")
FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(EMBEDDINGS_DIR, "metadata.pkl")


def create_faiss_index():
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    with open(PROCESSED_TEXT_PATH, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    model = SentenceTransformer("/app/models/all-MiniLM-L6-v2")
    # model = SentenceTransformer("models/all-MiniLM-L6-v2")

    # Embed text chunks
    texts = [chunk["content"] for chunk in chunks_data]
    text_embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    # Embed captions
    for chunk in chunks_data:
        for caption in chunk.get("captions", []):
            if caption["caption_text"] and caption["caption_text"].lower() != "no caption detected":
                caption_emb = model.encode(caption["caption_text"], convert_to_numpy=True)
                caption["embedding"] = caption_emb.tolist()
            else:
                caption["embedding"] = None

    # Build FAISS index
    dimension = text_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(text_embeddings)
    
    print(f"FAISS index size: {index.ntotal}")

    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(chunks_data, f)

    print(f"FAISS index saved to {FAISS_INDEX_PATH}")
    print(f"Metadata saved to {METADATA_PATH}")

if __name__ == "__main__":
    create_faiss_index()
