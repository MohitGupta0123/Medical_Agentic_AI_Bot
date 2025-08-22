import os
import json
import nltk
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdf_utils import (
    extract_text_with_tables,
    extract_images_pymupdf,
    extract_images_with_captions,
    extract_full_page_images
)

# Get project root dynamically (3 levels up from current file)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

DATA_DIR = os.path.join(BASE_DIR, "Artifacts")
RAW_PDF_DIR = os.path.join(DATA_DIR, "raw_pdf")
PROCESSED_TEXT_DIR = os.path.join(DATA_DIR, "processed_text")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
PAGE_IMAGES_DIR = os.path.join(DATA_DIR, "page_images")
IMAGE_PATH = os.path.join(DATA_DIR, "images")
IMAGE_CAPTIONS_DIR = os.path.join(DATA_DIR, "image_with_captions")
pdf_path = os.path.join(RAW_PDF_DIR, "medical_book.pdf")
OUTPUT_JSON_PATH = os.path.join(PROCESSED_TEXT_DIR, "chunks_metadata.json")

os.makedirs(PROCESSED_TEXT_DIR, exist_ok=True)

# ---------------- CHUNKING ----------------
def chunk_combined_content(pages_data, pdf_path, chunk_size=800, overlap=50, mode="recursive"):
    """
    Chunk combined text (text + tables) into smaller parts.
    
    Args:
        pages_data (list): Extracted pages with combined text/tables.
        pdf_path (str): Path to the PDF file.
        chunk_size (int): Size of each chunk (only for recursive).
        overlap (int): Overlap between chunks (only for recursive).
        mode (str): "recursive" or "sentence".
        
    Returns:
        list: List of chunk metadata dictionaries.
    """
    formatted_chunks = []

    if mode == "recursive":
        # Recursive character splitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

        for page in pages_data:
            chunks = splitter.split_text(page["content"])
            for i, chunk in enumerate(chunks):
                formatted_chunks.append({
                    "chunk_id": f"{page['page_num']}_{i}",
                    "page_num": page["page_num"],
                    "content": chunk.strip(),
                    "pdf_file": os.path.basename(pdf_path),
                    "images": []
                })

    elif mode == "sentence":
        # Sentence splitting with merging short sentences
        for page in pages_data:
            sentences = nltk.sent_tokenize(page["content"])
            buffer = ""
            for i, sentence in enumerate(sentences):
                # Merge short sentences (< 50 chars)
                if len(buffer) + len(sentence) < 50:
                    buffer += " " + sentence
                else:
                    if buffer:
                        formatted_chunks.append({
                            "chunk_id": f"{page['page_num']}_{i}",
                            "page_num": page["page_num"],
                            "content": buffer.strip(),
                            "pdf_file": os.path.basename(pdf_path),
                            "images": []
                        })
                    buffer = sentence
            if buffer:
                formatted_chunks.append({
                    "chunk_id": f"{page['page_num']}_{len(sentences)}",
                    "page_num": page["page_num"],
                    "content": buffer.strip(),
                    "pdf_file": os.path.basename(pdf_path),
                    "images": []
                })

    else:
        raise ValueError("Invalid mode. Use 'recursive' or 'sentence'.")

    return formatted_chunks

# ---------------- MERGE TEXT + IMAGES ----------------
def merge_text_and_images_with_captions(chunks, image_map, page_snapshot_map, 
                                        # caption_map
                                        ):
    """
    Add extracted images, page snapshots, and captions to chunks.
    """
    for chunk in chunks:
        page_num = chunk["page_num"]

        # Add inline figure images
        chunk["images"] = [path.replace("\\", "/") for path in image_map.get(page_num, [])]

        # Add page snapshot
        chunk["page_snapshot"] = page_snapshot_map.get(page_num)

        # Add captions (list of {image_path, caption_text})
        # chunk["captions"] = caption_map.get(page_num, [])

    return chunks

# ---------------- SAVE JSON ----------------
def save_chunks_to_json(final_data, output_path=OUTPUT_JSON_PATH):
    """
    Save the final chunked data with tables + images to JSON.
    """
    print(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)
    print(f"Saved chunks metadata to: {output_path}")

if __name__ == "__main__":
    pdf_path = "Artifacts/raw_pdf/medical_book.pdf"

    combined_pages, logs = extract_text_with_tables(pdf_path)
    text_chunks = chunk_combined_content(combined_pages, pdf_path, chunk_size=600, mode="sentence")
    image_map = extract_images_pymupdf(pdf_path, IMAGE_PATH)
    # caption_map = extract_images_with_captions(pdf_path)
    page_snapshot_map = extract_full_page_images(pdf_path)

    final_data = merge_text_and_images_with_captions(text_chunks, image_map, page_snapshot_map, 
                                                    #  caption_map
                                                     )
    save_chunks_to_json(final_data, OUTPUT_JSON_PATH)
