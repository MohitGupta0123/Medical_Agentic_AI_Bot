import fitz, re, os, contractions, camelot
from tqdm.notebook import tqdm
from langchain_community.document_loaders import PyPDFLoader
import nltk

# Make sure NLTK sentence tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Get project root dynamically (3 levels up from current file)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

DATA_DIR = os.path.join(BASE_DIR, "Artifacts")
RAW_PDF_DIR = os.path.join(DATA_DIR, "raw_pdf")
PROCESSED_TEXT_DIR = os.path.join(DATA_DIR, "processed_text")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
PAGE_IMAGES_DIR = os.path.join(DATA_DIR, "page_images")
IMAGE_CAPTIONS_DIR = os.path.join(DATA_DIR, "image_with_captions")
output_dir = IMAGE_CAPTIONS_DIR  # for inline images
snapshot_dir = PAGE_IMAGES_DIR   # for page snapshots

# Map variations to standard headings
heading_map = {
    "definition": "Definition",
    "description": "Definition",
    "overview": "Definition",
    "causes": "Causes",
    "etiology": "Causes",
    "incidence": "Causes",
    "diagnosis": "Diagnosis",
    "tests": "Diagnosis",
    "identification": "Diagnosis",
    "treatment": "Treatment",
    "therapy": "Treatment",
    "management": "Treatment"
}

# Replace headings with standardized version
def standardize_headings(text):
    # Detect lines starting with uppercase words
    pattern = r"(?<=\n)([A-Z][A-Za-z\s]+)(?=\n)"
    matches = re.findall(pattern, text)
    for match in matches:
        key = match.lower().strip()
        if key in heading_map:
            text = text.replace(match, heading_map[key])
    return text


def clean_text(text):
    """
    Clean extracted text from PDF for RAG:
    - Fix contractions, OCR errors, hyphenations
    - Remove headers, footers, page numbers, noisy sections
    - Normalize punctuation, spaces, and encoding artifacts
    - Return cleaned text + detailed change log
    """
    changes_log = {
        "contractions_expansion": [],
        "non_ascii_removal": [],
        "title_pages_removal": [],
        "headers_footers_removal": [],
        "bibliographic_removal": [],
        "key_terms_removal": [],
        "line_break_fix": [],
        "hyphen_fix": [],
        "bullet_conversion": [],
        "punctuation_normalization": [],
        "unicode_ligature_fix": [],
        "extra_space_clean": []
    }

    # Original backup
    original_text = text

    # 1. Expand contractions (don't → do not)
    expanded_text = contractions.fix(text)
    if expanded_text != text:
        changes_log["contractions_expansion"].append({
            "before": text[:500],
            "after": expanded_text[:500]
        })
    text = expanded_text

    # 2. Remove non-ASCII / corrupted symbols
    cleaned_ascii = re.sub(r'[^\x00-\x7F]+', ' ', text)
    if cleaned_ascii != text:
        changes_log["non_ascii_removal"].append({
            "before": text[:300],
            "after": cleaned_ascii[:300]
        })
    text = cleaned_ascii

    # 3. Remove title pages, contributor lists
    title_patterns = [
        r"The GALE\s+ENCYCLOPEDIA\s+of MEDICINE.*?(?=\n[A-Z])",
        r"STAFF\n.*?(?=\n[A-Z])",
        r"CONTRIBUTORS\n.*?(?=\n[A-Z])",
        r"ADVISORY BOARD\n.*?(?=\n[A-Z])",
        r"Library of Congress Cataloging.*?(?=\n[A-Z])"
    ]
    for pattern in title_patterns:
        matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if matches:
            changes_log["title_pages_removal"].append({
                "removed": matches[:3],
                "count": len(matches)
            })
            text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

    # 4. Remove headers, footers, page numbers
    header_footer_patterns = [
        r"GALE ENCYCLOPEDIA OF MEDICINE.*?\n",
        r"GEM\s*-\s*\d{4}\s*to\s*\d{4}.*?\n",
        r"Page \d+",
        r"\n\d+\n"
    ]
    for pattern in header_footer_patterns:
        matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if matches:
            changes_log["headers_footers_removal"].append({
                "removed": matches[:3],
                "count": len(matches)
            })
            text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

    text = standardize_headings(text)
    
    # 5. Remove bibliographic sections (Resources, Periodicals, Organizations)
    biblio_pattern = r"(Resources|Organizations|Periodicals|Further reading).*"
    biblio_matches = re.findall(biblio_pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if biblio_matches:
        changes_log["bibliographic_removal"].append({
            "removed": biblio_matches[:3],
            "count": len(biblio_matches)
        })
        text = re.sub(biblio_pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    # 6. Remove "KEY TERMS" and similar metadata
    key_terms_pattern = r"(KEY TERMS|SEE ALSO|Other Names).*"
    key_terms_matches = re.findall(key_terms_pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if key_terms_matches:
        changes_log["key_terms_removal"].append({
            "removed": key_terms_matches[:3],
            "count": len(key_terms_matches)
        })
        text = re.sub(key_terms_pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    # 7. Fix line breaks inside sentences
    fixed_lines = []
    for line in text.splitlines():
        if line and not re.match(r".*[.:;]$", line):
            fixed_lines.append(line.strip() + " ")
        else:
            fixed_lines.append(line.strip())
    fixed_text = " ".join(fixed_lines)
    if fixed_text != text:
        changes_log["line_break_fix"].append({
            "before": text[:500],
            "after": fixed_text[:500]
        })
    text = fixed_text

    # 8. Fix hyphenated words split across lines
    hyphen_fixed = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)
    if hyphen_fixed != text:
        changes_log["hyphen_fix"].append({
            "before": text[:200],
            "after": hyphen_fixed[:200]
        })
    text = hyphen_fixed

    # 9. Convert bullets to markdown (- )
    bullet_converted = re.sub(r"[•·]\s*", "- ", text)
    if bullet_converted != text:
        changes_log["bullet_conversion"].append({
            "before": text[:200],
            "after": bullet_converted[:200]
        })
    text = bullet_converted

    # 10. Normalize multiple punctuations (!!! → .)
    punct_norm = re.sub(r'[!?]{2,}', '.', text)
    punct_norm = re.sub(r'\.{2,}', '.', punct_norm)
    if punct_norm != text:
        changes_log["punctuation_normalization"].append({
            "before": text[:200],
            "after": punct_norm[:200]
        })
    text = punct_norm

    # 11. Fix Unicode ligatures (ﬁ → fi, ﬂ → fl)
    ligature_fixed = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    if ligature_fixed != text:
        changes_log["unicode_ligature_fix"].append({
            "before": text[:200],
            "after": ligature_fixed[:200]
        })
    text = ligature_fixed

    # 12. Clean extra spaces/newlines
    cleaned_text = re.sub(r"\s{2,}", " ", text).strip()
    if cleaned_text != text:
        changes_log["extra_space_clean"].append({
            "before": text[:200],
            "after": cleaned_text[:200]
        })
    text = cleaned_text

    return text, changes_log

# ---------------- TEXT + TABLE EXTRACTION ----------------
def extract_text_with_tables(pdf_path):
    """
    Extract page text using LangChain (PyPDFLoader) and tables using Camelot.
    Merge tables into text in reading order (tables appended after text of that page).
    """
    # Load text with LangChain
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Extract tables using Camelot
    tables_by_page = {}
    try:
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')  # use 'stream' if borderless
        print(tables)
        for table in tqdm(tables, desc="Extracting Tables"):
            page_num = table.page
            # Convert table to Markdown-like text
            table_text = "\nTable:\n" + "\n".join([" | ".join(row) for row in table.df.values.tolist()])
            tables_by_page.setdefault(page_num, []).append(table_text)
    except Exception as e:
        print(f"No tables detected or error extracting tables: {e}")

    # Combine text + tables per page
    combined_pages = []
    for doc in tqdm(docs, desc="Extracting Text"):
        page_num = doc.metadata['page'] + 1
        text_content = doc.page_content.strip()

        if page_num in tables_by_page:
            for table_text in tables_by_page[page_num]:
                text_content += "\n" + table_text

        print(f'Page No. {page_num}')
        text_content, logs = clean_text(text_content)  # Assume you have clean_text implemented
        print(logs)
        
        combined_pages.append({
            "page_num": page_num,
            "content": text_content
        })

    return combined_pages, logs


def extract_images_pymupdf(pdf_path, images_output_dir):
    """
    Extract inline figures/images from PDF using PyMuPDF.
    """
    os.makedirs(images_output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)

    image_map = {}
    for page_num, page in enumerate(doc, start=1):
        images = page.get_images(full=True)
        image_list = []

        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            image_filename = f"{os.path.basename(pdf_path).replace('.pdf','')}_page{page_num}_img{img_index+1}.{image_ext}"
            image_path = os.path.join(images_output_dir, image_filename)
            image_path = os.path.relpath(image_path).replace("\\", "/")

            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)

            image_list.append(image_path)

        image_map[page_num] = image_list

    return image_map


def extract_images_with_captions(pdf_path, output_dir=IMAGE_CAPTIONS_DIR, caption_lines=3):
    """
    Extract images and nearby captions from PDF pages.
    Captions are detected by finding text near image rectangles.
    
    Args:
        pdf_path (str): Path to PDF file.
        output_dir (str): Directory to save extracted images.
        caption_lines (int): Number of text lines near image to consider as caption.

    Returns:
        dict: {page_num: [ {image_path, caption_text} ]}
    """
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    caption_map = {}
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    for page_index, page in enumerate(doc, start=1):
        # Extract text blocks for caption proximity
        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
        text_blocks = [b for b in blocks if b[4].strip()]

        # Extract images on this page
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # Save image
            image_filename = f"{pdf_name}_page{page_index}_img{img_index+1}.{image_ext}"
            image_path = os.path.join(output_dir, image_filename)
            with open(image_path, "wb") as f:
                f.write(image_bytes)

            # Get image rectangle
            try:
                rect = page.get_image_rects(xref)[0]  # bounding box
            except:
                rect = None

            caption_text = ""
            if rect:
                # Sort text blocks by vertical position
                text_blocks_sorted = sorted(text_blocks, key=lambda b: b[1])  # y0
                caption_candidates = []
                for b in text_blocks_sorted:
                    x0, y0, x1, y1, text, *_ = b
                    # Text above or below image within ~100px
                    if (y0 >= rect.y1 and y0 - rect.y1 < 100) or (rect.y0 - y1 < 100 and y1 <= rect.y0):
                        caption_candidates.append(text)

                caption_text = " ".join(caption_candidates[:caption_lines])

            # Clean caption text
            caption_text = re.sub(r"Figure\s*\d+[:.]?", "", caption_text, flags=re.IGNORECASE).strip()
            caption_text = re.sub(r"\s{2,}", " ", caption_text)

            # Save into map
            caption_map.setdefault(page_index, []).append({
                "image_path": os.path.relpath(image_path).replace("\\", "/"),
                "caption_text": caption_text if caption_text else "No caption detected"
            })

    return caption_map

def extract_full_page_images(pdf_path, page_output_dir=PAGE_IMAGES_DIR):
    """
    Save each full page of the PDF as an image (snapshot).
    """
    os.makedirs(page_output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    page_snapshot_map = {}
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # High-res snapshot
        image_filename = f"{pdf_name}_page{page_num+1}_snapshot.png"
        image_path = os.path.join(page_output_dir, image_filename)
        pix.save(image_path)

        # Store relative path
        page_snapshot_map[page_num + 1] = os.path.relpath(image_path).replace("\\", "/")

    return page_snapshot_map