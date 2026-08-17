import os
import re

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import (
    TextLoader,
    PDFPlumberLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import create_retriever_tool
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore
from langchain_core.tools import tool

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(AGENT_DIR, "knowledge_base")
OCR_MIN_CHAR_PER_PAGE = 50


def sanitize_collection_name(name: str) -> str:
    """Turn an arbitrary string into a valid Chroma collection name:
    3-255 chars, only letters/digits/underscores/hyphens, and must start
    and end with a letter or digit."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", str(name))
    cleaned = cleaned.strip("_-") or "doc"
    return cleaned[:255]

text_splitter_user_docs = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    length_function=len,
)
text_splitter_kb = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
)
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=300,
    length_function=len,
)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

DOCS = [
    ("01_se_process_and_requirements", "12207_2017-2008_redline.pdf", "ISO/IEC/IEEE 12207:2017"),
    ("01_se_process_and_requirements", "15288-2023-2.pdf",            "ISO/IEC/IEEE 15288:2023"),
    ("01_se_process_and_requirements", "29148-2018.pdf",              "ISO/IEC/IEEE 29148:2018"),
    ("02_verification_and_testing",    "29119-1-2022.pdf",            "ISO/IEC/IEEE 29119-1:2022"),
    ("02_verification_and_testing",    "DOT-FAA-AR-07-39.PDF",        "DOT/FAA/AR-07/39"),
    ("02_verification_and_testing",    "IEEE-Test-Doc-829-2008.pdf",  "IEEE 829-2008"),
    ("03_hardware_environmental",      "MIL-STD-810H_CHG-1.pdf",      "MIL-STD-810H CHG-1"),
    ("03_hardware_environmental",      "MIL-STD-1553C.pdf",           "MIL-STD-1553C"),
    ("03_hardware_environmental",      "RTCA-DO-160G.pdf",            "RTCA DO-160G"),
    ("04_safety_security_config",      "MIL-STD-882E.pdf",            "MIL-STD-882E"),
    ("04_safety_security_config",      "MIL-STD-1586A.pdf",           "MIL-STD-1586A"),
    ("04_safety_security_config",      "NIST_SP_800-171A.pdf",        "NIST SP 800-171A"),
    ("04_safety_security_config",      "SP800-53_REV-3.PDF",          "NIST SP 800-53 Rev.3"),
]

def ocr_page(pdf_path, page_number):
    from pdf2image import convert_from_path
    import pytesseract
    images = convert_from_path(pdf_path, first_page=page_number+1, last_page=page_number+1)
    return pytesseract.image_to_string(images[0])

def load_with_ocr(path, category, standard_label):
    pages = PDFPlumberLoader(path).load()
    for i, page in enumerate(pages):
        if len(page.page_content.strip()) < OCR_MIN_CHAR_PER_PAGE:
            try:
                page.page_content = ocr_page(path, i)
            except Exception as e:
                print(f"OCR failed for {path} page {i}: {e}")
        page.metadata["category"] = category
        page.metadata["standard"] = standard_label
        page.metadata["source_file"] = os.path.basename(path)
    return pages

vector_store = Chroma(
    collection_name="iso_files",
    embedding_function=embeddings,
    persist_directory=os.path.join(AGENT_DIR, "chromadb"),
)
# Only walk the KB and OCR pages if the persisted collection is actually
# empty. On every warm restart (e.g. `--watch`) the collection already has
# the ids on disk, so this check must run BEFORE the OCR/chunking work,
# not after building all_chunks -- otherwise every restart re-OCRs the
# entire knowledge base for nothing, which is what was stalling startup.
EMBED_BATCH_SIZE = 32

def add_in_batches(chunks, label=""):
    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i:i + EMBED_BATCH_SIZE]
        vector_store.add_documents(documents=batch)
        print(f"  embedded {label} batch {i // EMBED_BATCH_SIZE + 1} "
              f"({i + len(batch)}/{len(chunks)})")

if not vector_store.get()["ids"]:
    for category, filename, standard_label in DOCS:
        full_path = os.path.join(KB_DIR, category, filename)
        if not os.path.exists(full_path):
            print(f"WARNING: missing file {full_path}, skipping")
            continue
        pages = load_with_ocr(full_path, category, standard_label)
        chunks = text_splitter_kb.split_documents(pages)
        print(f"Embedding {filename} ({len(chunks)} chunks)...")
        add_in_batches(chunks, label=filename)
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})

retriever_tool = create_retriever_tool(
    retriever,
    name="search_testing_standards",
    description=(
        "Search internal standards documents for requirements, testing, hardware "
        "qualification, safety, security, and quality guidance. "
        "Each result includes 'standard' and 'category' metadata — always cite the "
        "'standard' field, never invent a clause number not present in the retrieved text. "
        "Categories: 01_se_process_and_requirements (lifecycle/requirements standards), "
        "02_verification_and_testing (test design/documentation standards), "
        "03_hardware_environmental (EMI/environmental/hardware bus standards), "
        "04_safety_security_config (safety, cybersecurity, configuration control — note "
        "NIST SP 800-53 here is Rev.3, outdated vs. current Rev.5), "
        "05_quality (quality management). "
        "Always call this before generating test cases or validating a requirement."
    )
)
def load_pdf(file_path: str) -> list[Document]:
    pages = PDFPlumberLoader(file_path).load()
    empty_pages = [i for i, p in enumerate(pages) if len(p.page_content.strip()) < OCR_MIN_CHAR_PER_PAGE]
    if len(empty_pages) > len(pages)*0.5:
        raise ValueError(
            f"'{os.path.basename(file_path)}' appears to be a scanned/image-only PDF "
            f"with no extractable text. Please upload a text-based PDF, or convert it "
            f"with OCR first."
        )
    return pages


def load_document(file_path: str) -> list[Document]:
    ext = os.path.splitext(file_path)[1].lower() #extract the document' extension

    if ext == ".pdf":
        return load_pdf(file_path)
    if ext == ".docx":
        return UnstructuredWordDocumentLoader(file_path, mode="elements").load()
    if ext in (".xlsx", ".xls"):
        return UnstructuredExcelLoader(file_path, mode="elements").load()
    if ext == ".csv":
        return CSVLoader(file_path, autodetect_encoding=True).load()
    
    return TextLoader(file_path, autodetect_encoding=True).load() #if its anything else like .txt, .csv, or .md then send it directly


def build_session_retriever_tool(
    file_paths: list[str],
    session_id: str,
    collection_suffix: str | None = None,
    k: int = 5,
):
    """Index user-uploaded files into a session-scoped, in-memory Chroma
    collection and return a retriever tool the agent can call.

    No persist_directory => vectors live only in process memory, isolated from
    the internal standards store ('general memory').

    collection_suffix lets callers create a fresh in-memory collection when
    the attachment set changes, avoiding stale vectors in the Chroma client
    cache for the same thread.
    """

    failed_files: list[tuple[str, str]] = []
    documents: list[Document] = []
    for file_path in file_paths:
        try:
            documents.extend(load_document(file_path))
        except Exception as exc:
            print(f"[vector_embed] failed to load {file_path}: {exc}")
            failed_files.append((os.path.basename(file_path), str(exc)))

    if not documents:
        raise RuntimeError(
        f"No text could be extracted from any uploaded file. Failures: {failed_files}"
    )

    chunks = child_splitter.split_documents(documents)

    name = f"user_upload_{sanitize_collection_name(session_id)}"
    if collection_suffix:
        name = f"{name}_{collection_suffix}"

    session_store = Chroma(
        collection_name=name,
        embedding_function=embeddings,
    )
    session_store.add_documents(documents=chunks)
    session_retriever = session_store.as_retriever(search_type="mmr", search_kwargs={"k": k})

    return create_retriever_tool(
        session_retriever,
        name="search_user_document",
        description=(
            "Search the document(s) uploaded by the user. Use this tool to find "
            "specific project details, requirements, or architecture described "
            "in the user's uploaded files."
        ),
    )

@tool
def get_document_structure(file_name: str) -> str:
    """
    Use this tool FIRST to understand the overarching structure, Table of Contents, 
    and general scope of a user-uploaded document. 
    Provide the exact file name.
    """
    # Locate the file in your upload directory (adjust path logic as needed)
    file_path = os.path.join(AGENT_DIR, "uploads", file_name)
    
    if not os.path.exists(file_path):
        return f"File {file_name} not found."
    
    try:
        # Load the document
        docs = load_document(file_path)
        
        # Heuristic: The TOC and intro are almost always in the first 5 pages/chunks
        intro_pages = docs[:5] 
        
        structure_text = f"--- Document Structure / Intro for {file_name} ---\n"
        for page in intro_pages:
            structure_text += page.page_content + "\n"
            
        return structure_text
        
    except Exception as e:
        return f"Failed to extract structure: {str(e)}"
tools = [retriever_tool, get_document_structure]