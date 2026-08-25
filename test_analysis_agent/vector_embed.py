"""Knowledge base ingestion and retrieval. The retriever built here combines
query expansion (MultiQueryRetriever), cross-encoder re-ranking, and MMR
candidate selection to handle ambiguous/multi-step questions -- see
build_expanded_retriever / build_reranking_retriever below and the
"RAG pipeline" section in README.md. Query reformulation and multi-hop
retrieval across standards are handled at the agent level (agent.py's
system prompt), not here."""

import os
import re
import glob

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.tools import tool
from model import get_model
from langchain_docling.loader import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer


AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(AGENT_DIR, "knowledge_base")
OCR_MIN_CHAR_PER_PAGE = 50
EXPORT_TYPE = ExportType.MARKDOWN

def sanitize_collection_name(name: str) -> str:
    """Turn an arbitrary string into a valid Chroma collection name:
    3-255 chars, only letters/digits/underscores/hyphens, and must start
    and end with a letter or digit."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", str(name))
    cleaned = cleaned.strip("_-") or "doc"
    return cleaned[:255]

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

DOCS = [
    ("01_se_process_and_requirements", "15288-2023-2.pdf",             "ISO/IEC/IEEE 15288:2023"),
    ("02_verification_and_testing",    "IEEE-Test-Doc-829-2008.pdf",   "IEEE 829-2008"),
    ("03_safety_security_config",      "MIL-STD-1586A.pdf",            "MIL-STD-1586A"),
    ("03_safety_security_config",      "NIST_SP_800-171A.pdf",         "NIST SP 800-171A"),
    ("03_safety_security_config",      "SP800-53_REV-3.PDF",           "SP 800-53 Rev.3"),
    ("04_requirements",                "requirements_engineering.txt", "requirements_engineering")
]

DOC_METADATA_LOOKUP = {
    filename: {"category": category, "standard": standard_label} 
    for category, filename, standard_label in DOCS
}
hf_tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True),
    max_tokens=8192,
)

# Loaded once at import time (not per-query) -- a small local cross-encoder
# used to rerank a wider MMR candidate set down to the final top-k, instead
# of trusting embedding-similarity + MMR diversity alone.
RERANK_CANDIDATE_MULTIPLIER = 4
cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")


def build_reranking_retriever(base_vector_store: Chroma, k: int, search_type: str = "mmr"):
    """Retrieve a wider candidate set (k * RERANK_CANDIDATE_MULTIPLIER) via the
    base vector store, then rerank down to the final top-k with a local
    cross-encoder -- retrieve-then-rerank, cheaper and more accurate than
    relying on embedding similarity/MMR diversity alone."""
    base_retriever = base_vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k * RERANK_CANDIDATE_MULTIPLIER},
    )
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=k)
    return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base_retriever)


def build_expanded_retriever(base_vector_store: Chroma, k: int, search_type: str = "mmr"):
    """Reranked retriever (build_reranking_retriever) wrapped in query
    expansion/reformulation: the shared local LLM generates a couple of
    alternate phrasings of the query, each is retrieved+reranked separately,
    and the results are merged and de-duplicated. Improves recall on
    ambiguous or oddly-phrased questions at the cost of a few extra (local,
    already-loaded-model) LLM calls per turn."""
    reranked_retriever = build_reranking_retriever(base_vector_store, k, search_type=search_type)
    return MultiQueryRetriever.from_llm(
        retriever=reranked_retriever,
        llm=get_model(),
        include_original=True,
    )


def process_single_file(path: Path):
    try:
        loader = DoclingLoader(
            file_path=str(path), 
            export_type=ExportType.DOC_CHUNKS, # Ensure export type matches chunking needs
            chunker=HybridChunker(tokenizer=hf_tokenizer, max_tokens=512)
        )
        docs = loader.load()
        
        # Look up the metadata for this specific file
        file_info = DOC_METADATA_LOOKUP.get(path.name, {})
        
        # Inject metadata
        for doc in docs:
            doc.metadata["source_file"] = path.name
            if "category" in file_info:
                doc.metadata["category"] = file_info["category"]
            if "standard" in file_info:
                doc.metadata["standard"] = file_info["standard"]
                
        return docs
    except Exception as e:
        print(f"Error on {path.name}: {e}")
        return []
    


def load_concurrently_multi_format(directory_path: str, max_workers: int = 4, extensions: tuple = (".pdf", ".docx", ".xlsx", ".xls")): 
    
    dir_path = Path(directory_path)
    file_paths = [p for p in dir_path.rglob("*") if p.suffix.lower() in extensions]
    
    all_documents = []
    print(f"Starting concurrent load for {len(file_paths)} files...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map the futures to their file paths
        future_to_path = {executor.submit(process_single_file, path): path for path in file_paths}
        
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                docs = future.result()
                all_documents.extend(docs)
                print(f"Successfully loaded {path.name} ({len(docs)} chunks)")
            except Exception as exc:
                print(f"{path.name} generated an exception: {exc}")
                
    return all_documents


# def ocr_page(pdf_path, page_number):
#     from pdf2image import convert_from_path
#     import pytesseract
#     images = convert_from_path(pdf_path, first_page=page_number+1, last_page=page_number+1)
#     return pytesseract.image_to_string(images[0])

# def load_with_ocr(path, category, standard_label):
#     pages = PDFPlumberLoader(path).load()
#     for i, page in enumerate(pages):
#         if len(page.page_content.strip()) < OCR_MIN_CHAR_PER_PAGE:
#             try:
#                 page.page_content = ocr_page(path, i)
#             except Exception as e:
#                 print(f"OCR failed for {path} page {i}: {e}")
#         page.metadata["category"] = category
#         page.metadata["standard"] = standard_label
#         page.metadata["source_file"] = os.path.basename(path)
#     return pages

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
from langchain_community.vectorstores.utils import filter_complex_metadata
def add_in_batches(chunks, label="knowledge_base"):
    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i:i + EMBED_BATCH_SIZE]
        batch = filter_complex_metadata(batch) 
        vector_store.add_documents(documents=batch)
    
        print(f"  embedded {label} batch {i // EMBED_BATCH_SIZE + 1} "
              f"({i + len(batch)}/{len(chunks)})")


def ingest_knowledge_base():
    docs = load_concurrently_multi_format(KB_DIR)
    if docs:
        print(f"Embedding a total of {len(docs)} chunks into Chroma...")
        add_in_batches(docs)
    else:
        print("no docs were loaded")

if __name__ == "__main__":
    ingest_knowledge_base()

# if not vector_store.get()["ids"]:
#     for category, filename, standard_label in DOCS:
#         full_path = os.path.join(KB_DIR, category, filename)
#         if not os.path.exists(full_path):
#             print(f"WARNING: missing file {full_path}, skipping")
#             continue
#         pages = load_with_ocr(full_path, category, standard_label)
#         chunks = text_splitter_kb.split_documents(pages)
#         print(f"Embedding {filename} ({len(chunks)} chunks)...")
#         add_in_batches(chunks, label=filename)
retriever = build_expanded_retriever(vector_store, k=5)

retriever_tool = create_retriever_tool(
    retriever,
    name="search_testing_standards",
    description=(
        "Search internal standards documents for requirements, testing, hardware "
        "qualification, safety, security guidance. Call this tool for ANY question "
        "about the content, structure, process groups, definitions, or requirements "
        "of a standard — not just when generating test cases. "
        "Each result includes 'standard' and 'category' metadata — always cite the "
        "'standard' field, never invent a clause number not present in the retrieved text. "
        "Categories: 01_se_process_and_requirements (lifecycle/requirements standards), "
        "02_verification_and_testing (test design/documentation standards), "
        "03_hardware_environmental (EMI/environmental/hardware bus standards), "
        "04_safety_security_config (safety, cybersecurity, configuration control"
        "05_quality (quality management). "
        "Always call this before answering any question about standard content, and "
        "before generating test cases or validating a requirement."
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
    session_retriever = build_expanded_retriever(session_store, k=k)

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