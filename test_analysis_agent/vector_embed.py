##################-----------PACKAGES-----------###################
###################################################################
import os
import re
import time
import logging

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool
from langchain_core.tools import tool
from langchain_docling.loader import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

import debug_artifacts
import rag_debug
import retrieval_debug
from rag_debug import C, _c, field, section, status

log = logging.getLogger("vector_embed")


##################-----------MACROS-----------###################
###################################################################
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(AGENT_DIR, "knowledge_base")
USER_UPLOADS_DIR = os.path.join(AGENT_DIR, "uploads")
USER_COLLECTIONS_DIR = os.path.join(AGENT_DIR, "chromadb", "user_collections")
OCR_MIN_CHAR_PER_PAGE = 50
EMBED_BATCH_SIZE = 32
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

# Map user-facing file names to the on-disk copies in USER_UPLOADS_DIR so
# tools like get_document_structure can resolve them without the original
# attachment id. Populated by bridge.download_file via register_upload().
_UPLOAD_REGISTRY: dict[str, str] = {}


def register_upload(file_name: str, path: Path) -> None:
    """Record the on-disk location of an uploaded file for later tools."""
    _UPLOAD_REGISTRY[file_name] = str(path)


def resolve_upload(file_name: str) -> str | None:
    """Return the on-disk path for an uploaded file name, or None."""
    if file_name in _UPLOAD_REGISTRY:
        return _UPLOAD_REGISTRY[file_name]
    candidates = list(Path(USER_UPLOADS_DIR).rglob(file_name))
    return str(candidates[0]) if candidates else None


######--------models---------#############

embeddings = OllamaEmbeddings(model="nomic-embed-text")  # embedding model (loosely fetches the top 15-20 relevant chunks)
reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")  # reranker(keeps the top 5 most relevant docs after reranking)
compressor = CrossEncoderReranker(model=reranker_model, top_n=25)


###############--------KNOWLEDGE BASE DOCS----------###########
DOCS = [
    ("Environmental_and_hardware", "MIL-STD-461.pdf",               "MIL-STD-461"),
    ("Environmental_and_hardware", "MIL-STD-810H_CHG-1.pdf",        "MIL-STD-810H_CHG-1"),
    ("Environmental_and_hardware", "MIL-STD-1586A.pdf",             "MIL-STD-1586A"),
    ("Requirements_and_quality",   "830-1998.pdf",                  "830-1998"),
    ("Requirements_and_quality",   "15288-2023-2.pdf",              "15288-2023-2"),
    ("Requirements_and_quality",   "29119-1-2022.pdf",              "29119-1-2022"),
    ("Requirements_and_quality",   "29148-2018.pdf",                "29148-2018"),
    ("Requirements_and_quality",   "IEEE-Test-Doc-829-2008.pdf",    "IEEE-Test-Doc-829-2008"),
    ("Requirements_and_quality",   "ISO-9001-2015.pdf",             "ISO-9001-2015"),
    ("Requirements_and_quality",   "requirements_and_testing.md",   "requirements_and_testing"),
    ("Security_and_safety",        "MIL-STD-882E.pdf",              "MIL-STD-882E"),
    ("Security_and_safety",        "RTCA-DO-160G.pdf",              "RTCA-DO-160G"),
    ("Security_and_safety",        "SP800-53_REV-3.PDF",            "SP800-53_REV-3"),
]

DOC_METADATA_LOOKUP = {
    filename: {"category": category, "standard": standard_label}
    for category, filename, standard_label in DOCS
}


##############-----tokenizer for docling loader----------########
hf_tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True),
    max_tokens=8192,
)


################-------docling loader for single file-------########
_DOC_CONVERTER = None

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



def _get_docling_converter():
    """Shared DocumentConverter so the markdown debug export and the chunking
    loader reuse the same loaded pipeline/models."""
    global _DOC_CONVERTER
    if _DOC_CONVERTER is None:
        from docling.document_converter import DocumentConverter

        _DOC_CONVERTER = DocumentConverter()
    return _DOC_CONVERTER


def _export_markdown_debug(path: Path) -> None:
    """Log file facts and export the intermediate Docling markdown to
    ./debug_output/markdown/ (DEBUG_MODE only)."""
    import mimetypes

    section("INGESTION", f"File ingestion: {path.name}", C.INGESTION)
    field("filename", path.name)
    field("file_size_bytes", path.stat().st_size)
    field("mime_type", mimetypes.guess_type(path.name)[0] or "unknown")
    field("started_at", rag_debug.now_iso())

    start = time.perf_counter()
    result = _get_docling_converter().convert(str(path))
    markdown = result.document.export_to_markdown()
    duration_s = round(time.perf_counter() - start, 3)
    info = debug_artifacts.save_markdown_export(path.stem, markdown)
    field("conversion_duration_s", duration_s)
    field("md_char_count", info["chars"])
    field("md_saved_path", info["path"])


def _chunk_stats(source_name: str, docs: list) -> dict:
    """Log HybridChunker stats + first-N chunk previews, persist a JSON dump."""
    start = time.perf_counter()
    char_counts = [len(d.page_content or "") for d in docs]
    try:
        token_counts = [
            len(hf_tokenizer.tokenizer.encode(d.page_content or "")) for d in docs
        ]
    except Exception:  # noqa: BLE001 - tokenizer failure must not break ingest
        token_counts = []
    duration_s = round(time.perf_counter() - start, 4)

    section("CHUNKING", f"HybridChunker output for {source_name}", C.CHUNKING)
    field("total_chunks", len(docs))
    if char_counts:
        field("avg_chunk_chars", round(sum(char_counts) / len(char_counts), 1))
        field("min_max_chars", f"{min(char_counts)} / {max(char_counts)}")
    if token_counts:
        field("avg_chunk_tokens", round(sum(token_counts) / len(token_counts), 1))
    field("chunking_duration_s", duration_s)

    dump_path = debug_artifacts.save_chunk_dump(source_name, docs)
    if dump_path:
        field("chunk_dump", dump_path)

    if rag_debug.VERBOSE_CHUNKS:
        nl = chr(10)
        for idx in range(min(rag_debug.CHUNK_PREVIEW_COUNT, len(docs))):
            text = docs[idx].page_content or ""
            head = text[: rag_debug.PREVIEW_HEAD_CHARS].replace(nl, " ")
            tail = (
                text[-rag_debug.PREVIEW_TAIL_CHARS :].replace(nl, " ")
                if len(text) > rag_debug.PREVIEW_HEAD_CHARS
                else ""
            )
            tok = token_counts[idx] if idx < len(token_counts) else "?"
            print(f"  {_c(C.CHUNKING)}[{idx}]{_c(C.RESET)} chars={len(text)} tokens={tok}")
            print(f"      metadata: {docs[idx].metadata}")
            print(f"      head: {head!r}")
            if tail:
                print(f"      tail: {tail!r}")
    return {"total_chunks": len(docs), "duration_s": duration_s}

def _extract_section(meta: dict) -> str | None:
    headings = meta.get("dl_meta", {}).get("headings")
    if headings:
        return headings[-1] if isinstance(headings, list) else str(headings)
    return None

def _extract_page(meta: dict) -> str | None:
    doc_items = meta.get("dl_meta", {}).get("doc_items", [])
    pages = sorted({
        p.get("page_no")
        for item in doc_items
        for p in item.get("prov", [])
        if p.get("page_no") is not None
    })
    if not pages:
        return None
    if len(pages) == 1:
        return str(pages[0])
    if pages[-1] - pages[0] <= 2:
        return f"{pages[0]}-{pages[-1]}"
    return None  # too wide to be a useful page citation, rely on section instead

def process_single_file(path: Path):
    started = time.perf_counter()
    try:
        if rag_debug.DEBUG_MODE:
            _export_markdown_debug(path)
        loader = DoclingLoader(
            file_path=str(path),
            export_type=ExportType.DOC_CHUNKS,
            chunker=HybridChunker(tokenizer=hf_tokenizer, max_tokens=512),
        )
        docs = loader.load()

        file_info = DOC_METADATA_LOOKUP.get(path.name, {})
        stem = re.sub(r"[^A-Za-z0-9]+", "-", path.stem).strip("-")

        for i, doc in enumerate(docs):
            doc.metadata["source_file"] = path.name
            if "category" in file_info:
                doc.metadata["category"] = file_info["category"]
            if "standard" in file_info:
                doc.metadata["standard"] = file_info["standard"]

            section = _extract_section(doc.metadata)
            page = _extract_page(doc.metadata)
            tag = f"{stem}-{i+1:03d}"
            doc.id = tag
            doc.metadata["citation_tag"] = tag
            doc.metadata["section"] = section
            doc.metadata["page"] = page
            doc.page_content = (
                f'<chunk id="{tag}" source="{path.name}" '
                f'standard="{doc.metadata.get("standard", "")}" '
                f'category="{doc.metadata.get("category", "")}" '
                f'section="{section or ""}" '
                f'page="{page or ""}">'
                f"{doc.page_content}</chunk>"
            )

        _chunk_stats(path.name, docs)
        status(
            "ok",
            "INGESTION",
            f"{path.name}: {len(docs)} chunks in {round(time.perf_counter() - started, 2)}s",
        )
        return docs
    except Exception as e:
        status("err", "INGESTION", f"{path.name} failed: {e}")
        log.error("docling load failed", extra={"stage": "kb_parse", "meta": {"file": path.name, "error": str(e)}})
        return []


###########--------concurrently load the files to docling---------#########
def load_concurrently_multi_format(directory_path: str, max_workers: int = 4, extensions: tuple = (".pdf", ".docx", ".xlsx", ".xls")):

    dir_path = Path(directory_path)
    file_paths = [p for p in dir_path.rglob("*") if p.suffix.lower() in extensions]

    all_documents = []
    log.info("concurrent load started", extra={"stage": "kb_ingest", "meta": {"files": len(file_paths)}})

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(process_single_file, path): path for path in file_paths}

        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                docs = future.result()
                all_documents.extend(docs)
                log.info("file loaded", extra={"stage": "kb_ingest", "meta": {"file": path.name, "chunks": len(docs)}})
            except Exception as exc:
                log.error("file generated an exception", extra={"stage": "kb_ingest", "meta": {"file": path.name, "error": str(exc)}})

    return all_documents


###########-------knowledge base vector store---------##########
vector_store = Chroma(
    collection_name="iso_files",
    embedding_function=embeddings,
    persist_directory=os.path.join(AGENT_DIR, "chromadb"),
)


###########--------add chunks to the vector database--------#############
def _add_with_storage_debug(store, collection_name: str, documents: list) -> list:
    """Insert documents into ``store`` with storage-integrity logging:
    count before/after + a JSON snapshot (ids/metas/document previews)."""
    try:
        count_before = store._collection.count()
    except Exception:  # noqa: BLE001 - introspection is best-effort
        count_before = None

    inserted_ids: list[str] = []
    metas: list[dict] = []
    texts: list[str] = []
    for i in range(0, len(documents), EMBED_BATCH_SIZE):
        batch = filter_complex_metadata(documents[i : i + EMBED_BATCH_SIZE])
        ids = store.add_documents(documents=batch)
        inserted_ids.extend(ids)
        metas.extend(d.metadata for d in batch)
        texts.extend(d.page_content or "" for d in batch)

    try:
        count_after = store._collection.count()
    except Exception:  # noqa: BLE001
        count_after = None

    section("STORAGE", f"ChromaDB insert into '{collection_name}'", C.STORAGE)
    field("collection", collection_name)
    field("count_before", count_before)
    field("inserted", len(inserted_ids))
    field("count_after", count_after)
    snapshot_path = debug_artifacts.save_chroma_snapshot(
        collection_name, inserted_ids, metas, texts, count_before, count_after
    )
    field("snapshot", snapshot_path)
    status(
        "ok",
        "STORAGE",
        f"'{collection_name}': {count_before} -> {count_after} documents",
    )
    return inserted_ids


def add_in_batches(chunks, label="knowledge_base"):
    """Backwards-compatible wrapper; routes through the instrumented insert."""
    _add_with_storage_debug(
        vector_store, "iso_files" if label == "knowledge_base" else label, chunks
    )


###########-------add embeddings to the database-----------##########
def ingest_knowledge_base():
    docs = load_concurrently_multi_format(KB_DIR)
    if docs:
        log.info("embedding knowledge base", extra={"stage": "kb_ingest", "meta": {"chunks": len(docs)}})
        add_in_batches(docs)
    else:
        log.warning("no docs were loaded", extra={"stage": "kb_ingest"})


##########-------so that it wont do the embedding everytime-------##########
if __name__ == "__main__":
    ingest_knowledge_base()


########--------retrieval of the related chunks----------###########
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 20})

# Logged retriever: same base MMR retriever + reranker, but every query logs
# candidates, distances, pass/filter verdicts and the final context.
kb_compression_retriever = retrieval_debug.logged_compression_retriever(
    retriever, compressor, tag="search_testing_standards"
)
retriever_tool = create_retriever_tool(
    kb_compression_retriever,
    name="search_testing_standards",
    description=(
        "Search internal standards documents for systems engineering processes, testing, "
        "safety, cybersecurity, environmental/hardware qualification, and requirements "
        "engineering guidance. Call this tool for ANY question about the content, "
        "structure, process groups, definitions, or requirements of a standard — not "
        "just when generating test cases.\n\n"
        "Each result is wrapped in a <chunk id=\"...\" source=\"...\" standard=\"...\" "
        "category=\"...\" section=\"...\" page=\"...\"> tag. When citing information, "
        "cite it in human-readable form as \"(standard, Section X)\" or "
        "\"(standard, p. N)\" — e.g. (MIL-STD-461, Section 4.1.3) — using only the "
        "standard/section/page values that appeared in a chunk you actually "
        "retrieved. Never invent, guess, or reuse a section, clause, or page that "
        "was not present in the retrieved text. Do not cite the internal chunk id.\n\n"
        "Categories available:\n"
        "- Environmental_and_hardware (MIL-STD environmental and hardware qualification standards)\n"
        "- Requirements_and_quality (systems lifecycle, requirements engineering, and quality standards)\n"
        "- Security_and_safety (military/NIST safety and cybersecurity standards)\n\n"
        "Always call this tool before answering any question about standard content, "
        "and before generating test cases or validating a requirement."
        ),
)


#########--------user document loading according to the doc type---------############
def _load_binary_with_docling(file_path: str, **loader_kwargs) -> list[Document]:
    loader = DoclingLoader(
        file_path=file_path,
        export_type=ExportType.DOC_CHUNKS,
        chunker=HybridChunker(tokenizer=hf_tokenizer, max_tokens=512),
        **loader_kwargs,
    )
    return loader.load()

def _tag_chunks(docs: list[Document], source_name: str) -> list[Document]:
    stem = re.sub(r"[^A-Za-z0-9]+", "-", os.path.splitext(source_name)[0]).strip("-").lower()
    for i, doc in enumerate(docs):
        section = _extract_section(doc.metadata)
        page = _extract_page(doc.metadata)
        tag = f"{stem}-{i+1:03d}"
        doc.id = tag

        doc.metadata["section"] = section
        doc.metadata["page"] = page
        doc.metadata["citation_tag"] = tag
        doc.page_content = (
            f'<chunk id="{tag}" source="{source_name}" '
            f'section="{section or ""}" '
            f'page="{page or ""}">'
            f"{doc.page_content}</chunk>"
        )
    return docs


def load_document(file_path: str) -> list[Document]:
    ext = os.path.splitext(file_path)[1].lower()
    source_name = os.path.basename(file_path)

    if ext == ".csv":
        log.info("loading csv", extra={"stage": "user_parse", "meta": {"file": file_path}})
        docs = CSVLoader(file_path, autodetect_encoding=True).load()
    elif ext in (".txt", ".md"):
        log.info("loading text", extra={"stage": "user_parse", "meta": {"file": file_path}})
        docs = TextLoader(file_path, autodetect_encoding=True).load()
    else:
        # Binary formats: try plain Docling first (fast path for text-based PDFs).
        docs = _load_binary_with_docling(file_path)
        total_chars = sum(len(d.page_content or "") for d in docs)
        if total_chars < OCR_MIN_CHAR_PER_PAGE:
            log.info(
                "low text extraction, retrying with OCR",
                extra={"stage": "user_parse", "meta": {"file": file_path, "chars": total_chars}},
            )
            docs = _load_binary_with_docling(file_path, convert_kwargs={"ocr": True})

    docs = _tag_chunks(docs, source_name)

    log.info(
        "document parsed",
        extra={"stage": "user_parse", "meta": {"file": file_path, "chunks": len(docs)}},
    )
    return docs


#########--------make the chroma collection name valid---------###########
def sanitize_collection_name(name: str) -> str:
    """Turn an arbitrary string into a valid Chroma collection name:
    3-255 chars, only letters/digits/underscores/hyphens, and must start
    and end with a letter or digit."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", str(name))
    cleaned = cleaned.strip("_-") or "doc"
    return cleaned[:255]


def build_session_retriever_tool(
    file_paths: list[str],
    session_id: str,
    collection_suffix: str | None = None,
    k: int = 5,
) -> tuple[object, dict]:
    """
    Index user-uploaded files into a per-session Chroma collection and
    return ``(retriever_tool, ingest_report)``.

    Isolation: every session (thread) gets its own collection named after the
    session id (+ an optional suffix so a changed attachment set builds a
    fresh collection instead of reusing/corrupting the previous one).

    Persistence: collections live under ``chromadb/user_collections``, so the
    vectors (and the file=>collection mapping) survive process restarts.

    Ingest report: ``{"collection", "files_indexed", "failed_files",
    "chunk_count"}`` lets bridge.py surface partial failures to the client
    instead of silently returning a tool-less agent.
    """
    failed_files: list[tuple[str, str]] = []
    documents: list[Document] = []

    for file_path in file_paths:
        try:
            docs = load_document(file_path)
            documents.extend(docs)
            for doc in docs:
                doc.metadata.setdefault("source_file", os.path.basename(file_path))
                doc.metadata["session_id"] = session_id
            log.info(
                "user file parsed",
                extra={"stage": "user_ingest", "meta": {"file": file_path, "chunks": len(docs)}},
            )
        except Exception as exc:
            log.error(
                "user file failed to load",
                extra={"stage": "user_ingest", "meta": {"file": file_path, "error": str(exc)}},
            )
            failed_files.append((os.path.basename(file_path), str(exc)))

    if not documents:
        raise RuntimeError(
            f"No text could be extracted from any uploaded file. Failures: {failed_files}"
        )

    base_name = f"user_{session_id}"
    if collection_suffix:
        base_name = f"{base_name}_{collection_suffix}"
    collection_name = sanitize_collection_name(base_name)

    session_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=USER_COLLECTIONS_DIR,
    )
    _add_with_storage_debug(session_store, collection_name, documents)

    session_retriever = session_store.as_retriever(search_type="similarity", search_kwargs={"k": 50})
    session_compression_retriever = retrieval_debug.logged_compression_retriever(
        session_retriever, compressor, tag="search_user_document"
    )
    session_tool = create_retriever_tool(
        session_compression_retriever,  # Use the reranker here
        name="search_user_document",
        description=(
            "Search the document(s) uploaded by the user to find specific project "
            "details, requirements, architecture, or other content described in "
            "their files. Call this tool before answering any question about the "
            "uploaded document's content, and before generating test cases or "
            "validating a requirement against it.\n\n"
            "Each result is wrapped in a <chunk id=\"...\" source=\"...\" "
            "section=\"...\" page=\"...\"> tag. When citing information, cite it in "
            "human-readable form as \"(filename, Section X)\" or \"(filename, p. N)\" "
            "using only the source/section/page values that appeared in a chunk you "
            "actually retrieved — never invent, guess, or reuse a section or page "
            "that was not present in the retrieved text. Do not cite the internal "
            "chunk id. If section or page is missing, cite by filename alone. If a "
            "claim isn't backed by a chunk you retrieved, say the document does not "
            "specify it rather than answering from assumption or general knowledge."
        ),
    )

    report = {
        "collection": collection_name,
        "files_indexed": [os.path.basename(p) for p in file_paths
                          if os.path.basename(p) not in {f for f, _ in failed_files}],
        "failed_files": failed_files,
        "chunk_count": len(documents),
    }
    log.info(
        "session indexed",
        extra={"stage": "user_ingest", "meta": report},
    )
    return session_tool, report


@tool
def get_document_structure(file_name: str) -> str:
    """Use this tool FIRST to understand the overarching structure, Table of Contents,
    and general scope of a user-uploaded document.
    Provide the exact file name."""
    resolved = resolve_upload(file_name)
    if resolved is None:
        return (
            f"File {file_name} not found in the upload directory. "
            f"Searched {USER_UPLOADS_DIR}."
        )
    try:
        docs = load_document(resolved)
        # Heuristic: the TOC and intro are almost always in the first 5 chunks.
        intro_pages = docs[:5]
        structure_text = f"--- Document Structure / Intro for {file_name} ---\n"
        for page in intro_pages:
            structure_text += page.page_content + "\n"
        return structure_text
    except Exception as e:
        return f"Failed to extract structure: {str(e)}"


tools = [retriever_tool, get_document_structure]
