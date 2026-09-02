"""One-off script: process and embed each knowledge-base file sequentially,
inserting into Chroma immediately after each file finishes parsing. This way
progress is incremental -- if something fails partway, earlier files are
already safely embedded instead of being lost.
"""
from pathlib import Path

import vector_embed as ve

files = [p for p in Path(ve.KB_DIR).rglob("*") if p.suffix.lower() in (".pdf", ".docx", ".xlsx", ".xls")]

# .md files are not picked up by process_single_file's docling path in the
# original loader; ingest_knowledge_base() only walks .pdf/.docx/.xlsx/.xls
# via load_concurrently_multi_format, matching the original behavior.

print(f"Files to process: {len(files)}", flush=True)

for i, path in enumerate(files, 1):
    print(f"\n=== [{i}/{len(files)}] {path.name} ===", flush=True)
    docs = ve.process_single_file(path)
    if docs:
        ve.add_in_batches(docs)
        print(f"[{i}/{len(files)}] {path.name}: {len(docs)} chunks embedded and stored.", flush=True)
    else:
        print(f"[{i}/{len(files)}] {path.name}: FAILED, no chunks produced.", flush=True)

print("\nAll files processed.", flush=True)
