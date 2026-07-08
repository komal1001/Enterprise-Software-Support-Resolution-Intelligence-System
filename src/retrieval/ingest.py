"""
Ingestion pipeline — run ONCE before starting the API server.

    python -m src.retrieval.ingest

What this does:
  Step 1 — LlamaParse  : sends 7 PDFs to LlamaParse cloud API → gets back clean Markdown
  Step 2 — MarkdownNodeParser : splits Markdown at every ## header → one node per section
                                tables stay intact in one node (SentenceSplitter would break them)
  Step 3 — SentenceSplitter   : any node longer than 2000 chars is further split into 512-token chunks
  Step 4 — VectorStoreIndex   : embeds every node with text-embedding-3-small → stores in PostgreSQL (pgvector)
  Step 5 — _save_nodes        : saves all nodes as JSON → nodes_cache.json (needed for BM25 at retrieval time)

Langfuse trace is logged so ingestion cost and latency are visible in the dashboard.
"""

import os
import json
from dotenv import load_dotenv

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter

from src.retrieval.rag import (
    _build_vector_store,
    _build_embed_model,
    NODES_CACHE,
    MAX_NODE_CHARS,
)

load_dotenv()

DOCS_DIR = "Documents"


# ---------------------------------------------------------------------------
# _save_nodes — persists nodes to JSON for BM25 at retrieval time
# PostgreSQL only stores float vectors. Python TextNode objects must be saved
# separately so BM25Retriever can reload them without re-parsing the PDFs.
# ---------------------------------------------------------------------------

def _save_nodes(nodes: list) -> None:
    data = [{"text": n.text, "metadata": n.metadata, "id_": n.id_} for n in nodes]
    with open(NODES_CACHE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# ingest_documents — main entry point
# ---------------------------------------------------------------------------

def ingest_documents() -> None:
    from src.observability.langfuse_client import get_langfuse, timer

    t0 = timer()
    lf = get_langfuse()
    trace = lf.start_observation(
        name="ingestion-pipeline",
        as_type="span",
        input={"docs_dir": DOCS_DIR, "num_pdfs": 7},
    )

    # Step 1 — LlamaParse: cloud PDF parser → Markdown with tables and image descriptions
    parser = LlamaParse(
        api_key=os.environ["LLAMA_CLOUD_API_KEY"],
        result_type="markdown",
    )
    reader = SimpleDirectoryReader(DOCS_DIR, file_extractor={".pdf": parser})
    documents = reader.load_data()

    # Step 2 — MarkdownNodeParser: split at ## headers, tables stay in one node
    markdown_parser = MarkdownNodeParser()
    nodes = markdown_parser.get_nodes_from_documents(documents)

    # Step 3 — SentenceSplitter fallback: split any node that exceeds MAX_NODE_CHARS
    sentence_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    final_nodes = []
    for node in nodes:
        if len(node.text) > MAX_NODE_CHARS:
            sub_nodes = sentence_splitter.get_nodes_from_documents([node])
            final_nodes.extend(sub_nodes)
        else:
            final_nodes.append(node)

    # Step 4 — embed every node and store vectors in PostgreSQL via pgvector
    vector_store    = _build_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model     = _build_embed_model()

    VectorStoreIndex(
        final_nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    # Step 5 — save nodes to JSON so BM25Retriever can reload them at query time
    _save_nodes(final_nodes)

    # Estimate embedding cost (Azure doesn't return token counts via LlamaIndex)
    # text-embedding-3-small = $0.02 per 1M tokens, ~4 chars per token
    total_chars      = sum(len(n.text) for n in final_nodes)
    estimated_tokens = total_chars // 4
    estimated_cost   = estimated_tokens * 0.02 / 1_000_000

    trace.update(
        output={
            "nodes_created":                 len(final_nodes),
            "documents_parsed":              len(documents),
            "estimated_embedding_tokens":    estimated_tokens,
            "estimated_embedding_cost_usd":  round(estimated_cost, 6),
        },
        metadata={
            "latency_ms":    round(timer() - t0),
            "embed_model":   "text-embedding-3-small",
            "vector_table":  "data_document_chunks",
            "bm25_cache":    NODES_CACHE,
        },
    )
    trace.end()
    lf.flush()

    print(f"Ingested {len(final_nodes)} nodes from {len(documents)} documents.")
    print(f"Estimated embedding cost: ${estimated_cost:.6f}")
    print(f"Nodes saved to {NODES_CACHE} for BM25 retrieval.")


def ingest_pdf_bytes(pdf_bytes: bytes, filename: str) -> dict:
    """
    Ingest a single PDF supplied as raw bytes (used by the /admin/ingest API endpoint).

    Steps: LlamaParse → MarkdownNodeParser → SentenceSplitter → pgvector embed+store
    → append nodes to in-memory BM25 corpus (_nodes global in rag.py).

    Returns dict with 'chunks' count and 'filename'.
    Note: nodes_cache.json is NOT written — new chunks survive only in pgvector
    (persistent on Neon) and the in-memory BM25 corpus (current server session).
    On next deploy the BM25 corpus reloads from nodes_cache.json, so new documents
    will still be found via pgvector semantic search but not via BM25 keyword search
    until nodes_cache.json is regenerated and committed.
    """
    import tempfile, pathlib

    # Write bytes to a temp file so LlamaParse / SimpleDirectoryReader can open it
    _suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ".pdf"
    with tempfile.NamedTemporaryFile(suffix=_suffix, delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = pathlib.Path(tmp.name)

    try:
        parser = LlamaParse(
            api_key=os.environ["LLAMA_CLOUD_API_KEY"],
            result_type="markdown",
        )
        reader = SimpleDirectoryReader(
            input_files=[str(tmp_path)],
            file_extractor={".pdf": parser},
        )
        # Override metadata so source shows the original filename, not the temp path
        documents = reader.load_data()
        for doc in documents:
            doc.metadata["file_name"] = filename

        markdown_parser  = MarkdownNodeParser()
        nodes            = markdown_parser.get_nodes_from_documents(documents)
        sentence_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        final_nodes: list = []
        for node in nodes:
            if len(node.text) > MAX_NODE_CHARS:
                final_nodes.extend(sentence_splitter.get_nodes_from_documents([node]))
            else:
                final_nodes.append(node)

        vector_store    = _build_vector_store()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        embed_model     = _build_embed_model()
        VectorStoreIndex(final_nodes, storage_context=storage_context, embed_model=embed_model)

        # Append to in-memory BM25 corpus for this server session
        from src.retrieval.rag import _get_nodes
        existing = _get_nodes()
        existing.extend(final_nodes)
        # Invalidate per-thread retriever so it rebuilds with new nodes
        import threading as _threading
        from src.retrieval import rag as _rag
        _rag._thread_local = _threading.local()

        return {"chunks": len(final_nodes), "filename": filename}
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    ingest_documents()
