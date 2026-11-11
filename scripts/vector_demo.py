r"""Semantic vector retrieval demo using the REAL ingestion pipeline and (by default) the real model.

Usage (from the repo root):
    backend\.venv\Scripts\python scripts\vector_demo.py
    backend\.venv\Scripts\python scripts\vector_demo.py --provider hashing    # no model needed

Runs in a throwaway directory with the real Alembic migrations; the downloaded model is cached
in data/models (the project drive). Everything printed is measured from the run: nothing is
hard-coded. Steps: load stored chunks -> generate embeddings -> FAISS index -> semantic queries
with similarity scores -> delete a document -> rebuild / recover.
"""

import argparse
import io
import mimetypes
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.core.database import make_engine, make_session_factory  # noqa: E402
from app.models import ChunkEmbedding, Document, DocumentChunk  # noqa: E402
from app.services.embeddings.embedder import SentenceTransformerEmbedder  # noqa: E402
from app.services.embeddings.factory import build_embedder  # noqa: E402
from app.services.embeddings.service import VectorService  # noqa: E402
from app.services.embeddings.vector_store import load_index  # noqa: E402
from app.services.extraction.factory import build_extraction_provider  # noqa: E402
from app.services.graph.service import GraphService  # noqa: E402
from app.services.ingestion import service  # noqa: E402
from app.services.ingestion.pipeline import run_ingestion  # noqa: E402
from app.services.retrieval.vector_search import search_vectors  # noqa: E402

SAMPLES = [
    "backend_stack.md",
    "team_notes.txt",
    "geography.txt",
    "vector_search_notes.md",
    "kitchen_notes.txt",
]
QUERIES = [
    "Which technologies are used to build Python APIs?",
    "What is the capital of France?",
    "How do I make bread?",
    "How does cosine similarity compare two embeddings?",
    "Who works in Berlin?",
]


class TimedEmbedder:
    """Transparent proxy that measures real embedding time (model load is timed separately)."""

    def __init__(self, inner):
        self._inner = inner
        self.seconds = 0.0
        self.texts = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def embed(self, texts):
        t = time.perf_counter()
        out = self._inner.embed(texts)
        self.seconds += time.perf_counter() - t
        self.texts += len(texts)
        return out


def heading(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def preview(text: str, n: int = 74) -> str:
    one = " ".join(text.split())
    return one if len(one) <= n else one[: n - 1] + "…"


def show(outcome, label: str = "") -> None:
    print(f'\n  Q: "{outcome.query}" {label}')
    if not outcome.results:
        print("     (no results)")
    for r in outcome.results:
        where = (
            f"{r.filename}"
            + (f" §{r.section}" if r.section else "")
            + (f" p.{r.page_number}" if r.page_number else "")
        )
        print(f"     {r.rank}. similarity={r.similarity:.4f}  [{where}]  {preview(r.content)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--provider",
        choices=["sentence_transformers", "hashing"],
        default="sentence_transformers",
    )
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to a legacy codepage

    with tempfile.TemporaryDirectory(
        prefix="graphrag-vec-demo-", ignore_cleanup_errors=True
    ) as tmp:
        settings = Settings(
            data_dir=Path(tmp),
            embedding_provider=args.provider,
            embedding_cache_dir=ROOT / "data" / "models",
            _env_file=None,
        )
        url = settings.resolved_database_url
        cfg = Config(str(BACKEND / "alembic.ini"))
        cfg.set_main_option("script_location", str(BACKEND / "migrations"))
        cfg.attributes["url"] = url
        command.upgrade(cfg, "head")
        engine = make_engine(url)
        factory = make_session_factory(engine)
        graph = GraphService(factory, settings.graph_file)
        graph.initialize()
        embedder = TimedEmbedder(build_embedder(settings))
        t_load = time.perf_counter()
        embedder.identity()  # load the model now so the embedding time below excludes loading
        model_load_s = time.perf_counter() - t_load
        vectors = VectorService(
            factory, settings.vector_index_file, embedder, sync_on_startup=False
        )
        vectors.initialize()
        provider = build_extraction_provider(settings)

        heading("1. INGEST SAMPLE DOCUMENTS (parse -> chunk -> entities -> graph -> embeddings)")
        doc_ids = {}
        t0 = time.perf_counter()
        with factory() as session:
            for name in SAMPLES:
                path = ROOT / "scripts" / "samples" / name
                doc, job, _ = service.accept_upload(
                    session,
                    settings,
                    io.BytesIO(path.read_bytes()),
                    name,
                    mimetypes.guess_type(name)[0],
                )
                doc_ids[name] = doc.id
                run_ingestion(factory, settings, job.id, provider, graph, vectors)
        ingest_s = time.perf_counter() - t0
        with factory() as session:
            docs = {d.id: d for d in session.scalars(select(Document))}
            for d in sorted(docs.values(), key=lambda d: d.filename):
                vi = d.doc_metadata.get("vector_index", {})
                print(
                    f"  {d.filename:<24} status={d.processing_status:<10} chunks={vi.get('chunks_total')} "
                    f"vectors indexed={vi.get('chunks_indexed')} (embedded now: {vi.get('embedded_now')})"
                )
        print(f"  total pipeline time for {len(SAMPLES)} documents: {ingest_s:.1f}s")

        heading("2. STORED CHUNKS (SQLite is the source of truth)")
        with factory() as session:
            for c in session.scalars(
                select(DocumentChunk).order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
            ):
                print(
                    f"  {docs[c.document_id].filename:<24} #{c.chunk_index} {('§' + c.section) if c.section else '':<14} {preview(c.content, 60)}"
                )
            total_chunks = session.query(DocumentChunk).count()

        heading("3. EMBEDDINGS AND FAISS INDEX")
        ident = embedder.identity()
        meta = vectors.snapshot_meta
        stats = vectors.stats()
        print(f"  model:           {ident.model_id}")
        print(f"  model revision:  {ident.revision}")
        print(
            f"  dimension:       {ident.dimension}   (normalized: inner product == cosine similarity)"
        )
        print(f"  config fingerprint: {meta.config_fingerprint[:16]}…")
        print(
            f"  vectors indexed: {vectors.index_count} of {total_chunks} chunks   status counts: {stats['embedding_status_counts']}"
        )
        print(f"  model load:      {model_load_s:.1f} s (one-off)")
        print(
            f"  embedding time:  {embedder.texts} texts in {embedder.seconds * 1000:.0f} ms "
            f"= {embedder.texts / embedder.seconds:.1f} texts/s "
            "(tiny batches of 1-3 chunks as documents arrive; see the benchmark for batched throughput)"
        )
        index_file = settings.vector_index_file
        _, fmeta = load_index(index_file)
        print(
            f"  index file:      {index_file.name}, {index_file.stat().st_size} bytes, revision {fmeta.revision}, "
            f"in sync with SQLite: {stats['file_in_sync']}"
        )
        with factory() as session:
            row = session.scalars(select(ChunkEmbedding)).first()
            print(
                f"  bookkeeping row: chunk={row.chunk_id[:10]}… model={row.model_id} dim={row.dimension} "
                f"status={row.status} hash={row.content_hash[:12]}…"
            )

        heading(
            "4. SEMANTIC QUERIES (top 3, cosine similarity = retrieval similarity, NOT answer confidence)"
        )
        for q in QUERIES:
            show(search_vectors(vectors, factory, q, top_k=3))
        show(
            search_vectors(
                vectors,
                factory,
                "How do I make bread?",
                top_k=3,
                similarity_threshold=0.5,
            ),
            "(threshold 0.5)",
        )
        show(
            search_vectors(
                vectors,
                factory,
                "What is the capital of France?",
                top_k=3,
                document_ids=[doc_ids["team_notes.txt"]],
            ),
            "(restricted to team_notes.txt)",
        )
        out = search_vectors(vectors, factory, "What is the capital of France?", top_k=1)
        print(f"\n  timings for one query (ms): {out.timings_ms}")

        heading("5. DELETE A DOCUMENT -> ITS VECTORS ARE REMOVED")
        before = vectors.index_count
        bread_before = search_vectors(vectors, factory, "How do I make bread?", top_k=1).results[0]
        with factory() as session:
            service.delete_document(session, settings, doc_ids["kitchen_notes.txt"])
        sync_report = vectors.sync()
        print(
            f"  vectors: {before} -> {vectors.index_count}   (removed {sync_report.removed}, embedded {sync_report.embedded}: "
            "deletion needs no model)"
        )
        print(
            f"  top hit for 'How do I make bread?' was {bread_before.filename} ({bread_before.similarity:.4f}); now:"
        )
        show(search_vectors(vectors, factory, "How do I make bread?", top_k=1))
        print(
            f"  index file now holds {load_index(index_file)[0].count} vectors; "
            f"consistency: {vectors.check_consistency()['memory']['consistent']}"
        )

        heading("6. REBUILD FROM SQLITE-BACKED CHUNKS")
        ref = search_vectors(
            vectors,
            factory,
            "Which technologies are used to build Python APIs?",
            top_k=5,
        )
        report = vectors.rebuild()
        again = search_vectors(
            vectors, factory, "Which technologies are used to build Python APIs?", top_k=5
        )
        same_order = [r.chunk_id for r in ref.results] == [r.chunk_id for r in again.results]
        max_diff = max(
            abs(x.similarity - y.similarity)
            for x, y in zip(ref.results, again.results, strict=True)
        )
        print(f"  rebuild: embedded {report.embedded} chunks in {report.embed_ms:.0f} ms")
        print(
            f"  same ranking as before the rebuild: {same_order}; "
            f"max |similarity difference| = {max_diff:.2e}"
        )
        print(
            "  (float32 vectors computed with a different batch composition differ by ~1 ULP: the "
            "rebuilt index is logically equivalent, not necessarily bit-identical)"
        )

        heading("7. RECOVERY AND MODEL-CHANGE DETECTION")
        expected = sorted((e.chunk_id, e.content_hash) for e in vectors.indexed_entries())

        def fresh(emb=None, **kw):
            return VectorService(factory, index_file, emb or build_embedder(settings), **kw)

        raw = index_file.read_bytes()
        for label, damage in [
            ("truncated file", raw[: len(raw) // 2]),
            ("bit-flipped payload", raw[:-9] + bytes([raw[-9] ^ 0xFF]) + raw[-8:]),
        ]:
            index_file.write_bytes(damage)
            svc = fresh()
            rep = svc.initialize()
            ok = sorted((e.chunk_id, e.content_hash) for e in svc.indexed_entries()) == expected
            print(
                f"  {label:<20}: source={rep['source']!r} ({rep['reason']}); re-embedded {rep['sync']['embedded']}; "
                f"same chunks indexed: {ok}"
            )
        index_file.unlink()
        svc = fresh()
        rep = svc.initialize()
        print(
            f"  missing file        : source={rep['source']!r} ({rep['reason']}); re-embedded {rep['sync']['embedded']}"
        )
        svc = fresh()
        rep = svc.initialize()
        print(
            f"  valid current file  : source={rep['source']!r}; re-embedded {rep['sync']['embedded']} (nothing recomputed)"
        )

        if args.provider == "sentence_transformers":
            other = SentenceTransformerEmbedder(
                settings.embedding_model,
                revision=settings.embedding_model_revision,
                cache_dir=settings.model_cache_dir,
                max_seq_length=32,
            )
            svc = fresh(other)
            rep = svc.initialize()
            print(
                f"  changed embedding config (max_seq_length=32): source={rep['source']!r} "
                f"({rep['reason']}); rebuilt={rep['sync']['rebuilt']}, re-embedded {rep['sync']['embedded']} "
                "-> old vectors discarded, never mixed"
            )
        engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
