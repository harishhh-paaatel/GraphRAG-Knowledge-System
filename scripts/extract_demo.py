r"""Run the real ingestion pipeline (including knowledge extraction) on files and print the result.

Usage (from the repo root):
    backend\.venv\Scripts\python scripts\extract_demo.py scripts\samples\backend_stack.md scripts\samples\team_notes.txt

Uses a throwaway SQLite database created with the real Alembic migrations, so nothing in
data/ is touched. Pass --provider mock to see the empty-extraction behaviour.
"""

import argparse
import io
import mimetypes
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.core.database import make_engine, make_session_factory  # noqa: E402
from app.models import (  # noqa: E402
    Document,
    Entity,
    EntityCandidate,
    EntityMention,
    ProcessingJob,
    Relationship,
)
from app.services.extraction.factory import build_extraction_provider  # noqa: E402
from app.services.ingestion import service  # noqa: E402
from app.services.ingestion.pipeline import run_ingestion  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--provider", choices=["rule_based", "mock"], default="rule_based")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="graphrag-demo-", ignore_cleanup_errors=True) as tmp:
        settings = Settings(data_dir=Path(tmp), extraction_provider=args.provider, _env_file=None)
        url = settings.resolved_database_url
        cfg = Config(str(BACKEND / "alembic.ini"))
        cfg.set_main_option("script_location", str(BACKEND / "migrations"))
        cfg.attributes["url"] = url
        command.upgrade(cfg, "head")
        engine = make_engine(url)
        factory = make_session_factory(engine)
        provider = build_extraction_provider(settings)

        with factory() as session:
            for path in args.files:
                mime = mimetypes.guess_type(path.name)[0]
                _, job, _ = service.accept_upload(
                    session, settings, io.BytesIO(path.read_bytes()), path.name, mime
                )
                run_ingestion(factory, settings, job.id, provider)

            session.expire_all()
            names = {e.id: e.canonical_name for e in session.scalars(select(Entity))}
            print(f"\n=== provider: {provider.name} ===")
            for doc in session.scalars(select(Document).order_by(Document.filename)):
                job = session.scalars(
                    select(ProcessingJob).where(ProcessingJob.document_id == doc.id)
                ).one()
                k = doc.doc_metadata.get("knowledge", {})
                print(
                    f"\n[{doc.filename}] status={doc.processing_status} job={job.status}/{job.stage}"
                    f" chunks={doc.doc_metadata.get('chunk_count')} knowledge={ {x: k.get(x) for x in ('mentions', 'relationships', 'rejected_relationships', 'failed_chunks')} }"
                )
            print("\n--- entities (canonical | type | aliases | #mentions | docs) ---")
            for e in session.scalars(
                select(Entity).order_by(Entity.entity_type, Entity.canonical_name)
            ):
                ms = e.mentions
                print(
                    f"{e.canonical_name:<20} {e.entity_type:<13} aliases={e.entity_metadata.get('aliases')}"
                    f" mentions={len(ms)} docs={len({m.document_id for m in ms})} id={e.id[:8]}"
                )
            print("\n--- relationships (source -TYPE-> target | supporting text | chunk) ---")
            for r in session.scalars(select(Relationship).order_by(Relationship.relationship_type)):
                print(
                    f"{names[r.source_entity_id]} -{r.relationship_type}-> {names[r.target_entity_id]}"
                    f' | "{r.supporting_text}" | chunk={r.source_chunk_id[:8]} prov={r.provenance}'
                    f" conf={r.confidence}"
                )
            print("\n--- merge candidates awaiting review ---")
            cands = session.scalars(select(EntityCandidate)).all()
            for c in cands:
                print(
                    f"{names[c.entity_a_id]} ? {names[c.entity_b_id]} reason={c.reason} score={c.score}"
                )
            if not cands:
                print("(none)")
            print(
                "\nrow counts:",
                {
                    m.__tablename__: len(session.scalars(select(m)).all())
                    for m in (
                        Document,
                        Entity,
                        EntityMention,
                        Relationship,
                        EntityCandidate,
                    )
                },
            )
        engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
