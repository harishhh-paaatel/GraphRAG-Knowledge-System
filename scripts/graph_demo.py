r"""Build the knowledge graph from the sample documents with the REAL pipeline and show it.

Usage (from the repo root):
    backend\.venv\Scripts\python scripts\graph_demo.py

Runs in a throwaway directory using the real Alembic migrations. Everything printed is read
back from SQLite or from the NetworkX graph; nothing is hard-coded. The last section
corrupts / deletes the graph file on purpose to demonstrate recovery from SQLite.
"""

import io
import json
import mimetypes
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.core.database import make_engine, make_session_factory  # noqa: E402
from app.models import Document, Entity, EntityCandidate, EntityMention, Relationship  # noqa: E402
from app.services.extraction.factory import build_extraction_provider  # noqa: E402
from app.services.graph import queries  # noqa: E402
from app.services.graph.consistency import logical_form  # noqa: E402
from app.services.graph.service import GraphService  # noqa: E402
from app.services.graph.store import load_graph  # noqa: E402
from app.services.ingestion import service  # noqa: E402
from app.services.ingestion.pipeline import run_ingestion  # noqa: E402

SAMPLES = [
    ROOT / "scripts/samples/backend_stack.md",
    ROOT / "scripts/samples/team_notes.txt",
]


def heading(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def name(g, node_id: str) -> str:
    return g.nodes[node_id]["canonical_name"]


def node_id(g, canonical_name: str) -> str:
    return next(n for n, d in g.nodes(data=True) if d["canonical_name"] == canonical_name)


def show_edge(g, ref, label: str = "") -> None:
    u, v, k = ref
    d = g.edges[u, v, k]
    print(f"  {name(g, u)} --{k}--> {name(g, v)}  {label}".rstrip())
    for ev in d["supporting_evidence"]:
        print(
            f'      evidence: "{ev["supporting_text"]}"  (doc {ev["document_id"][:6]}, '
            f"chunk {ev['chunk_id'][:6]}, {ev['provenance']}, confidence={ev['confidence']})"
        )


def main() -> int:
    with tempfile.TemporaryDirectory(
        prefix="graphrag-graph-demo-", ignore_cleanup_errors=True
    ) as tmp:
        settings = Settings(data_dir=Path(tmp), _env_file=None)
        url = settings.resolved_database_url
        cfg = Config(str(BACKEND / "alembic.ini"))
        cfg.set_main_option("script_location", str(BACKEND / "migrations"))
        cfg.attributes["url"] = url
        command.upgrade(cfg, "head")
        engine = make_engine(url)
        factory = make_session_factory(engine)
        graph_service = GraphService(factory, settings.graph_file)
        graph_service.initialize()
        provider = build_extraction_provider(settings)

        doc_ids = {}
        with factory() as session:
            for path in SAMPLES:
                doc, job, _ = service.accept_upload(
                    session,
                    settings,
                    io.BytesIO(path.read_bytes()),
                    path.name,
                    mimetypes.guess_type(path.name)[0],
                )
                doc_ids[path.name] = doc.id
                run_ingestion(factory, settings, job.id, provider, graph_service)

        snap = graph_service.snapshot()
        g = snap.graph

        with factory() as session:
            heading("1. EXTRACTED ENTITIES (SQLite)")
            for e in session.scalars(
                select(Entity).order_by(Entity.entity_type, Entity.canonical_name)
            ):
                aliases = (e.entity_metadata or {}).get("aliases")
                print(
                    f"  {e.canonical_name:<20} {e.entity_type:<13} aliases={aliases} id={e.id[:8]}"
                )
            heading("2. EXTRACTED RELATIONSHIPS (SQLite)")
            names = {e.id: e.canonical_name for e in session.scalars(select(Entity))}
            for r in session.scalars(select(Relationship).order_by(Relationship.relationship_type)):
                print(
                    f"  {names[r.source_entity_id]} --{r.relationship_type}--> "
                    f'{names[r.target_entity_id]}   "{r.supporting_text}"'
                )
            counts = {
                m.__tablename__: len(session.scalars(select(m)).all())
                for m in (
                    Document,
                    Entity,
                    EntityMention,
                    Relationship,
                    EntityCandidate,
                )
            }
            print("  row counts:", counts)

        heading("3. GENERATED GRAPH NODES (NetworkX)")
        for n, d in sorted(
            g.nodes(data=True),
            key=lambda x: (x[1]["entity_type"], x[1]["canonical_name"]),
        ):
            print(
                f"  {d['canonical_name']:<20} {d['entity_type']:<13} in={g.in_degree(n)} out={g.out_degree(n)} "
                f"mentions={d['mention_count']} docs={len(d['source_document_ids'])} aliases={d['aliases']}"
            )
        heading("4. GENERATED GRAPH EDGES (NetworkX, with aggregated evidence)")
        print(
            f"  {g.number_of_edges()} edges, revision {snap.revision}; node attrs include entity_id, "
            "canonical_name, entity_type, description, aliases, mention_count, source_document_ids"
        )
        for ref in queries.find_edges(g):
            show_edge(
                g,
                ref,
                f"[{len(g.edges[ref]['supporting_evidence'])} evidence, "
                f"{len(g.edges[ref]['source_document_ids'])} doc(s)]",
            )

        heading("5. MULTI-HOP TRAVERSAL")
        start = node_id(g, "Sebastian Ramirez")
        t = queries.traverse(g, [start], hops=2, direction="out", max_nodes=50, max_edges=50)
        print("  from 'Sebastian Ramirez', 2 hops, direction=out (default):")
        for n, depth in sorted(t.nodes.items(), key=lambda kv: (kv[1], name(g, kv[0]))):
            print(f"    depth {depth}: {name(g, n)}")
        for ref, label in t.edges:
            print(f"    edge ({label}): {name(g, ref[0])} --{ref[2]}--> {name(g, ref[1])}")
        print(f"    truncated: {t.truncated or 'no'}")
        acme = node_id(g, "Acme Inc")
        print("\n  from 'Acme Inc', 1 hop, direction=out vs in vs both (stored direction is kept):")
        for d in ("out", "in", "both"):
            res = queries.traverse(g, [acme], hops=1, direction=d, max_nodes=50, max_edges=50)
            edges = [f"{name(g, u)} -{k}-> {name(g, v)} ({lab})" for (u, v, k), lab in res.edges]
            print(f"    {d:>4}: {edges}")

        heading("6. SHORTEST PATHS")
        for src, tgt, directed in [
            ("Sebastian Ramirez", "Pydantic", True),
            ("Alice", "Berlin", True),
            ("Berlin", "Alice", True),
            ("Alice", "Bob Smith", True),
            ("Alice", "Bob Smith", False),
            ("Alice", "FastAPI", False),
        ]:
            p = queries.shortest_path(
                g, node_id(g, src), node_id(g, tgt), directed=directed, max_hops=4
            )
            mode = "directed" if directed else "undirected"
            if p is None:
                print(f"  {src} -> {tgt} ({mode}): no path")
                continue
            print(f"  {src} -> {tgt} ({mode}): {' -> '.join(name(g, n) for n in p.nodes)}")
            for i, ref, lab in p.steps:
                print(f"      step {i} [{lab}]: {name(g, ref[0])} --{ref[2]}--> {name(g, ref[1])}")

        heading("7. SUBGRAPH (Acme Inc, 2 hops, both directions, induced edges)")
        sub = queries.traverse(
            g,
            [acme],
            hops=2,
            direction="both",
            max_nodes=50,
            max_edges=50,
            induced=True,
        )
        print("  nodes:", sorted(name(g, n) for n in sub.nodes))
        for ref, lab in sub.edges:
            print(f"  edge ({lab}): {name(g, ref[0])} --{ref[2]}--> {name(g, ref[1])}")

        heading("8. GRAPH ANALYTICS")
        print(json.dumps(graph_service.stats(), indent=2))

        heading("9. PERSISTENCE AND RECOVERY")
        path = settings.graph_file
        expected = logical_form(g)
        _, rev, meta = load_graph(path)
        print(
            f"  file: {path.name}, {path.stat().st_size} bytes, revision {rev}, "
            f"{meta['node_count']} nodes / {meta['edge_count']} edges, sha256 {meta['sha256'][:12]}..."
        )
        print(
            "  consistency (memory, file vs SQLite):",
            {
                k: v["consistent"]
                for k, v in graph_service.check_consistency().items()
                if isinstance(v, dict)
            },
        )

        def recover(label: str) -> None:
            fresh = GraphService(factory, path)
            report = fresh.initialize()
            same = logical_form(fresh.snapshot().graph) == expected
            print(
                f"  {label}: initialized from {report['source']!r} ({report['reason']}); "
                f"identical to original graph: {same}"
            )

        recover("valid file      ")
        original = path.read_bytes()
        path.write_bytes(original[: len(original) // 2])
        recover("truncated file  ")
        path.unlink()
        recover("missing file    ")
        path.write_bytes(original.replace(b"Acme Inc", b"Evil Inc", 1))
        recover("tampered file   ")
        quarantined = path.with_name(path.name + ".corrupt")
        print(
            f"  corrupt file kept for inspection: {quarantined.name} ({quarantined.stat().st_size} bytes)"
        )

        with factory() as session:
            service.delete_document(session, settings, doc_ids["backend_stack.md"])
        graph_service.sync()
        g2 = graph_service.snapshot().graph
        removed = sorted(name(g, n) for n in set(g.nodes) - set(g2.nodes))
        renamed = {
            name(g, n): name(g2, n)
            for n in set(g.nodes) & set(g2.nodes)
            if name(g, n) != name(g2, n)
        }
        print(
            f"\n  after deleting backend_stack.md: {g.number_of_nodes()} -> "
            f"{g2.number_of_nodes()} nodes, {g.number_of_edges()} -> {g2.number_of_edges()} edges"
        )
        print(f"  entities removed (no remaining evidence): {removed}")
        print(
            f"  entities kept but recomputed from remaining evidence (same ID, new name): {renamed}"
        )
        shared = next(n for n in g2.nodes if g2.nodes[n]["entity_type"] == "ORGANIZATION")
        print(
            f"  shared entity {name(g2, shared)!r} now supported by {len(g2.nodes[shared]['source_document_ids'])} document "
            f"(was {len(g.nodes[shared]['source_document_ids'])}); its aliases: {g2.nodes[shared]['aliases']}"
        )
        engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
