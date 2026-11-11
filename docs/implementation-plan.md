# Implementation Plan and Status

## Proposed directory structure
Follows the suggested layout, with these changes:
- Repo root is the existing workspace folder (not nested `graphrag-knowledge-service/`).
- Added `backend/app/models` (ORM) vs `schemas` (Pydantic) as suggested; added `backend/app/core/` for config/logging/security, `backend/pyproject.toml` (ruff/pytest config) alongside `requirements.txt`, and `docs/status.md` (this file).
- `data/` is git-ignored except `.gitkeep`; `evaluation/reports/` is committed only when produced by real runs.

```
E:\GraphRAG Knowledge System\
├── backend\ app\{api,core,models,schemas,repositories,services\{ingestion,extraction,graph,embeddings,retrieval,generation,evaluation\},providers,workers}\main.py
│            tests\{unit,integration,e2e,fixtures}  migrations\  requirements.txt  pyproject.toml  Dockerfile
├── frontend\ src\{components,pages,hooks,services,types,utils}  package.json  Dockerfile
├── data\{documents,graphs,embeddings,evaluations}
├── evaluation\{datasets,scripts,reports}
├── docs\{architecture.md,api.md,evaluation.md,design-decisions.md,implementation-plan.md}
├── scripts\  .github\workflows\ci.yml  .env.example  .gitignore  docker-compose.yml  README.md  LICENSE
```

## Phases with acceptance criteria

| Phase | Deliverables | Measurable acceptance criteria |
|---|---|---|
| 1 Backend foundation | venv on E:, config (pydantic-settings), SQLAlchemy models + Alembic baseline, structured logging w/ secret redaction, `GET /health`, error envelope, pytest + ruff setup, local `git init` | `pytest` green; `alembic upgrade head` on empty DB succeeds; `/health` 200; ruff clean |
| 2 Ingestion | upload endpoint, validation, checksum dedupe, parsers (PDF/DOCX/TXT/MD), normalizer, chunker, job table + worker | duplicate upload → 409; bad type → 415; oversize → 413; chunking deterministic across runs; sample PDF/DOCX/TXT/MD each parse in tests; upload returns 202 without blocking |
| 3 Extraction | provider-agnostic extractor, rule-based extractor, Pydantic validators, normalization/dedupe | relationship with unsupported text is rejected (test); "OpenAI"/"Open AI" merge, same-name different-type do not; ambiguity preserved |
| 4 Knowledge graph | NetworkX engine, GraphML atomic persistence, traversal, centrality, components, shortest path, graph APIs, rebuild script | graph reload equals pre-save graph; delete removes orphan nodes; hop-limited traversal tests; rebuild from SQL reproduces graph |
| 5 Vector retrieval | embedding service (configurable model), FAISS index + id map, doc filter, persistence | top-K returns known nearest in fixtures; filter by document works; index survives restart; delete removes vectors |
| 6 Retrieval engine | vector / graph / hybrid strategies, RRF, dedupe, token budget, search endpoints | strategies selectable & unit-tested; hybrid has no duplicate chunks; empty-graph fallback flagged in metadata |
| 7 Generation | provider interface (Ollama, OpenAI-compat, Mock), retry/timeouts, grounded prompt, citation builder, insufficient-evidence handling, `/query`, history | every citation maps to a retrieved chunk (test); insufficient-evidence path tested; `generation_mode` always present; providers tested via `httpx.MockTransport` |
| 8 Evaluation | dataset, P@K/R@K/MRR/Hit@K, faithfulness/relevance/citation metrics (clearly separated), latency timing, report generator, API | metric functions match hand-computed values; run records dataset size + model/config; report generated only from stored runs |
| 9 Frontend | 6 pages, dark/light, Cytoscape explorer, TanStack Query, no secrets in UI | `npm run build` + `tsc` clean; each page works against the live backend (manually verified in browser); responsive at mobile width |
| 10 DevOps | Dockerfiles, compose with named volumes + healthchecks, GitHub Actions (lint, test, health check) | `docker compose up` → healthy; CI workflow has no secrets and passes locally-equivalent commands |
| 11 Final QA | full test run, bug fixes, README/docs, screenshots from the real app, GitHub readiness review | all tests pass (reported as measured); screenshots captured from running app; **no push without your explicit approval** |

## Working rules for every phase
Inspect before editing; tests written with code; app runnable after each phase; this file's status table updated; no dependency added without a stated reason; stop and report at the end of each phase for approval.

## Status

| Phase | State |
|---|---|
| 0 Inspection & architecture | Complete, approved |
| 1 Backend foundation | Complete, approved. |
| 2 Document ingestion | Complete, approved. |
| 3 Knowledge extraction | Complete, approved. 152 tests pass (71 from Phases 1–2 + 81 new), 97% line coverage on `app/`, Ruff clean, suite run 3× with identical results. Migration `0002` verified (no drift vs ORM, downgrade/upgrade round-trip). |
| 3.1 Evidence-derived entity attributes | Complete, approved. 172 tests pass (152 + 20 new), 97% line coverage, Ruff clean, suite run 3× identically; a mutation check (reverting to first-seen values) makes 9 of the new tests fail. Migration `0003` verified. |
| 4 Knowledge graph (NetworkX) | Complete, approved. 289 tests pass (172 from Phases 1-3.1 + 117 new), 97% line coverage on `app/` (graph modules 93-100%), Ruff clean, suite run 3x identically. Migration `0004` verified (no drift, full 0004->0001 round trip). Mutation checks: 46 tests fail if the revision is never bumped, 18 fail if edge direction is reversed. Verified against a live uvicorn server. |
| 5 Vector retrieval (FAISS) | **Complete — awaiting your review and approval** (reliability fixes 5A, docs 5B and final verification 5C done). Default suite 538 passed, 17 skipped (the skipped ones are the opt-in real-model tests); 17 real-model tests pass when enabled; 96% line coverage on `app/`; Ruff check and format clean; migration `0006` verified (single head, no drift, 0006→0001→0006 round trip); live uvicorn smoke test against the real model passed. Full results: `docs/phase-5-final-report.md`. |
| 6 Hybrid retrieval | **Complete - awaiting your final approval.** M0-M4 approved; M5 (close-out: `scripts/hybrid_demo.py`, rewritten retrieval section of `docs/architecture.md`, final verification) done. Final verification: Ruff clean (145 files); 1111 tests passed / 17 skipped; 97% line coverage; migration head `0006`, no drift; 17 real-model tests passed; live concurrent API test on the real model passed (74 requests, 0 errors, 0 evidence mismatches); 65/65 mutation checks detected across M1-M4; Git audit clean. **Retrieval-quality improvement over vector-only retrieval has NOT been established** (needs the Phase 8 labelled evaluation). Report: `docs/phase-6-final-report.md`. Phase 7 not started. |
| 7 Grounded answer generation | **In progress, one milestone at a time with your review after each.** Proposal `docs/phase-7-proposal.md` approved. **M0 (foundations) implemented and awaiting approval:** settings, provider interface and dependency injection, `/query` schemas, query-history store on the existing table, privacy rules and error taxonomy. Ruff clean; 1361 tests passed / 17 skipped; 97% coverage; migration head still `0006`; 32/32 mutation checks detected. No provider is registered yet (M1), no endpoint is exposed (M4), no answer is generated. |
| 8-11 | Not started |

### Phase 1 notes
- Run from `backend/`: `.venv\Scripts\python -m pytest`, `-m ruff check .`, `-m alembic upgrade head`, `-m uvicorn app.main:app`.
- venv and pip cache live on E: (`backend/.venv`, `E:\pip-cache`). Installed: fastapi 0.162, uvicorn 0.54, pydantic 2.14, SQLAlchemy 2.0.57, alembic 1.20, pytest 8.4, httpx 0.28, ruff 0.17.
- Known: Starlette emits a deprecation warning that `TestClient` prefers `httpx2`; harmless for now, revisit when pinning versions in Phase 10.
- `/health` originally reported `graph_index` and `vector_index` as `not_configured` (deliberately not faked as "ok"); since Phases 4 and 5 both report their real state.
- Local git repo initialized; nothing committed, no remote.

## Phase 2 notes (document ingestion)

**Endpoints:** `POST /documents/upload` (202), `GET /documents` (pagination `limit/offset`, `status` filter), `GET /documents/{id}`, `GET /documents/{id}/status`, `DELETE /documents/{id}` (204).

**Flow:** stream upload to `data/tmp` while hashing (SHA-256) and enforcing the size limit → extension allow-list (415) → declared MIME check (415) → content sniffing (400: PDF header, DOCX zip structure + zip-bomb guard, NUL bytes in text, empty file) → duplicate check (409 with the existing `document_id`) → move to `data/documents/<id>/source.<ext>` → insert Document + pending ProcessingJob in one transaction → queue on the thread pool → return 202.
Worker stages: `extracting → chunking → persisting → done`. Persisting deletes the document's old chunks and inserts the new ones, and marks the job and document completed, in **one transaction**, which is what makes retries and recovery idempotent.

**Design decisions made in this phase**
- Corrupted/blank files are accepted (202) and fail *asynchronously* with a visible `failed` status and a safe `error_message`; parsing at request time would block the API. Only cheap structural checks run synchronously.
- Re-uploading a file whose previous attempt **failed** re-queues it on the same document (`retry: true`, job `attempt` incremented) instead of returning 409. This is the retry path; no extra endpoint was added.
- Startup recovery re-queues `pending`/`running` jobs (running ones get `attempt + 1`); jobs whose file vanished are failed. A database without tables is tolerated (logged, not fatal).
- Chunks are built per segment (PDF page, DOCX/Markdown section), so a chunk never spans pages or sections. Chunk ID = `sha1(checksum:index)`. Exact duplicate chunks within a document (whitespace/case-insensitive) are dropped, and indexes are kept contiguous.
- Defaults: chunk 800 chars, overlap 100, 2 worker threads, 25 MB upload (all `GRAPHRAG_*` env vars).
- Deleting a document returns 409 while `pending`/`processing`. DB rows are removed first (source of truth); a file-removal failure is logged, not fatal.
- Bugs found and fixed during verification: `alembic upgrade head` failed on a fresh checkout when `data/` did not exist (now created in `migrations/env.py`); switched `import fitz` to `import pymupdf` after PyMuPDF's deprecation warning.

**Known limitations**
- Scanned/image-only PDFs are rejected (no OCR). Password-protected PDFs are rejected.
- Text/Markdown must be UTF-8 (BOM tolerated); other encodings fail rather than risk silent mojibake.
- Page numbers exist for PDFs only; DOCX has no reliable page concept without rendering, so DOCX/MD/TXT chunks carry `section` instead (and null page).
- PDF heading detection is not implemented (PDF chunks have no `section`). PDF text order for multi-column or table-heavy layouts is whatever PyMuPDF returns.
- Chunk size is in characters, not tokens; DOCX headers/footers, footnotes and text boxes are not extracted; Markdown setext (`===`) headings are not recognised.
- Duplicate detection is by exact bytes: the same text saved as a different file format, or re-saved, is a different document.
- The thread pool is in-process: a crash mid-job is recovered on next startup, but there is no cross-process queue. Jobs run in submission order per worker with no priority or cancellation.
- Exact-duplicate chunk removal can drop legitimately repeated text (e.g. a repeated sentence on two pages); only the first occurrence is kept.
- The upload size is enforced while streaming, but Starlette still spools the multipart body to a temp file first, so a very large rejected upload briefly uses disk/time. No rate limiting yet.
- No authentication or per-user isolation (single-user design, see architecture.md).
- `embedding`, entity and graph stages are intentionally absent: documents end at `completed` meaning "ingested and chunked" only. Phases 3–5 will extend the stage list.
- Verification used a throwaway data dir `E:\graphrag-verify` (outside the repo); it was not deleted because the delete was blocked by a safety check. You can remove it.

## Phase 3 notes (knowledge extraction)

**Pipeline change:** stages are now `extracting → chunking → persisting → extracting_knowledge → done`. A document becomes `completed` only after its entities, mentions and relationships are stored. Provider calls run outside DB transactions; the write transaction holds an in-process `WRITE_LOCK` because entity resolution reads the whole entity table.

**Modules** (`backend/app/services/extraction/`): `types` (data classes, relationship vocabulary and type rules), `normalization`, `validation` (evidence checks), `resolver`, `persistence`, `rule_based`, `mock`, `llm`, `factory`.

**Schema (migration 0002):** `entity_mentions.provenance`, `relationships.provenance`, unique `(entity_id, chunk_id, surface_form)` on mentions, new table `entity_candidates` (pair, reason, score, status). Deviation from architecture.md: merge candidates are a table, not JSON metadata, so they are queryable for review.

**Validation rules** (everything is checked against the chunk the provider saw): entity type must be enabled; the surface form must occur in the chunk; a relationship needs a known type, a supporting text that occurs in the chunk, both endpoints extracted from that chunk and named in the supporting text, no self-loops, and a valid source/target type pair for its relationship type. Anything else is rejected, counted, and sampled into `documents.metadata.knowledge.sample_rejections`.

**Resolution policy:** exact normalized key + same type merges; compact key (spaces/hyphens/apostrophes removed, ≥5 chars, same type, exactly one hit) merges ("Open AI" ~ "OpenAI"); organizations ignore trailing Inc/Corp/Ltd/etc. Everything else stays separate and is flagged in `entity_candidates` as `same_name_different_type`, `fuzzy_name` (similarity ≥ 0.88, same type, same digits), `token_subset` (person/org names) or `compact_ambiguous`. Nothing flagged is ever merged automatically. Entity IDs are deterministic (`sha1(type:key)`), so reprocessing keeps entity identity.

**Decisions**
- Model-reported LLM confidence is ignored (uncalibrated); rule-based reports none. `confidence` is therefore NULL everywhere today.
- A chunk that states one fact twice keeps the first evidence (table allows one row per source/target/type/chunk). A sentence repeated by chunk overlap is stored once per document, but each chunk still gets its own mention rows (needed later for retrieval).
- If the provider fails on some chunks the document still completes and `failed_chunks` is recorded; if it fails on *all* chunks the job fails.
- `extraction_provider=llm` fails fast at startup until Phase 7 supplies a client.

**Bugs found by the new tests and fixed:** (1) a chunk stating the same fact twice violated the relationship unique key and failed the whole job; (2) re-running a document deleted and recreated its exclusive entities with new random IDs (now deterministic).

**Known limitations**
- The rule-based extractor is a test/demo tool, not a quality baseline: a ~80-term gazetteer, a ~65-name first-name list, English only, pattern-matched verbs that must sit *directly* between two entities (no coordination: "A uses B and C" yields only A-B), lowercase names are not recognised, single-token people are not detected, no pronoun/coreference handling, no dates.
- No relationship direction/negation reasoning beyond the passive-voice patterns ("was created by"); "does not use" is not detected as a negation but also does not match a pattern, so it is skipped rather than inverted.
- Fuzzy matching scans all entities of the type per new entity (O(n)); fine for thousands, not for millions. Entity resolution holds a single process-wide write lock, so extraction persistence is serialized (provider calls are not).
- Merge candidates are recorded but there is no review API/UI yet; `status` stays `pending`.
- Entity/relationship counts live in `documents.metadata.knowledge`; there is no endpoint exposing the extracted records yet (graph APIs arrive in Phase 4).

**Reproduce locally** (from the repo root; uses a throwaway DB with the real migrations):
```
backend\.venv\Scripts\python scripts\extract_demo.py scripts\samples\backend_stack.md scripts\samples\team_notes.txt
backend\.venv\Scripts\python scripts\extract_demo.py --provider mock scripts\samples\team_notes.txt
cd backend
.venv\Scripts\python -m pytest --cov=app
.venv\Scripts\python -m ruff check . ../scripts
.venv\Scripts\python -m alembic upgrade head
```

## Phase 3.1 notes (evidence-derived entity attributes)

**Problem:** an entity's canonical name and description kept their first-seen values after the document that introduced them was deleted, and aliases carried no provenance.

**Change:** `entity_mentions` now stores what the extractor reported for that evidence (`extracted_name`, `description`). `refresh_entities` recomputes everything derived from the mentions that remain, in the same transaction as any ingest, re-run or delete:

| Attribute | Rule | Provenance (in `entities.metadata`) |
|---|---|---|
| canonical name | `extracted_name` of the *earliest remaining evidence* | `canonical_name_source: {document_id, chunk_id}` |
| description | first non-empty description among remaining evidence, else `NULL` | `description_source: {document_id, chunk_id}` or `null` |
| aliases | surface forms (other than the canonical name) still present | `aliases` plus `alias_sources: {alias: [document_ids]}` |

"Earliest" = document `upload_timestamp`, then document ID, chunk index, name, surface form: deterministic, independent of processing order, and unchanged when a document is reprocessed. Deleting a *later* document therefore never changes established values.

**Identity is never recomputed:** `id`, `normalized_key` and `entity_type` are not touched, so recomputation cannot merge or split entities and IDs stay deterministic (`sha1(type:key)`). An entity with no remaining mentions (and no relationships) is deleted; its relationships, mentions and merge candidates go with it via FK cascade. An entity supported by another document is kept for as long as one valid supporting document remains.

**Migration 0003** adds the two columns and backfills legacy rows: `extracted_name` from the entity's current canonical name, and the entity's current description onto its earliest mention (so legacy descriptions become removable with their evidence). Covered by a test that builds a pre-0003 database.

**Tests added (20):** `tests/integration/test_evidence_recompute.py` (two documents supporting one entity; delete the first; delete a later one; delete the last supporting document; canonical-name and description recomputation including processing-order independence and reprocessing; alias cleanup per document; relationship evidence cleanup and no dangling references; distinct/look-alike entities stay distinct with stable IDs; deterministic IDs across delete + re-ingest; end-to-end through the API) and one migration-backfill test in `test_database.py`.

**Limitations of this change**
- Ties on identical `upload_timestamp` fall back to document ID order, which is arbitrary but stable.
- The canonical name can change (to another spelling the remaining evidence used) when the earliest document is deleted; consumers should key on entity `id`, never on the name.
- Backfilled legacy mentions all carry the entity's former canonical name as `extracted_name`, so alias-spelling provenance for pre-0003 data is approximate (surface forms are still exact).
- Recomputation is O(mentions of the touched entities) per ingest/delete, run inside the write lock.

## Phase 4 notes (NetworkX knowledge graph)

Design, file format, consistency protocol and analytics conventions are in `docs/architecture.md` ("Knowledge graph (Phase 4)"). Summary of decisions:

- `MultiDiGraph`; one edge per (source, type, target) aggregating all evidence rows; direction never reversed; JSON node-link file with SHA-256 + counts, written atomically; SQLite `graph_state.revision` bumped in the same transaction as any evidence change.
- Reads self-heal (rebuild when SQLite is ahead); startup adopts the file only if it validates and is current; corrupt files are quarantined to `knowledge_graph.json.corrupt`.
- Relationship row IDs became deterministic (found by the reprocessing test: random IDs made evidence identity change on every re-run).
- Mentions are stored once per (entity, chunk, surface form), so `mention_count` counts distinct chunk/surface pairs, not textual occurrences.

**Files created:** `app/services/graph/{revision,builder,store,queries,analytics,consistency,service}.py`, `app/api/graph.py`, `app/schemas/graph.py`, `migrations/versions/0004_graph_state.py`, `scripts/graph_demo.py`, and tests `unit/test_graph_queries.py`, `unit/test_graph_store.py`, `integration/test_graph_builder.py`, `integration/test_graph_sync.py`, `integration/test_graph_api.py`.
**Files modified:** `models/operations.py` + `models/__init__.py` (GraphState), `core/config.py` (graph limits, `graph_file`), `services/extraction/persistence.py` (revision bump, deterministic relationship IDs), `services/ingestion/pipeline.py` + `workers/ingestion_worker.py` (eager graph sync, never fails a job), `api/documents.py` (sync after delete), `api/deps.py`, `api/health.py` (real `graph_index` status), `main.py` (startup initialization), `tests/integration/test_database.py` (0004 test), `requirements.txt` (networkx), `pyproject.toml`.

### Endpoints (all `GET /api/v1/graph/...`)
| Path | Parameters | Notes |
|---|---|---|
| `/overview` | none | counts, type distributions, density, weak + strong components, top entities by degree centrality, graph state; cached per revision |
| `/nodes` | `entity_type`, `search` (names + aliases), `document_id`, `sort=name\|degree`, `limit` 1-200, `offset` | |
| `/edges` | `relationship_type`, `source_id`, `target_id`, `document_id`, `limit`, `offset` | every edge includes `supporting_evidence`, `source_document_ids`, `source_chunk_ids`, `provenance`; response has `documents` {id: filename} |
| `/entities/{id}` | `mention_limit` 0-500 | node + incoming/outgoing edges with evidence + mentions |
| `/entities/{id}/neighbors` | `direction=out\|in\|both` (default out), `hops` (<= cap), repeatable `relationship_type`, `max_nodes`, `max_edges`, `induced` | returns nodes with depth, edges with `traversal` label, `truncated` list, applied limits |
| `/path` | `source`, `target`, `directed` (default true), `max_hops` | 404 `no_path` / `entity_not_found`; undirected steps are labelled `reverse` and keep their stored direction |

Errors: 404 `entity_not_found`/`no_path`, 422 `validation_error`/`limit_exceeded`, 503 `graph_unavailable`.

### Real output (sample documents, rule-based extraction; produced by `scripts/graph_demo.py`)
SQLite after ingesting `backend_stack.md` + `team_notes.txt`: 12 entities, 15 mentions, 9 relationship rows. Graph: **12 nodes, 8 edges** (9 rows -> 8 edges because `Acme Inc --LOCATED_IN--> Berlin` aggregates two evidence rows from two documents).

```
Sebastian Ramirez --DEVELOPED--> FastAPI --USES--> Pydantic        (shortest path, directed, 2 hops)
Alice --WORKS_AT--> Acme Inc --LOCATED_IN--> Berlin                 (shortest path, directed, 2 hops)
Berlin -> Alice (directed): no path          Alice -> Bob Smith (directed): no path
Alice -> Bob Smith (undirected): Alice -> Acme Inc -> Bob Smith   steps: [forward] Alice-WORKS_AT->Acme Inc, [reverse] Bob Smith-WORKS_AT->Acme Inc
Alice -> FastAPI (undirected): no path (different components)
traverse from Sebastian Ramirez, 2 hops, out: depth0 Sebastian Ramirez; depth1 FastAPI; depth2 Pydantic, Starlette
Acme Inc, 1 hop: out = [-> Berlin]; in = [Alice ->, Bob Smith ->]; both = all three, each edge keeping its stored direction
analytics: 4 weakly connected components (sizes 4,4,2,2), 12 strongly connected (acyclic), top centrality Acme Inc / FastAPI (0.272727)
```
Persistence and recovery (all measured by the demo): valid current file -> adopted without rebuild; truncated, missing and tampered files -> rebuilt from SQLite, logical graph identical to the original in every case, corrupt file quarantined (10 530 bytes kept). Deleting `backend_stack.md`: 12 -> 8 nodes, 8 -> 4 edges; Alice, Pydantic, Sebastian Ramirez, Starlette removed; the shared entity kept (same ID), recomputed from remaining evidence to canonical name "Acme Corp", now supported by 1 document.

### Known limitations
- **Full rebuild per change**, not incremental: O(entities + relationships) after every ingest/delete, run under the write lock. Fine for thousands of records, not millions.
- Single-process only. The consistency protocol is correct across crashes, but two app instances sharing one data directory would race on the graph file (the revision check would make them rebuild repeatedly, not corrupt data).
- If a write succeeds in SQLite and the process dies before the graph syncs, the stale window lasts until the next read or restart (readers self-heal; nothing serves stale data).
- `degree_centrality` uses multigraph degrees (parallel edges of different types count separately) and can exceed 1; strongly connected components of an extracted knowledge graph are usually all singletons (acyclic), which is real data, not a bug.
- Shortest path is fewest hops; ties are broken by insertion order (deterministic for a given build, not semantically meaningful). No weighted paths.
- Traversal ordering favors alphabetical relationship type then entity ID; truncation keeps the closest nodes (BFS order) but is otherwise arbitrary among equals.
- Merge candidates (Phase 3) are not represented in the graph; the graph shows resolved entities only.
- The graph file is not human-editable and is not GraphML (documented rationale); no GraphML/GEXF export yet.
- Analytics are recomputed in full on the first request after each change (cached afterwards).
- Graph endpoints are read-only and unauthenticated like the rest of the API; no rate limiting yet.

### Reproduce (from the repo root unless noted)
```
backend\.venv\Scripts\python scripts\graph_demo.py
backend\.venv\Scripts\python scripts\extract_demo.py scripts\samples\backend_stack.md scripts\samples\team_notes.txt
cd backend
.venv\Scripts\python -m pytest --cov=app
.venv\Scripts\python -m ruff check . ../scripts
.venv\Scripts\python -m alembic upgrade head
.venv\Scripts\python -m uvicorn app.main:app --port 8000
```
Then upload the samples (`curl -F "file=@scripts/samples/backend_stack.md" localhost:8000/api/v1/documents/upload`) and browse `http://localhost:8000/docs` for `/api/v1/graph/*`.

## Phase 5 notes (embeddings and FAISS vector retrieval)

Design, file format, snapshot protocol and similarity semantics are in `docs/architecture.md` ("Vector embeddings and retrieval (Phase 5)"). Decisions in brief:

- **Model:** `sentence-transformers/all-MiniLM-L6-v2`, pinned to commit `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` (384 dimensions, 256-token limit, CPU, L2-normalised so inner product = cosine). The pin applies to the default model only. Cache lives under `data/models` (git-ignored, on E:).
- **Index:** exact `IndexFlatIP` FAISS behind a `VectorIndex`/`IndexEngine` protocol (nothing above `vector_index.py` names FAISS; a test enforces this). SQLite stays the source of truth; the index is rebuildable.
- **Token limit policy `window-mean-v1`:** chunks are measured with the model's own tokenizer; over-limit chunks are split losslessly into windows, embedded, token-weighted-mean pooled and re-normalised. One vector per chunk, so chunk IDs and provenance are unchanged. `token_count` and `window_count` are recorded (migration `0006`).
- **Snapshots:** the served index is an immutable `IndexSnapshot` published by one attribute assignment after validation. Searches during a rebuild use the previous snapshot (`index_stale=true`); before the first index exists they get 503 `vector_index_building`. A failed rebuild or a corrupt file never replaces the previous snapshot.
- **Fingerprint:** model, requested revision, max sequence length, normalisation and the oversize-policy id. A change discards the index and rebuilds; vectors from different configurations are never mixed.
- **Similarity:** rounded to 6 decimals; the threshold compares the rounded score to the rounded threshold; ranking uses exact scores. Similarity is retrieval similarity, not answer confidence.
- **Offline mode:** `GRAPHRAG_EMBEDDING_OFFLINE=true` uses the local cache only and fails with an explicit error; no other model is ever substituted. The default test suite uses a deterministic hashing stub and needs no network or model.

**Endpoints:** `POST /api/v1/search/vector`, `GET /api/v1/vector/index[?check=true]`, `POST /api/v1/vector/rebuild`.

### Known limitations
- Exact (flat) search is linear in the number of vectors. Measured in the benchmark: 9.23 ms median FAISS time at 100,000 synthetic vectors; no approximate index.
- Single process: two app instances sharing a data directory could race on the index file.
- Rebuild is synchronous and CPU-bound (as long as embedding the whole corpus); reads continue on the previous snapshot.
- The rule-based/hashing components are test tools. Retrieval quality was only checked on a 5-document sample corpus by inspection; no quality benchmark (recall/MRR) exists yet — that is Phase 8.
- No re-ranking, hybrid retrieval or graph expansion (Phase 6), and no answer generation (Phase 7).
- Query text longer than the token limit is windowed and pooled by the same policy; extremely long queries are capped at 2,000 characters by the API.
- English-centric model; no GPU path was tested.

### Reproduce (from the repo root unless noted)
```
backend\.venv\Scripts\python scripts\vector_demo.py
backend\.venv\Scripts\python scripts\vector_benchmark.py
cd backend
.venv\Scripts\python -m pytest --cov=app
$env:GRAPHRAG_RUN_SLOW = "1"; .venv\Scripts\python -m pytest tests/slow      # real model, opt-in
.venv\Scripts\python -m ruff check . ../scripts
.venv\Scripts\python -m ruff format --check . ../scripts
.venv\Scripts\python -m alembic upgrade head
```

## Phase 6 notes (hybrid retrieval, M0-M3)

Design and decisions: `docs/phase-6-proposal.md` (revision 2). Status below is what was built and measured, not what is planned.

### Built so far
| Milestone | Content | Verification at its gate |
|---|---|---|
| M0-M1 | `hybrid_*` settings; `retrieval/scope.py` (document scope, chunk lookup); `retrieval/fusion.py` (weighted RRF, pure); `retrieval/path_status.py` (hybrid / vector_only / graph_only / partial / failed) | 691 passed / 17 skipped |
| M2 | `retrieval/entity_linker.py` (exact, token-boundary linking), `retrieval/graph_search.py` (directed traversal, evidence, integer ranking), `POST /search/graph` | 868 passed / 17 skipped; 20/20 mutations detected |
| M3 | `retrieval/hybrid.py` (orchestrator), `schemas/hybrid.py`, `api/hybrid.py`, `POST /search/hybrid` | 961 passed / 17 skipped (+93 over M2: 28 orchestration unit tests, 49 hybrid API tests, 16 boundary cases); 97% line coverage (`hybrid.py`, `api/hybrid.py`, `schemas/hybrid.py` 100%); Ruff check + format clean; migrations: single head `0006`, no drift, round trip OK; 19/19 mutation checks detected |

No migration, no new dependency, and no change to the vector algorithm, the FAISS code or the graph services in any milestone.

### `POST /api/v1/search/hybrid` behaviour
* The graph path runs first (in memory; it also validates the graph parameters before any embedding is spent), then the vector path (`search_vectors`, unchanged). Neither receives anything from the other; fusion sees only the two ranked chunk-ID lists.
* Fusion is weighted Reciprocal Rank Fusion (`rrf_k` default 60, weights default 1.0, all per request). A chunk found by both paths is merged once by `chunk_id`; the result keeps the vector rank and similarity, and the graph rank, integer features, `decided_by` and relationship evidence, each exactly as the standalone endpoints return them (asserted by a test that compares the blocks for equality).
* `path_summary` says what happened: `hybrid` (both paths ran), `partial` (one path failed: HTTP 200, `complete: false`, `is_complete_hybrid: false`, a first warning stating "NOT a complete hybrid result", the failed path's code), and `contributing` lists the paths that actually returned results (a path that ran but found nothing, such as a query naming no graph entity, is reported as such).
* Both paths failing is 503 `retrieval_unavailable`, with each path's status, error code and message in `details`. Request errors (unknown `entity_ids`, a parameter over its cap, a query the embedder rejects) are 422 and never reported as a partial result.
* Unexpected errors inside one path are logged with a stack trace and reported with a generic message and a `*_error` code; the exception text is not returned.
* `explain=true` adds a templated, deterministic explanation per result: the fusion arithmetic, and the graph feature summary when the graph found the chunk.
* The complete vector list, graph list and fused list are kept on the service result (`HybridOutcome`) so Phase 8 can cut any of them at any K; the API returns only the fused top `limit`.

### Measured example (sample corpus, real model, 8 chunks; not a quality measurement)
"Who works at Acme Inc?": `backend_stack.md` fused 0.032787 (vector rank 1, similarity 0.6713, graph rank 1), `team_notes.txt` fused 0.032258 (vector rank 2, graph rank 2, decided by `chunk_id`), then three vector-only chunks. With the graph path forced down the same call returns the vector results with mode `partial`; with the vector path forced down it returns the two graph chunks with mode `partial`; with both forced down it returns 503. On this tiny corpus the vector path alone already ranks the same two chunks first, so these runs show the mechanics only. **No claim is made that hybrid retrieval improves on vector-only retrieval; that needs the labelled dataset of Phase 8.**

### M5 - close-out (details: `docs/phase-6-final-report.md`)
* Added `scripts/hybrid_demo.py` (vector, graph and fused views of the same queries, direction, explanations, failure modes, deletion; mechanics only). Replaced the Phase 0 retrieval sketch in `docs/architecture.md` section 4 with the implemented design and updated its status line.
* Final verification, run in sequence: Ruff check and format clean; 1111 passed / 17 skipped; 97% coverage; migrations at `0006` with no drift; 17 real-model tests passed; live API test (74 concurrent requests during uploads and deletions, 0 errors, 0 deleted-document leaks, 0 of 28 evidence items mismatching the live database, identical answer after a restart); all four demos exited 0; Git audit clean (no commits; 178 untracked source/doc/test files, none generated).
* Not established: retrieval quality (no labelled dataset; no Precision@K, Recall@K, MRR or Hit@K was computed).

### M4 - reliability and hardening (details and measurements: `docs/phase-6-m4-report.md`)
* **Consistency:** both paths share one document scope resolved once per request; the graph path confirms relationship evidence against SQLite before ranking (`dropped.stale_evidence`); the chunks about to be returned are re-checked in one read transaction after fusion, and vanished, un-completed or out-of-scope ones are dropped and replaced by the next result (`dropped.stale_results`); revision lag of the graph snapshot is reported.
* **Failure recovery:** optional per-path time budget (`GRAPHRAG_HYBRID_PATH_TIMEOUT_SECONDS`, default 30, 0 = unlimited) with one small bounded worker pool per path; a path that overruns is reported `vector_timeout` / `graph_timeout`, the other path's result is returned unchanged, and both overrunning is 503 `retrieval_unavailable`. A first design with one shared pool let a stuck path starve the other; a test caught it and it was fixed.
* **Verification at the M4 gate:** Ruff clean (144 files); 1111 passed / 17 skipped (+150 over M3); 97% coverage (`hybrid.py`, `scope.py`, `fusion.py`, `vector_search.py` 100%); migration head `0006`, no drift, round trip OK; 16/16 mutation checks detected; live test on the real model with 6 search workers during uploads and deletions: 73 requests, 0 errors, 0 deleted-document leaks, 0 of 28 relationship-evidence items mismatching the live database, same ranked answer after a restart.
* **Measured cost of the guarantees (preliminary single runs):** about +1 to 2 ms per hybrid request and one more SQL statement (hybrid 7 to 8, graph 4 to 5, vector 3 to 3); real-model service total 24.9 to 26.1 ms. Query tokenisation (0.06 ms) and ORM-versus-column lookups (0.1 ms) were measured and deliberately left alone.
* **Unresolved:** a timed-out path keeps running in the background (bounded: 2 workers per path); only `/search/hybrid` has a time budget; a first model load (15-17 s here) can exceed a short budget on a slow machine; relationship evidence is confirmed once in the graph path, so a removal within milliseconds of that check can still be returned while chunk-level validity is guaranteed at response time; embedding is serialised and uncached (about 29 requests/s with the real model); no retrieval-quality measurement.

### Known limitations (M0-M3)
* Fused scores are rank-based: not probabilities, not relevance, not comparable across queries or configurations. RRF defaults (k = 60, equal weights) are the textbook starting point and have not been tuned.
* Graph recall is limited by exact entity linking. A query that names an entity differently from the corpus contributes nothing from the graph; names are not matched across a comma or full stop, and a stored name with an internal ". " (such as "St. Louis") is not matched. Words joined by a hyphen or slash are one token.
* Graph ranking ties on all features are resolved by `chunk_id`: stable, but arbitrary with respect to relevance. Symmetric evidence (as in the sample corpus) shows this directly; in hybrid mode the independent vector rank then reorders those chunks.
* The graph rank reflects graph structure only ("close to the named entities, with more distinct relationships"); it says nothing about whether a chunk answers the question.
* The two paths are called one after the other, not in parallel. A graph read still rebuilds a stale graph synchronously, so a request right after a write can wait for the rebuild; the vector path serves the previous consistent index while it synchronizes (`vector.status: "stale"`), so the two paths can briefly reflect different revisions (both revisions are reported).
* An embedder rejection of the query (`EmbeddingInputError`) is treated as a request error (422), not as a vector-path failure, even though the graph path could still answer.
* Overlapping neighbouring chunks (100-character overlap) are different chunk IDs and can both appear; near-duplicate handling and token budgets are deferred to Phase 7.
* Single process: the alias index and the graph snapshot are in memory. No authentication or rate limiting (as in earlier phases).
* Unexercised branches (97% coverage overall): a few explanation-text branches in `api/search.py`, one `dropped_out_of_scope` counter, one `continue` in the entity linker, and the unchanged vector-route `invalid_query` branch.

### Reproduce
```
cd backend
.venv\Scripts\python -m pytest --cov=app
.venv\Scripts\python -m ruff check . ../scripts
.venv\Scripts\python -m ruff format --check . ../scripts
.venv\Scripts\python -m alembic upgrade head
.venv\Scripts\python -m uvicorn app.main:app --port 8000
```
Then upload the samples and call `POST /api/v1/search/hybrid` with `{"query": "Who works at Acme Inc?"}`; `http://localhost:8000/docs` documents the request and response.

## Phase 7 notes (grounded answer generation)

Design and decisions: `docs/phase-7-proposal.md`. Status below is what was built and measured.

### M0 - foundations (complete; approved)
* **Settings** (`GRAPHRAG_LLM_*`, `GRAPHRAG_ANSWER_*`, `GRAPHRAG_QUERY_HISTORY`, in `.env.example`, checked by a test): default provider `extractive`; a provider URL must be loopback unless `GRAPHRAG_LLM_ALLOW_REMOTE=true`; the API key is never a setting (only the NAME of an environment variable); the context budget plus answer tokens plus 400 tokens of prompt overhead must fit the model window.
* **Provider interface** (`services/generation`): `LLMProvider` protocol, request/result types whose `repr` never shows text, an error taxonomy with stable codes (`llm_not_configured`, `llm_unavailable`, `llm_timeout`, `llm_bad_response`, `llm_context_too_long`, `generation_busy`), a conservative token estimator, a provider registry and a lazily-built `LLMRuntime` injected through `app.state` and `get_llm_provider`. At the M0 boundary the registry was intentionally empty; M1 now registers the shipped providers.
* **Schemas** for `/api/v1/query` and the history (`schemas/query.py`): the caller cannot choose provider, model or URL; no answer-level confidence field exists; an answer is present only when `status` is `answered`.
* **Query history** on the existing `query_history` table (checked against a database built by the real migrations): the table is sufficient, no migration. Constraints that follow: no status/provider/model columns (kept in `metadata`), `answer` is NOT NULL (unanswered stored as empty), VARCHAR(16) lengths enforced in code because SQLite does not, metadata bounded to 64 KB, excerpt snapshots of 300 characters.
* M0 did not change query behavior or the database schema.

### M1 - providers (implementation complete; awaiting review)
* Registered the mock, extractive, Ollama and OpenAI-compatible providers. The extractive provider remains the default; no model or network is needed for it.
* Both HTTP providers map the generation request and response, classify malformed responses and context-limit failures, avoid echoing response bodies, support optional environment-based bearer keys, and share transport/timeout/5xx retry behavior with short backoff bounded by the overall request deadline. 4xx responses are not retried.
* Provider tests use scripted responses and `httpx.MockTransport`; no real endpoint or model was used. Full suite: 1,465 passed, 17 skipped. Ruff check and formatting passed. Alembic upgrade and drift check passed on a disposable SQLite database; head remains `0006`.
* **Not built yet:** context building (M2), parsing and grounding (M3), the `/query` and history endpoints (M4, M5), or any answer-generation pipeline.
