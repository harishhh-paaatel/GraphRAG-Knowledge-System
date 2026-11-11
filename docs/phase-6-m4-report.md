# Phase 6 — M4 Completion Report: Hybrid Retrieval Reliability and Hardening

Date: 2026-11-06 · Status: **M4 implemented and verified — awaiting your review and approval.**
Nothing was committed or pushed. M5 has not been started.

Every number below was measured in this session by running the command shown. Benchmarks are **preliminary single runs on one machine** (not repeated, machine not isolated); they measure latency and query counts, never retrieval quality.

## 1. Summary

M4 hardened the M3 hybrid endpoint without changing its contract or the vector and graph algorithms. The work was driven by tests and measurements: five defects or gaps were found (section 3), one of them a flaw in my own first M4 design, and each is now covered by a regression test that fails when the fix is removed.

| Check | Result |
|---|---|
| Ruff check and format check | Clean (144 files) |
| Full suite | **1111 passed, 17 skipped, 0 failed** (961 before M4, +150) in 146 s |
| Coverage | **97 %** overall (4,820 statements, 142 missed). `hybrid.py` 100 %, `vector_search.py` 100 %, `scope.py` 100 %, `fusion.py` 100 %, `api/hybrid.py` 100 %, `graph_search.py` 98 %, `entity_linker.py` 99 %, `api/search.py` 94 % |
| Migrations | Head `0006`, no drift, 0006→0001→0006 round trip OK. **No migration was added.** |
| Mutation checks | **16 of 16 detected** (M4 code); sources restored byte-identical (SHA-256 checked after every mutation) |
| Live API test (real model, real uvicorn) | Passed — section 6 |
| Stability | New timing- and thread-sensitive test files were run three times each with identical results |

## 2. What changed

**Source (all additive or behaviour-preserving for the existing endpoints):**

| File | Change |
|---|---|
| `services/retrieval/hybrid.py` | One shared document scope for both paths; final verification of the fused results; optional per-path time budgets with one bounded worker pool per path; revision-lag and stale-result warnings; `close()` on shutdown |
| `services/retrieval/graph_search.py` | Relationship evidence is confirmed against SQLite before ranking (`_verified`); optional shared `scope` |
| `services/retrieval/vector_search.py` | Optional shared `scope` parameter only. The search algorithm is unchanged |
| `services/retrieval/scope.py` | `ChunkRecord` now carries the stored `processing_status` |
| `api/hybrid.py`, `api/search.py`, `schemas/graph_search.py` | `dropped` gained `stale_evidence` and `stale_results` (additive, default 0); warnings for them |
| `core/config.py`, `.env.example`, `main.py` | `GRAPHRAG_HYBRID_PATH_TIMEOUT_SECONDS` (default 30, 0 = unlimited); the hybrid retriever is closed at shutdown |
| `scripts/hybrid_benchmark.py` | New: latency and SQL-statement benchmark |

**Tests (+150):** `test_hybrid_hardening.py` (22), `test_fusion_reference.py` (72), `test_hybrid_consistency.py` (12), `test_hybrid_integrity.py` (21), `test_hybrid_reliability.py` (20), plus 2 scope tests and 1 settings test. Existing M0–M3 tests were adapted where behaviour deliberately changed (section 3, items 2 and 3).

**Reports:** four benchmark JSONs in `evaluation/reports/phase6_m4_benchmark_{before,after}_{hashing,real_model}.json`.

## 3. Defects and gaps found, and how each is now covered

1. **A stuck path could starve the other (found by my own test, in my first M4 design).** The first timeout implementation used one shared worker pool. Two hung graph calls filled it, the vector path was queued behind them, and a request that should have returned the vector result returned 503. Fixed with one small bounded pool per path. Regression tests: `test_stuck_paths_cannot_pile_up_threads_or_starve_the_other_path` and its mirror for the vector path; the shared-pool mutation (X10) is detected.
2. **Stale relationship evidence could be returned (gap in M3).** If a document was re-processed or its relationships removed after the graph snapshot was built, but the chunk still existed, M3 returned the evidence. The graph path now confirms every evidence item (same row id, same chunk, same document) against `relationships` in the same read transaction as the chunk lookup, and re-collects without the stale items. Counted as `dropped.stale_evidence`. Costs one query. Tests: three scenarios in `test_hybrid_consistency.py`; mutations X1 and X2 detected. The two older tests that expected the earlier symptom (`missing_chunks == 1`) were updated to the earlier, more precise detection.
3. **Two document scopes per request (consistency gap in M3).** Each path resolved the document scope itself, at slightly different moments. The orchestrator now resolves it once and passes it to both. This also removed one query. Tests: scope call count, `test_a_shared_scope_gives_exactly_the_same_vector_search...`; mutations X8, X15, X16 detected.
4. **A chunk deleted between a path's hydration and the response could be returned (M3 window of milliseconds, or longer for the stale vector index).** The chunks about to be returned are now looked up once more in one read transaction; vanished chunks, documents that left `completed`, and chunks outside the requested documents are dropped, the next fused result takes their place, and the content, filename, page and section returned are the stored ones. Counted as `dropped.stale_results`. Tests: deletion, status change and content-change races driven deterministically; concurrent ingestion and deletion; mutations X3–X7 detected.
5. **No protection against a path that hangs.** A hung model load or a long graph rebuild would hang the request. An optional per-path time budget now turns that into a labelled `partial` result (`vector_timeout` / `graph_timeout`), or 503 `retrieval_unavailable` if both overrun. Tests: timeouts, recovery, a path's own `TimeoutError` not mistaken for the budget, bounded threads; mutations X9, X11, X12 detected.

Also found, but a test-fixture problem rather than an application defect: hand-built test documents never bumped the vector revision (real ingestion does), so the vector index never synchronised. Fixed in the test helpers (`add_indexed_doc`).

## 4. Measurements

**Method:** `scripts/hybrid_benchmark.py`, in-process TestClient, real graph service, real FAISS index, synthetic corpus ("PersonN works at OrgM. OrgM is located in CityK."). 60 requests per endpoint; latency is client-side (includes about 8 ms of fixed TestClient/serialisation overhead); "service" is the service's own reported total.

| Corpus (embedder) | Endpoint | Before M4 median (p95) ms | After M4 median (p95) ms | SQL statements before → after |
|---|---|---|---|---|
| 200 docs (hashing stub) | vector | 10.8 (14.1) | 10.3 (11.6) | 3 → 3 |
| | graph | 13.4 (15.6) | 11.8 (13.5) | 4 → 5 |
| | hybrid | 15.7 (21.6) | 17.2 (19.2) | 7 → 8 |
| 800 docs (hashing stub) | hybrid | 19.5 (24.2) | 21.4 (44.7) | 7 → 8 |
| 200 docs (real model) | vector | 24.6 (27.9) | 25.2 (26.8) | 3 → 3 |
| | graph | 12.3 (14.4) | 12.2 (14.7) | 4 → 5 |
| | hybrid | 32.5 (37.4) | 33.7 (35.8) | 7 → 8 |

- **Cost of the M4 guarantees:** about **+1 to 2 ms per hybrid request** (+1 SQL statement: the relationship-evidence check and the final verification add two, the shared scope removes one). Service total with the real model: 24.9 → 26.1 ms (+5 %). The final verification step itself measured about 1.2–1.5 ms. The per-path time budget (thread hop) was measured separately by toggling `GRAPHRAG_HYBRID_PATH_TIMEOUT_SECONDS` between 0 and 30: about +0.4 ms.
- **Where the time goes (real model, hybrid service total 26 ms):** vector 17.8 ms (query embedding about 13.4 ms of it), graph 5.6 ms, fusion 0.26 ms, verification 1.5 ms.
- **Redundant work check:** the alias index was built once per corpus across every run (`alias_index_builds = 1`); a test asserts that repeated searches trigger no graph rebuild and no vector re-sync, that a hybrid request embeds the query exactly once and re-embeds no chunk, and that concurrent first requests build each index once. The model is loaded once (first request 15.5–17.0 s, dominated by loading; every later request is steady-state).
- **Query count does not grow with results or corpus size:** a test asserts identical statement counts for 1 result and 25 results; the benchmark shows the same counts at 50, 200 and 800 documents.
- **Concurrency (4 workers):** real model 29.2 requests/s before and after (CPU-bound on the single embedding lock); hashing stub 63 → 51 requests/s at 200 documents.

**Measured, then deliberately not optimised:**
- Query tokenisation (`describe`) is 0.06 ms against 13.4 ms for the model pass: the double tokenisation is irrelevant.
- Selecting columns instead of ORM objects in `lookup_chunks` saves about 0.1 ms per call (0.52 → 0.42 ms median): below noise, not worth changing a shared helper.
- The second read of the graph revision (`status()`) looks redundant but is the staleness signal reported to the caller, so it stays.
- A query-embedding cache would remove roughly half the hybrid latency for repeated identical queries, but no repeated-query workload was measured, so it is **not** implemented (listed under limitations).

## 5. Test coverage of the six areas you listed

1. **Consistency:** stale graph with deleted or re-pointed relationships; document deleted, un-completed or edited between the two paths; stale vector index together with a stale graph; unfinished documents invisible to every endpoint; concurrent ingestion (3 readers, 1 writer) and concurrent deletion (2 readers, 1 deleter) with every response validated.
2. **Failure recovery:** per-path time budgets (vector, graph, both), a hung path costing one budget while the other result returns, a stuck path unable to starve the other, bounded worker threads, recovery after a slow path becomes fast, alternating failure/health without residue (the healthy answer after the failures is identical to the answer before them), shutdown releasing threads, zero timeout disabling the pool.
3. **Ranking correctness:** 72 randomised cases checked against an independent exact-arithmetic (`Fraction`) reference of weighted RRF: order, scores, original ranks, contiguity, duplicates merged once, deterministic tie-breaking, one-path-empty reduction; plus the M1–M3 unit tests.
4. **Evidence integrity:** a checker that compares every claim in a response with the database (chunk exists, document completed and in the requested set, filename/page/section/chunk index stored values, relationship row exists with the same chunk, document, source and target entity ids, type, supporting text, provenance and confidence, mention rows exist) run in 8 hybrid modes, 8 graph modes, with document filters, with each path down, and with failed documents.
5. **Performance:** SQL statement budgets (vector ≤ 3, graph ≤ 5, hybrid ≤ 8) as regression tests, no N+1, no repeated embedding, no repeated rebuilds or syncs (section 4).
6. **API reliability:** malformed JSON, wrong content type, mistyped fields, oversized lists, hostile and unusual queries (NUL byte, right-to-left override, emoji, SQL and HTML text, about 2,000 characters), wrong methods and paths, error envelope, status reporting, 96 mixed concurrent requests that must equal the serial answers, 6 concurrent first requests, requests during bulk ingestion.

## 6. Live API test (real uvicorn server, real model, cached offline)

`alembic upgrade head`, server start in 2.0 s, 17 documents ingested. Then 6 search workers (hybrid, graph and vector, rotating) ran **while** 6 documents were uploaded and 3 were deleted through the API:

- 73 requests in 2.6 s, **0 errors**, 0 partial results.
- **0** responses contained a document whose DELETE had already returned 204 before the request began.
- Latency under that load (embedding competes with ingestion for the CPU): graph median 16 ms (p95 70), vector 211 ms (p95 469), **hybrid 315 ms (p95 618)**. Idle hybrid latency is about 26–34 ms; the load figure is a contention measurement, not a regression.
- 13 of 24 hybrid responses had a stale vector index at request time and correctly served the last consistent index (`status: stale`, no error).
- The verification step dropped nothing in the live run: the races it protects against are rare and are exercised deterministically by the integration tests.
- After ingestion settled: mode `hybrid`, no warnings, vector index and graph both consistent, 17 of 20 results found by both paths, 28 relationship evidence items checked against the live SQLite file with **0 mismatches**.
- After a restart: the same question returned the same ranked chunks; the vector index loaded from its file and was consistent.

## 7. Unresolved limitations

- **A timed-out path cannot be killed.** Python cannot interrupt it; it keeps running in the background. Each path has two workers, so after two hung calls that path's later requests time out immediately (queued, then cancelled) until a hung call finishes. The other path is unaffected, and the failure is visible in `paths.*`. Only `/search/hybrid` has a time budget; `/search/graph` and `/search/vector` do not.
- **The default 30 s budget can be shorter than a first model load on a slow machine** (about 15–17 s here). In that case the first hybrid request returns a `partial` graph-only result with `vector_timeout`, and the load continues in the background. `.env.example` says so.
- **A graph read still waits for a graph rebuild** after a write (inside the budget); a rebuild longer than the budget shows as `graph_timeout`.
- **A residual window of milliseconds remains for relationship evidence:** it is confirmed once, in the graph path, and the final verification re-checks chunks, documents and scope but not relationship rows again. A relationship removed (while its chunk survives) in that window could still be returned. Chunk-level validity is guaranteed at response time; relationship-level validity at the time of the graph check.
- **Scores are as of each path's query time:** vector similarity and graph features are not recomputed by the final verification; only existence, scope and stored text are.
- `candidates_fused` and the complete fused list on the service result are counted before the final verification.
- **Embedding is serialised**, so concurrent throughput with the real model stays about 29 requests/s; query embedding is not cached.
- **Latency under heavy concurrent ingestion is high** (hybrid p95 0.6 s in the live test) because ingestion and queries share one CPU; not optimised.
- Benchmarks are single preliminary runs; client-side latencies include fixed TestClient overhead; timing-sensitive tests use real sleeps (a few seconds in total) and could be flaky on a very slow machine, though they were stable across repeated runs.
- No retrieval-quality measurement exists; nothing here claims hybrid retrieval is better than vector-only retrieval (needs the labelled dataset of Phase 8).
- The earlier M3 limitations (exact entity linking recall, arbitrary `chunk_id` tie-breaks on symmetric evidence, no near-duplicate handling, single process, no authentication or rate limiting) are unchanged.

## 8. Reproduce

```
cd backend
.venv\Scripts\python -m ruff check . ../scripts
.venv\Scripts\python -m ruff format --check . ../scripts
.venv\Scripts\python -m pytest --cov=app
.venv\Scripts\python -m alembic upgrade head
cd ..
backend\.venv\Scripts\python scripts\hybrid_benchmark.py
backend\.venv\Scripts\python scripts\hybrid_benchmark.py --provider sentence_transformers --sizes 200
```

## 9. Status

**M4 is complete pending your approval.** M5 has not been started.
