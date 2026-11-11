# Phase 6 Final Report — Hybrid Retrieval (Vector + Knowledge Graph)

Date: 2026-11-06 · Status: **Phase 6 implementation and verification complete — awaiting your final review and approval.**
Nothing was committed or pushed. Phase 7 has not been started.

All numbers were measured in this session by running the command shown. Benchmarks are preliminary single runs on one machine.

> **Retrieval-quality improvement has NOT been established.** This phase built and verified the mechanics of hybrid retrieval: correctness, provenance, consistency and failure handling. No labelled dataset exists, no Precision@K, Recall@K, MRR or Hit@K was computed, and nothing in this report shows that hybrid retrieval returns better results than vector-only retrieval. On the 8-chunk sample corpus the vector path alone already ranks the same documents first. That question belongs to Phase 8.

## 1. What Phase 6 delivered

| Endpoint | Behaviour |
|---|---|
| `POST /api/v1/search/graph` | Graph-only retrieval: exact entity linking, bounded directed traversal (`outgoing` / `incoming` / `both`), evidence chunks ranked by integer graph features. Needs no vector index, FAISS or embedding model |
| `POST /api/v1/search/hybrid` | Runs the vector path and the graph path independently, fuses their rankings with weighted Reciprocal Rank Fusion, merges duplicates by `chunk_id`, and labels the response from the state of both paths |
| `POST /api/v1/search/vector` | Unchanged (Phase 5) |

New modules under `backend/app/services/retrieval/`: `scope.py` (shared document scope and chunk lookup), `fusion.py` (pure weighted RRF), `path_status.py` (what a response may call itself), `entity_linker.py`, `graph_search.py`, `hybrid.py`; plus `schemas/graph_search.py`, `schemas/hybrid.py`, `api/hybrid.py`, six `GRAPHRAG_HYBRID_*` settings. **No database migration, no new dependency, no new infrastructure.** The vector search algorithm, the FAISS code and the graph services were not changed (the vector search gained one optional `scope` parameter).

Milestones, each reviewed and approved before the next: M0–M1 foundations and fusion; M2 entity linker, graph retrieval, `/search/graph`; M3 hybrid integration; M4 reliability and hardening (`docs/phase-6-m4-report.md`); M5 this close-out (demo, architecture documentation, final verification, this report).

## 2. Architecture changes

Described in `docs/architecture.md` section 4 (rewritten; the Phase 0 sketch is gone). In short:

- **Independent paths, rank-only fusion.** The graph path imports nothing from embeddings, FAISS or vectors; the vector path imports nothing from NetworkX or the graph package; only `hybrid.py` knows both. Import-boundary tests enforce this. Fusion consumes two ranked chunk-ID lists and never compares scores.
- **Graph ranking uses no similarity.** Lexicographic integer key `(-seed_coverage, -connects_seeds, -relationship_evidence, distance, -support, -corroboration, -provenance_known, chunk_id)`; every result reports its features and `decided_by`. Ties on every feature fall to the unique `chunk_id`: reproducible, but arbitrary with respect to relevance.
- **Conservative exact entity linking.** Whole-token, normalised matching of names and aliases; ambiguity is reported with every candidate and never guessed; explicit `entity_ids` resolve it. No fuzzy matching, no LLM.
- **Direction preserved.** `source → target` is returned exactly as stored in every traversal mode.
- **Explicit path status.** `hybrid` (both paths ran), `partial` (one failed: HTTP 200, `complete: false`, first warning states the result is not a complete hybrid), 503 `retrieval_unavailable` (both failed). Request errors are 422 and never reported as a path failure.
- **Consistency (M4).** One shared document scope; relationship evidence confirmed against SQLite; the returned chunks re-checked in one read transaction after fusion; per-path time budgets with one bounded worker pool per path.

Deviations from the approved proposal (all agreed during review or recorded in the milestone reports): the vector tie-break for graph ranking was withdrawn at your request; the "include all candidates" ambiguity mode was removed; milestones M5–M7 of the proposal were collapsed because explainability (M3) and hardening (M4) were delivered earlier; the graph path runs before the vector path so a refused request costs no embedding; the per-path time budget (not in the proposal) was added in M4.

## 3. Verification results (this run, in this order)

| Step | Command / method | Result |
|---|---|---|
| Lint | `ruff check . ../scripts` | All checks passed |
| Format | `ruff format --check . ../scripts` | 145 files already formatted |
| Default suite | `pytest` (hashing stub, no network or model) | **1111 passed, 17 skipped, 0 failed** in 179 s |
| Coverage | `pytest --cov=app` | **97 %** overall (4,820 statements, 142 missed). `hybrid.py`, `vector_search.py`, `scope.py`, `fusion.py`, `path_status.py`, `api/hybrid.py`, `schemas/hybrid.py` 100 %; `graph_search.py` 98 %; `entity_linker.py` 99 %; `api/search.py` 94 % |
| Migrations | `mig_check.py` on a temporary database | Chain 0001→0006, single head `0006`, **no drift**, downgrade to 0001 and re-upgrade OK |
| Real-model tests | `GRAPHRAG_RUN_SLOW=1 pytest tests/slow` (cached model) | **17 passed** |
| Live API test | real uvicorn + real model, see section 4 | Passed |
| Demos | `extract_demo.py`, `graph_demo.py`, `vector_demo.py`, `hybrid_demo.py` | All exited 0 |
| Git audit | `git status`, `git check-ignore` | Clean — section 7 |

The 17 skipped tests are the opt-in real-model tests; they were run separately and passed.

Test count by milestone: Phase 5 538 → M0–M1 691 → M2 868 → M3 961 → M4 1111 passed. Mutation checks (apply one deliberate fault, require a test to fail, restore the file byte-identical): **65 of 65 detected** (M1 10, M2 20, M3 19, M4 16). Two M2 survivors exposed missing tests and were closed before that count.

## 4. Live API test (real server, real model, cached offline)

17 documents ingested; then 6 search workers (hybrid, graph, vector rotating) ran while 6 documents were uploaded and 3 deleted through the API:

- 74 requests in 2.2 s, **0 errors**, 0 partial results.
- **0** responses contained a document whose `DELETE` had already returned 204 before the request began.
- 11 of 25 hybrid responses saw a stale vector index at request time and correctly served the last consistent one.
- After ingestion settled: mode `hybrid`, no warnings, vector index and graph consistent, 28 relationship-evidence items checked against the live SQLite file with **0 mismatches**.
- After a restart: the same ranked chunks; the index loaded from its file and was consistent.
- Latency under that load (queries compete with ingestion for one CPU): graph median 16 ms (p95 54), vector 158 ms (p95 252), hybrid 263 ms (p95 537). This is a contention measurement, not idle latency.

## 5. Demo output (`scripts/hybrid_demo.py`, real model, 8-chunk sample corpus)

The same query through the three endpoints (excerpt from this run):

```
Q: 'Who works at Acme Inc?'    entities linked: ['Acme Inc']   graph status: ok
  vector only : backend_stack.md 0.671323 · team_notes.txt 0.487289 · backend_stack.md 0.154783
  graph only  : backend_stack.md rank 1 · team_notes.txt rank 2 (decided by chunk_id)
  HYBRID      : mode=hybrid contributing=['vector', 'graph']
     #1 fused 0.032787 backend_stack.md  vec#1 sim 0.671  graph#1
     #2 fused 0.032258 team_notes.txt    vec#2 sim 0.487  graph#2
     #3 fused 0.015873 backend_stack.md  vec#3 sim 0.155  graph -
Q: 'How do I make bread?'      entities linked: none   graph status: no_entities
  HYBRID      : mode=hybrid contributing=['vector']  (+ warning: the graph path ran but contributed no results)
```

Direction (`Acme Inc`): `outgoing` → `Acme Inc -LOCATED_IN-> Berlin`; `incoming` → `Alice -WORKS_AT-> Acme Inc`, `Bob Smith -WORKS_AT-> Acme Inc`; `both` → all three, each edge keeping its stored direction.

Failure modes (injected the way the tests inject them): graph down → HTTP 200 `partial`, `complete=False`, vector chunks; vector down → `partial`, graph chunks; both down → HTTP 503 `retrieval_unavailable` with `{'vector': 'vector_index_building', 'graph': 'graph_unavailable'}`; recovered → `hybrid` again; graph path hung with a 2 s budget → `partial` returned after 2.0 s. A `DELETE` of `team_notes.txt` removed it from both paths. Hybrid service latency in that run: median total 19.3 ms (graph 3.1, vector 14.4, fusion 0.09, verification 1.0).

What this shows: the mechanics work and every claim carries its evidence. What it does not show: that the fused order is better than the vector order. In these examples the fused top results are the same chunks the vector path already ranks first.

## 6. Benchmark conditions and results

`scripts/hybrid_benchmark.py` (M4), in-process TestClient, real graph service, real FAISS index, synthetic corpus ("PersonN works at OrgM. OrgM is located in CityK."), 60 requests per endpoint, **single preliminary run**, same machine as the Phase 5 benchmark (Windows 11, AMD64 Family 25 Model 68, 16 logical CPUs, Python 3.12.0, torch 2.15.0+cpu, faiss 1.15.1, CPU only). Client-side latencies include about 8 ms of fixed TestClient overhead. JSON reports: `evaluation/reports/phase6_m4_benchmark_*.json`. Latency and query counts only.

| 200 documents | Median ms (p95) before M4 → after M4 | SQL statements per request |
|---|---|---|
| vector, real model | 24.6 (27.9) → 25.2 (26.8) | 3 → 3 |
| graph | 12.3 (14.4) → 12.2 (14.7) | 4 → 5 |
| hybrid, real model | 32.5 (37.4) → 33.7 (35.8) | 7 → 8 |

The M4 integrity guarantees cost about +1 to 2 ms per hybrid request and one SQL statement. Real-model service total: 24.9 → 26.1 ms, of which query embedding is about 13.4 ms. First request after start-up (model load) 15.5–17.0 s. 4-worker concurrent throughput with the real model: 29 requests/s (CPU-bound on the single embedding lock). The alias index and the vector index are built once; a hybrid request embeds the query exactly once.

## 7. Git audit

The repository has **no commits** (`git ls-files` is empty). 178 files are untracked and not ignored; none matches a generated-artifact pattern (`__pycache__`, `.pyc`, coverage, caches, `.bin`, databases, `.faiss`, `.corrupt`, logs, safetensors, venvs, temp or backup files). `git check-ignore` confirms the virtual environment, the model cache, the databases, the index and graph files, the caches, `.coverage` and `.env` are ignored. Benchmark and verification scratch data lives under `E:\graphrag-cache`, outside the repository. Nothing was added, staged or committed.

## 8. Unresolved limitations

- **Retrieval quality is unmeasured** (see the statement at the top). Fusion defaults (k = 60, equal weights) are untuned.
- **Graph recall is limited by exact linking.** A query that names an entity differently from the corpus gets nothing from the graph. A comma or full stop ends a name; words joined by a hyphen or slash are one token; a stored name with an internal ". " ("St. Louis") is not matched; names over 8 words are not indexed. The rule-based extractor is a demo tool, so the graph itself is small and shallow.
- **Graph rank ties** on every feature are resolved by `chunk_id`: stable but arbitrary. Symmetric evidence, as in the sample corpus, shows this directly.
- **Timeouts cannot kill a running path.** A timed-out call keeps running in the background (two workers per path); only `/search/hybrid` has a time budget. The default 30 s budget can be shorter than a first model load on a slow machine, giving a `partial` result on the first request.
- **Relationship evidence is confirmed once**, inside the graph path. A relationship removed within milliseconds of that check, while its chunk survives, could still be returned. Chunk-level validity is guaranteed at response time.
- **A graph read waits for a rebuild** after a write; the vector path can serve a briefly stale index (reported, not an error).
- **Throughput:** embedding is serialised and queries are not cached (about 29 requests/s with the real model); hybrid p95 reached 0.54 s under concurrent ingestion on one CPU.
- A query the embedder rejects (for example only punctuation) is a 422 request error for hybrid too, even though the graph path could answer.
- Neighbouring chunks overlap by 100 characters and can both be returned; near-duplicate handling and token budgets are deferred to Phase 7. Single process, no authentication, no rate limiting.
- Timing-sensitive tests use real sleeps (a few seconds in total); they were stable across repeated runs but could be flaky on a very slow machine.

## 9. Remaining technical debt

- A query-embedding cache (keyed by the embedding fingerprint) would remove roughly half the hybrid latency for repeated queries; not built because no repeated-query workload was measured.
- A time budget for `/search/graph`, and a response-time re-check of relationship evidence, were left out deliberately (small gain for added complexity).
- The deprecation warning from `sentence-transformers` (`get_sentence_embedding_dimension`) and the Starlette `TestClient`/`httpx2` notice remain.
- `api/search.py` has the lowest coverage of the new surface (94 %): a few explanation-text branches.
- The repository's line endings are CRLF; new files were normalised by tooling and should be checked once when a first commit is prepared.

## 10. Completion status

**Phase 6 is complete pending your approval.** All milestones M0–M5 were executed and verified as listed above; the failed, skipped and unexecuted items are stated, not hidden: the 17 real-model tests are skipped by default and were run separately; retrieval quality, multi-process deployment, GPU use and other platforms were not evaluated. Phase 7 has not been started and nothing was committed or pushed.
