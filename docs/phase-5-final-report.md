# Phase 5 Final Report — Embeddings and FAISS Vector Retrieval

Date: 2026-11-05 · Status: **Phase 5 implementation and verification complete — awaiting your review and approval.**
Nothing was committed or pushed. Phase 6 has not been started.

All numbers below were measured in this session by running the commands shown; none are estimated. Raw outputs: scratchpad files from the verification chain, `evaluation/reports/phase5_vector_benchmark.json` (the post-fix benchmark). `..._run1.json` and `..._run2.json` in the same folder are earlier benchmark runs from before the reliability fixes and are kept as history.

## 1. Architecture and implementation summary

Full design: `docs/architecture.md` → "Vector embeddings and retrieval (Phase 5)". Plan and notes: `docs/implementation-plan.md`.

- **Source of truth:** SQLite. The FAISS index (`data/embeddings/vector_index.bin`) is a derived, rebuildable artifact. `vector_state.revision` is bumped in the same transaction as any chunk change; the index file records the revision it was built from.
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`, pinned to commit `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; 384 dimensions; 256-token input limit; CPU; L2-normalised so inner product = cosine similarity.
- **Index:** exact `IndexFlatIP` behind `VectorIndex` / `IndexEngine` protocols; `FaissFlatIPEngine` is the only engine and nothing above `vector_index.py` names FAISS (enforced by a test). No NumPy stand-in exists.
- **Token-limit policy `window-mean-v1`:** the model's tokenizer counts tokens; over-limit chunks are split losslessly (sentence → word → character), embedded, token-weighted-mean pooled, re-normalised. One vector per chunk, so chunk IDs and provenance are unchanged; `token_count` and `window_count` are stored (migration `0006`).
- **Snapshots:** the served index is an immutable `IndexSnapshot` published by one assignment after `validate()`. Rebuilds are built privately; a failed rebuild or corrupt file never replaces the previous snapshot.
- **Fingerprint:** model, requested revision, max sequence length, normalisation, oversize policy id. Any change discards the index and rebuilds (old and new vectors are never mixed).
- **Similarity:** rounded to 6 decimals; threshold compares rounded to rounded; ranking uses exact scores; it is retrieval similarity, **not** answer confidence.
- **Offline mode:** `GRAPHRAG_EMBEDDING_OFFLINE=true` uses the local cache only; a missing cache or revision gives an explicit error and no other model is substituted.
- **Endpoints:** `POST /api/v1/search/vector`, `GET /api/v1/vector/index[?check=true]`, `POST /api/v1/vector/rebuild`.

## 2. Completed capabilities (Phase 5A reliability items)

| # | Requirement | Result |
|---|---|---|
| 1 | Silent truncation | Detected with the real tokenizer, recorded, handled by `window-mean-v1`; 28 windowing unit tests incl. boundary cases; a real-model test shows truncation was real and the policy removes it |
| 2 | Last-known-good search during ingestion | Reads take a non-blocking lock check and serve the previous compatible snapshot with `index_stale=true`; 503 `vector_index_building` before any index exists; concurrency tests |
| 3 | Atomic index+metadata replacement | `IndexSnapshot` + `validate(expected)` before publish; failed rebuild keeps the previous snapshot; corrupt files quarantined as `.corrupt`; regression tests |
| 4 | Score policy | `round_similarity` / `passes_threshold` with boundary tests; docs state similarity ≠ confidence |
| 5 | Honest vector abstraction | Protocols implemented and used consistently; no fake implementation |
| 6 | Pinned revision | Pin in config (40-hex validated), verified against the loaded snapshot, included in the fingerprint; mismatch triggers rebuild; tested without any external update |
| 7 | Offline behaviour | Cached-offline, missing-cache-offline, clear errors, no silent fallback; the default suite has no internet or model dependency |

Docs (5B): `docs/architecture.md`, `docs/implementation-plan.md`, `.env.example` updated; `.env.example` is checked by a test against the real `Settings` fields.

## 3. Verification results (5C, run sequentially)

| Step | Command / method | Result |
|---|---|---|
| 1 Lint | `ruff check . ../scripts`; `ruff format --check . ../scripts` (from `backend/`) | All checks passed; 115 files already formatted |
| 2 Default suite | `pytest` (hashing stub, no network/model) | **538 passed, 17 skipped, 0 failed** in 82.7 s |
| 3 Coverage | `pytest --cov=app` | **96 % total** (3571 statements, 135 missed). Vector modules: `service.py` 95 %, `embedder.py` 96 %, `windowing.py` 98 %, `vector_index.py` 97 %, `vector_store.py` 93 %, `snapshot.py` 79 %, `vector_search.py` 97 %, `api/search.py` 96 % |
| 4 Migrations | `mig_check.py` on a temp DB | Chain 0001→0006, single head `0006`, **no drift vs ORM**, downgrade to 0001 and re-upgrade OK |
| 5 Real-model tests | `GRAPHRAG_RUN_SLOW=1 pytest tests/slow` (cached model, `HF_HOME` not exported) | **17 passed** in 16.3 s |
| 6 Live FastAPI smoke (real model) | uvicorn + HTTP client, throwaway data dir | Passed — details in §4 |
| 7 Vector API tests | search / reliability / snapshot / service integration files | **90 passed** in 11.5 s (also part of step 2) |
| 8 Demos | `extract_demo.py`, `graph_demo.py`, `vector_demo.py`, `vector_benchmark.py` | All exited 0 (see §4–§6). `extract_demo.py` first exited 2 because I ran it without file arguments; re-run with the sample files exited 0 |
| 9 Delete / stale / rebuild | live smoke + `vector_demo.py` | Passed — §4 |
| 10 Git audit | `git status`, `git check-ignore`, untracked-file listing | Clean — §8 |

Earlier in 5A, five mutation checks (windowing disabled, raw-threshold compare, unverified pin, incompatible snapshot served while rebuilding, reads blocking on the sync lock) were each detected by failing tests, and sources were restored byte-identical. The M5 run was re-done under a hard time guard and exited normally.

## 4. Real-model retrieval demonstration

Live server, 5 sample documents (8 chunks), real model, revision pinned. Index state `ok`, 8 vectors, 384 dims, `max_tokens` 256, `oversized_chunks_pooled` 0, consistency check memory+file both consistent. HTTP results (top hit first, cosine similarity):

| Query | Top results |
|---|---|
| What is the capital of France? | geography.txt 0.5389 · backend_stack.md 0.1392 · team_notes.txt 0.1258 |
| How do I make bread? | kitchen_notes.txt 0.5729 · backend_stack.md 0.1219 · backend_stack.md 0.0850 |
| How does cosine similarity compare two embeddings? (demo) | vector_search_notes.md §Cosine similarity 0.6271 · §Vector search 0.4844 · §Rebuilding 0.3274 |
| Which technologies are used to build Python APIs? | team_notes.txt 0.3225 · backend_stack.md 0.2591 · backend_stack.md 0.0373 |
| Who works in Berlin? | backend_stack.md 0.4899 · geography.txt 0.4858 · team_notes.txt 0.3660 |

**Honest reading:** the first three queries find the intended document clearly. The last two are weaker: for "Python APIs" the top hit is `team_notes.txt` (which mentions SQLAlchemy and Python) ahead of `backend_stack.md` (FastAPI), and for "Who works in Berlin?" the Berlin geography text scores almost level with the team chunk. This corpus is 8 short chunks, so this is an inspection of a handful of queries, **not a retrieval-quality measurement**; no recall/MRR was computed (planned for Phase 8).

Other API behaviour observed: threshold 0.95 → 0 results; `document_ids` filter → only that document; blank query → 422; `top_k=0` → 422. After `DELETE /documents/{id}` for `kitchen_notes.txt` (204), the bread query no longer returned it, revision advanced 6→7, and the consistency check stayed consistent.

**Stale detection and rebuild (live):** with the server stopped I deleted `team_notes.txt` rows directly in SQLite and bumped `vector_state.revision`; the server was restarted with `GRAPHRAG_VECTOR_SYNC_ON_STARTUP=false`. `GET /vector/index?check=true` reported state `stale`, the deleted chunk under `extra_in_index`, and `revision_matches: false` **before any search**. The next search synchronised lazily (the deleted document was absent, results consistent). `POST /vector/rebuild` re-embedded 6 chunks in 81.7 ms and the index was consistent; a normal restart loaded the index from file and answered the same queries.

`vector_demo.py` additionally showed: deleting a document removed its vector without invoking the model (8→7); a rebuild gave the same ranking with max |Δ similarity| 0.0; recovery from truncated file, bit-flipped payload and missing file all rebuilt correctly (corrupt files quarantined); a valid file was reused without re-embedding; a changed `max_seq_length` triggered a full rebuild and discarded the old vectors.

`extract_demo.py` (3 samples): 17 entities, 21 mentions, 9 relationships, 0 merge candidates. `graph_demo.py`: shortest paths, traversal, persistence/recovery and delete behaviour as recorded in the Phase 4 notes (e.g. deleting `backend_stack.md` took the graph from 12→8 nodes, 8→4 edges).

## 5. Benchmark results and measurement conditions

**Preliminary results from a single run** (no repeats, machine not isolated). Script `scripts/vector_benchmark.py`, post-fix, revision-pinned (`evaluation/reports/phase5_vector_benchmark.json`).
Conditions: Windows 11, AMD64 Family 25 Model 68 (16 logical CPUs), Python 3.12.0, torch 2.15.0+cpu (8 threads), faiss 1.15.1 (16 threads), numpy 2.5.4, CPU only. Corpus: **2,000 synthetic chunks** (mean 479 characters, 302–665), all distinct. The model was already in the local cache. The machine was not otherwise isolated, so treat figures as indicative of this machine only.

| Measurement | Result |
|---|---|
| Model load | 10.38 s; resident memory +316.3 MB |
| Embedding throughput by batch size | 1: 44.8 texts/s · 8: 73.9 · 32: 75.5 · 64: 73.5 · 128: 70.3 |
| Indexing 2,000 chunks | 27.3 s total (27.0 s embedding, 0.31 s everything else); 74.1 texts/s; RSS +143 MB |
| Index file | 3,273,413 bytes (raw vectors 3,072,000) |
| Save / load (median of 5) | 15.4 ms / 14.7 ms |
| FAISS-only search, 2,000 vectors | median 0.30–0.50 ms, p95 0.37–0.69 ms for top_k 1–100 |
| End-to-end search (query embedding + FAISS + SQLite) | median 19.0–21.2 ms, p95 20.2–22.7 ms for top_k 1–50 |
| Query embedding alone (n=100) | median 9.5 ms, p95 12.2 ms, p99 20.6 ms |
| Scaling (synthetic random vectors, FAISS only, top-10) | 10k: 1.10 ms · 50k: 4.74 ms · 100k: 9.23 ms median; serialized 15 / 77 / 154 MB |

These are latency/throughput figures only; they say nothing about retrieval quality. The scaling rows use random vectors, not real embeddings.

## 6. Reliability improvements (before → after)

- **Truncation:** previously over-limit text was silently cut by the model; now measured, pooled, recorded and reported (`oversized_chunks_pooled`, `query_token_count`, `query_windows`).
- **Search during ingestion:** previously a search could block or hit a half-built state; now it serves the last-known-good snapshot flagged `index_stale`, or a clear 503 when no compatible index exists.
- **Publishing:** previously index and metadata could be replaced separately; now one validated, immutable snapshot, with the previous one kept on failure.
- **Model drift:** previously an upstream model update could silently change vectors; now the revision is pinned, verified after load, and part of the fingerprint.
- **Offline:** explicit errors, no silent fallback.
- **Diagnostics:** `/vector/index?check=true` compares SQLite, memory, file and bookkeeping.

## 7. Failed, skipped and unexecuted checks

**Failed (final runs):** none. During development I hit and fixed an over-slow concurrency test (80 s, now about 1 s), two wrong test expectations, and meaningless assertions (`or True`) that I replaced with real checks; these are not present in the final runs. The only failing invocation in the final chain failure of note was my own misuse: `extract_demo.py` run without arguments (exit 2), then re-run correctly.

**Skipped:** 17 tests in the default suite are the opt-in real-model tests (`slow` marker); they were run separately and passed (step 5).

**Not executed / not verifiable here:**
- No quality benchmark (recall@k, MRR, nDCG) — no labelled dataset exists yet.
- Multi-process or multi-instance use of one data directory.
- GPU execution; other operating systems; other embedding models (only the default and the hashing stub were exercised).
- Corpora larger than 2,000 real chunks end-to-end; the 10k–100k scaling figures cover FAISS search only, on random vectors.
- Real-model behaviour for chunks above the 256-token limit was tested in the slow suite, but the live-server demo corpus had none (`oversized_chunks_pooled` 0).
- Coverage was not measured for the opt-in slow tests.
- No network-failure behaviour beyond the offline tests (first-time model download was not re-tested; the model was already cached).

## 8. Git audit (step 10)

The repository has **no commits and nothing is staged**, so `git ls-files` is empty simply because nothing has been added yet; "not tracked" does not by itself prove a file is excluded, which is why ignore rules were checked separately. 142 files (including this report) are untracked-and-not-ignored, i.e. they would be picked up by a future `git add .`, and are all source, tests, migrations, docs, scripts, `.gitkeep` files, `.env.example`, `.gitignore`, config files, and the three benchmark JSON reports. `git check-ignore` confirms these are ignored: `backend/.venv` (1.3 GB), model cache files under `data/models` (88 MB), `data/graphrag.db` and its `-wal`, `backend/.pytest_cache`, `.ruff_cache`, `.coverage`, `.env`. Nothing was found matching model, index, database, venv, cache, `.corrupt` or log patterns in the untracked set. The largest untracked file is 35 KB.

I did not delete the verification/cache directories (`E:\graphrag-verify`, `E:\graphrag-cache`). The live-smoke data directory `E:\graphrag-cache\tmp\live-smoke` is a throwaway under that cache.

## 9. Known limitations

- Exact flat search is linear; no approximate index (9.23 ms median FAISS at 100k random vectors on this machine).
- Single-process design; shared data directories across instances are unsupported.
- `POST /vector/rebuild` is synchronous and CPU-bound; reads continue on the previous snapshot.
- Similarity is retrieval similarity only; there is no calibration to confidence or relevance.
- Sample-corpus retrieval is weak on some multi-concept queries (§4).
- English-centric model; 256-token limit handled by pooling, which is a policy, not a lossless representation.
- Unauthenticated endpoints (as in earlier phases); rebuild endpoint can be triggered by anyone with access.

## 10. Remaining technical debt

- A deprecation warning from `sentence-transformers` (`get_sentence_embedding_dimension` renamed) appears in `embedder.py`; harmless today, easy to fix.
- Starlette `TestClient`/`httpx2` deprecation notice (carried from Phase 1).
- Rebuild is not incremental in the model-change case and not asynchronous.
- `snapshot.py` has the lowest vector-module coverage (79 %): defensive `validate` error branches are not all exercised.
- Benchmark is a preliminary single run; repeat runs and an idle machine would give tighter figures.
- Quality evaluation (Phase 8) and hybrid fusion (Phase 6) are untouched.

## 10b. Scope of the claims in this report

This report makes no claim of production readiness and no claim of retrieval quality. It shows that the implemented behaviour works as designed on the tests, the 5-document sample corpus and a 2,000-chunk synthetic latency benchmark, on one Windows machine.

## 11. Phase 5 completion status

**Complete pending your approval.** All seven reliability items, the documentation tasks and the ten verification steps were executed; the results above are from those executions. Phase 6 has not started, and I will not start it, commit or push until you explicitly approve Phase 5.
