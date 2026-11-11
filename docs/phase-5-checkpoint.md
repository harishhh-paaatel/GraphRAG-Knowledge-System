# Phase 5 checkpoint (end of day, 2026-11-05)

Status: **paused mid-phase, awaiting resume.** Nothing is committed or pushed (the repository has no commits). Phase 6 has not been started. This file records only what was actually run; anything not run is marked **NOT VERIFIED**.

## 1. Where Phase 5 stands

Estimated completion: **~85%** (an estimate, not a measurement). The implementation, its tests, the real-model checks, the demo and the benchmark are done. What remains is verification housekeeping, documentation and the final report (section 4).

### Implemented (files exist and were exercised by tests)
| Area | Files |
|---|---|
| Schema + migration | `backend/app/models/vectors.py` (`chunk_embeddings`, `vector_state`), `backend/migrations/versions/0005_vector_index_state.py` |
| Embeddings | `backend/app/services/embeddings/{embedder,factory,vector_index,vector_store,revision,service}.py` |
| Retrieval | `backend/app/services/retrieval/vector_search.py` |
| API | `backend/app/api/search.py` (`POST /api/v1/search/vector`), `backend/app/api/vector.py` (`GET /api/v1/vector/index`, `POST /api/v1/vector/rebuild`), `backend/app/schemas/vector.py` |
| Wiring (modified) | `pipeline.py`, `ingestion/service.py`, `ingestion_worker.py`, `main.py`, `api/deps.py`, `api/documents.py`, `api/health.py`, `core/config.py`, `models/__init__.py`, `requirements.txt` |
| Tests (new) | `tests/unit/{test_embedder,test_vector_index}.py`, `tests/integration/{test_vector_service,test_vector_search_api}.py`, `tests/slow/test_real_model.py`, `tests/vector_helpers.py`; edited `tests/conftest.py`, `tests/integration/{test_database,test_health_and_errors}.py` |
| Scripts / samples | `scripts/vector_demo.py`, `scripts/vector_benchmark.py`, samples `geography.txt`, `vector_search_notes.md`, `kitchen_notes.txt` |
| Reports | `evaluation/reports/phase5_vector_benchmark_run1.json`, `..._run2.json` |

### Design decisions made (all implemented)
- SQLite `document_chunks` decides what must be indexed; the index file and `chunk_embeddings` are reconciled against it. `vector_state.revision` is bumped in the same transaction as any chunk change (pipeline chunk replacement, document deletion).
- One index file (magic + JSON header with ID table + checksums + FAISS payload) replaced atomically with `os.replace`. Published index objects are never mutated (copy-on-write).
- Embedding runs outside the DB write lock. If SQLite moves during embedding, finished vectors that are still current are kept and the index is recorded at the older revision.
- One chunk that cannot be embedded no longer blocks the others (per-chunk isolation); a backend-wide failure aborts without touching the index or file.
- Embedding configuration is fingerprinted (model, requested revision, max sequence length, normalization); a change discards the old index. Resolved dimension/revision are also checked once the model is loaded.
- Only chunks of `completed` documents are searchable. Scores are labelled retrieval similarity (cosine), not answer confidence.

## 2. Verification actually performed (with when)

| Check | Result | Notes |
|---|---|---|
| Full default suite (stub embedder), run 3x | **426 passed, 10 skipped**, identical each time | The 10 skipped are the opt-in real-model tests. |
| Coverage (same run) | **96% total** (3263 statements, 122 missed). Embedding modules 92-100%: embedder 95, service 95, vector_index 97, vector_store 92, vector_search 97 | |
| Real-model tests (`GRAPHRAG_RUN_SLOW=1`) | **10 passed** | Run before a final lint-only edit to that file. |
| Mutation checks (4) | Each mutation made tests fail as intended: content-change detection removed -> 1 failed; model-change detection removed -> 4 failed; tie-break by chunk ID removed -> 2 failed; revision bump on delete removed -> 1 failed. Sources restored; `cmp` re-run today confirms the three edited files are **byte-identical** to the backups. Restored baseline: **107 passed** (the three vector test files). | |
| Migrations | `alembic upgrade head` reaches **0005**, **no drift** vs ORM, full downgrade 0005 -> 0001 and re-upgrade OK | |
| Ruff | `ruff check . ../scripts`: **All checks passed** | Last clean run was *before* the final test (`test_reads_self_heal_...`) was appended. `ruff format` ran after it; `ruff check` was **not** re-run. |
| Real model | `sentence-transformers/all-MiniLM-L6-v2` loads on CPU: **384 dimensions**, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, normalized | cache under `data/models` (E:) |
| Real semantic search / delete / rebuild / recovery / model-mismatch | `scripts/vector_demo.py` ran with exit code 0 against the real model: queries ranked correctly for 4 of 5 questions; deleting a document removed its vector (8 -> 7); rebuild gave the same ranking (max score difference 8.9e-08); truncated, bit-flipped and missing files were rebuilt; a changed config (`max_seq_length=32`) forced a rebuild | See "unresolved" for the one query that ranked oddly. |
| Benchmark (2 runs, real model, 2000 synthetic chunks, CPU) | Run-to-run agreement 1-3% | see below |

Benchmark highlights (measured, `evaluation/reports/phase5_vector_benchmark_run{1,2}.json`; the corpus is **synthetic**, so these are speed/memory numbers, not relevance):
- Embedding throughput: ~70 texts/s at batch 1, ~118 at batch 8-32, ~105-110 at batch 64-128 (mean chunk 479 chars / 67 words).
- Full `VectorService.sync` of 2000 chunks: ~17.3 s, of which ~17.1-17.2 s is embedding.
- FAISS search (2000 vectors): median 0.18-0.29 ms for top-K 1..100. End-to-end search (query embedding + FAISS + SQLite lookup): median 12.1-13.9 ms, p95 <= 14.8 ms; query embedding alone ~9.1-9.4 ms.
- Index file for 2000 x 384: 3,258,629 bytes (raw vectors 3,072,000). Save ~12.1 ms, load ~9.5 ms.
- Random-vector scaling (latency/memory only): 10k vectors ~0.82-0.91 ms, 50k ~4.0-4.2 ms, 100k ~7.9-8.0 ms (top-10 median); 154 MB serialized at 100k.
- Process memory: ~254 MB after imports, ~569 MB after model load (about +315 MB); index-build RSS delta varied between runs (91-135 MB).

## 3. Not verified / not done
- **NOT VERIFIED:** the full default suite and `ruff check` after the last test was added (expected 427 passed + 10 skipped; only the three vector test files, 107 tests, were re-run).
- **NOT VERIFIED:** `scripts/extract_demo.py` and `scripts/graph_demo.py` after Phase 5 changes (not re-run; their files were reformatted by ruff).
- **NOT DONE:** live-server HTTP verification of the new endpoints with the real model (the HTTP behaviour was tested through the in-process test client with the stub embedder only).
- **NOT DONE:** `docs/architecture.md` and `docs/implementation-plan.md` updates for Phase 5 (required by the task).
- **NOT DONE:** `.env.example` entries for the new `GRAPHRAG_EMBEDDING_*` / `GRAPHRAG_VECTOR_*` settings.
- **NOT DONE:** the final 12-item Phase 5 report.

## 4. FAISS / Sentence Transformers installation status
Installed in the project venv (`E:\GraphRAG Knowledge System\backend\.venv`, `include-system-site-packages = false`); `pip list` run today shows: `faiss-cpu 1.15.1`, `sentence-transformers 6.2.0`, `torch 2.15.0` (CPU build; a CUDA-availability check printed `False` earlier), `transformers 5.22.1`, `numpy 2.5.4`, `huggingface_hub 2.4.0`, `networkx 3.7`. These packages were also exercised for real (FAISS index operations, model load, embeddings) by the test runs, demo and benchmark above. All Phase 5 requirements entries (`numpy`, `faiss-cpu`, `sentence-transformers`) are in `backend/requirements.txt`.

Also observed, **not caused by this session's installs**: the system Python on C: (`C:\Users\K G HARISH PATEL\AppData\Local\Programs\Python\Python312`) already lists `sentence-transformers 6.1.0`, `torch 2.14.1`, `transformers 5.19.0`, `numpy 2.2.6`, `scipy`, `scikit-learn`. Their versions differ from the venv's and every `pip install` this session used the venv interpreter. Install dates were not checked. The project does not use them (the venv does not see system packages).

### The "failed installation" diagnostic, exact details
- Command: a **dry run** (`pip install --dry-run --report ... faiss-cpu sentence-transformers`) followed by a small Python snippet that parsed the report.
- pip itself **succeeded**; it printed `Would install ... faiss-cpu-1.15.1 ... sentence-transformers-6.2.0 ... torch-2.15.0 ... transformers-5.22.1 ...` and found no GPU/CUDA packages.
- The tool reported exit code 1 because of **my report-parsing snippet**, not pip. Exact error: `UnicodeDecodeError: 'charmap' codec can't decode byte 0x8d in position 34996: character maps to <undefined>` raised by `json.load(open(...))` (traceback through `encodings\cp1252.py`). Root cause: the file was opened without `encoding="utf-8"`, so Windows used its default cp1252 codec.
- Fix applied: re-read the same report with `encoding="utf-8"` (worked), then installed for real (succeeded; C: free space unchanged at 4.7 GB at that moment).
- A separate diagnostic attempt today (an import check from PowerShell) failed with a `SyntaxError` caused by **quoting in my one-liner**, not by the environment. That check did not complete; the installation evidence is the `pip list` above plus the successful real runs.
- Not caused by: environment variables, paths, package compatibility or dependency resolution (all resolved cleanly for Python 3.12 / win_amd64).
- The two diagnostic tool calls you interrupted (disk/cache locations and git line counts) were not run at that point; the git counts below were gathered afterwards.

## 5. Background tasks
- All stopped/finished. A final process check today found **no** python/pytest/uvicorn processes and no listener on ports 8000, 8765, 8766, 8767.
- The last background task (mutation checks + restored baseline) completed with exit code 0 on its own. Nothing was killed.
- No concurrent edits happened: the mutation loop ran sequentially in a single command and restored every file it touched before finishing (verified by `cmp`). It was only moved to the background because a message arrived while it was running.

## 6. Git working tree
- The repository has **no commits**; `git status` shows `.env.example`, `.gitignore`, `backend/`, `data/`, `docs/`, `evaluation/`, `scripts/` as untracked. Nothing is staged.
- **The "added lines" figure is the entire hand-written project, not generated files.** Counting untracked, not-ignored files with `git ls-files -o --exclude-standard`: **14,861 lines in 127 files** today, counted before this checkpoint file was created (you saw 14,849; the 12-line difference is consistent with edits made since, not separately verified).
  | Folder | Lines | Files |
  |---|---|---|
  | `backend/app` (source) | 6,199 | 68 |
  | `backend/tests` | 5,943 | 29 |
  | `scripts` | 1,217 | 9 |
  | `evaluation` (2 benchmark JSONs) | 556 | 2 |
  | `docs` | 437 | 2 |
  | `backend/migrations` | 432 | 7 |
  | config/other (`alembic.ini`, `pyproject.toml`, requirements, `.gitignore`, `.env.example`) | 77 | 7 |
  | `data/` | 0 | 4 (`.gitkeep` only) |
- Ignored (checked with `git check-ignore`): `backend/.venv`, `data/graphrag.db`, `.pytest_cache`, `.ruff_cache`, `.coverage`, `.env`. The 88 MB model cache and generated graph/index files live under `data/*/` and are not in the count (consistent with `data/` showing 0 lines); I did not run `check-ignore` on the nested model paths themselves.
- Candidates to decide before any commit (no action taken): the two `evaluation/reports/*.json` benchmark files are real measured output from this machine; `scripts/samples/*` are intentional sample inputs.

## 7. Integrity
- Verified today: `backend/.venv/Scripts/python.exe` exists, `pyvenv.cfg` intact, packages listed above present; E: free 129.66 GB, C: free 4.69 GB, D: 103.4 GB.
- Verified today: the three sources touched by the mutation checks are identical to their backups.
- Not individually re-hashed: every other project file (no process modified them after their last test run; no file was deleted or reverted this session).
- Temporary directories **outside** the repo, created by this work and left in place (not deleted; the earlier delete attempt was blocked by a safety check): `E:\graphrag-verify`, `E:\graphrag-cache` (pip cache, temp, HF scratch). The mutation backups were in the bash `/tmp` and may not survive; they are no longer needed.

## 8. Known errors, risks and unresolved issues
1. **A real-model query ranked oddly (reported, not hidden):** "Which technologies are used to build Python APIs?" put `team_notes.txt` (0.3225, a mixed chunk containing "SQLAlchemy is part of Python" and "Python and FastAPI are popular") ahead of the FastAPI chunk in `backend_stack.md` (0.2591). The other four demo queries ranked as expected. Mixed multi-topic chunks dilute similarity; scores are modest (0.3-0.6).
2. **Rebuild is logically equivalent, not bit-identical:** vectors computed in a different batch composition differ by about 1 float32 ULP (max measured 1.2e-7); the ranking was identical in every measured case. Ties are resolved exactly only for bit-equal scores, so two chunks within float noise of each other could swap order after a rebuild with different batching.
3. `POST /api/v1/vector/rebuild` is synchronous, CPU-bound and **unauthenticated**.
4. Startup runs a synchronous sync by default (`vector_sync_on_startup=True`): model load ~8-10 s plus embedding of anything missing blocks startup. A stale index can also trigger embedding inside a search request (503 `vector_index_unavailable` if that fails).
5. Each sync hashes every chunk's text (O(total text)) and clones the index (O(vectors)); fine at thousands of chunks, not at millions.
6. A vector-indexing failure does not fail the document: the document completes with `metadata.vector_index.status` = `failed`/`partial` and `/health` shows `degraded`. Intentional, but easy to miss.
7. Windows: HF cache cannot use symlinks (Developer Mode off), so cached files are copied; harmless. `os.replace` can fail if another process holds the index file open (reported as a persistence error, index stays in memory).
8. Model RSS is ~315 MB per process; single-process design (two app instances sharing one data dir would repeatedly resync).
9. `.env.example` and the docs do not yet describe the new settings (pending).
10. The one-off `E:\graphrag-verify` / `E:\graphrag-cache` directories need a manual decision.

## 9. Exact next steps for tomorrow (in execution order)
All commands from `E:\GraphRAG Knowledge System\backend` unless noted.
1. Confirm nothing moved: `git status --short` (expect the same seven untracked entries) and no running python processes.
2. Re-verify lint and the full default suite after the last added test:
   `.venv\Scripts\python -m ruff check . ../scripts` then `.venv\Scripts\python -m pytest` (expect **427 passed, 10 skipped**; report the real numbers if different).
3. Re-run the earlier-phase regressions from the repo root: `backend\.venv\Scripts\python scripts\extract_demo.py scripts\samples\backend_stack.md scripts\samples\team_notes.txt` and `backend\.venv\Scripts\python scripts\graph_demo.py`.
4. Live-server check with the real model: set `GRAPHRAG_DATA_DIR` (a throwaway dir on E:), `GRAPHRAG_EMBEDDING_PROVIDER=sentence_transformers`, `GRAPHRAG_EMBEDDING_CACHE_DIR=E:\GraphRAG Knowledge System\data\models`, run `alembic upgrade head`, start uvicorn, upload the sample files, call `POST /api/v1/search/vector`, `GET /api/v1/vector/index?check=true`, delete a document and search again, restart and confirm the index loads from file, then stop the server.
5. Update `docs/architecture.md` (vector section: model, consistency protocol, file format, endpoints) and `docs/implementation-plan.md` (Phase 5 status, results, limitations, reproduction commands); add the new settings to `.env.example`.
6. Optionally re-run the real-model tests: `$env:GRAPHRAG_RUN_SLOW="1"; .venv\Scripts\python -m pytest tests/slow`.
7. Write the final 12-item Phase 5 report and **wait for approval before Phase 6**.

To see the benchmark again (about 2 minutes, run alone): from the repo root `backend\.venv\Scripts\python scripts\vector_benchmark.py --output evaluation\reports\<new-name>.json`.
