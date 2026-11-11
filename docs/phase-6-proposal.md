# Phase 6 Proposal — Hybrid Retrieval (Vector + Knowledge Graph)

Status: **PROPOSAL ONLY (revision 2) — nothing has been implemented, no code, test, schema or migration has been changed.** Awaiting your explicit implementation approval.

**Revision 2 changes (from your review):** (1) D3 withdrawn — graph retrieval is now fully independent of FAISS, vectors and the embedding model, with an exact deterministic ranking formula (section 4.4); (2) D2 refined — `outgoing | incoming | both` traversal, stored direction preserved on every evidence item (section 4.3 and 5); (3) D1 refined — token-boundary exact matching of names and aliases inside natural-language queries, explicit `entity_ids` to resolve ambiguity, the "include" ambiguity mode removed (section 4.2); (4) new section 12 verifying the independence/failure requirements against the actual code and mapping each to a test.
Scope guard: retrieval only (no LLM answers, no frontend, no new infrastructure, no paid APIs, CPU-first, SQLite stays the source of truth, FAISS and NetworkX are reused, no commits).

## 0. Summary and the decisions I need from you

Hybrid retrieval = two independent candidate generators that both produce **chunks**, fused by **rank** (weighted Reciprocal Rank Fusion), with all evidence carried through to the response.

- **Vector path:** the existing `search_vectors()` — reused as is.
- **Graph path (new, fully independent of FAISS/embeddings):** link entities in the query to graph nodes (exact, token-boundary matching of names and aliases only) → walk a small, bounded neighbourhood in the existing NetworkX snapshot (`outgoing`, `incoming` or `both`; stored direction always preserved) → collect the *evidence chunks* of the relationships found → rank those chunks with a deterministic, graph-only formula ending in the chunk ID. The graph path imports nothing from `services/embeddings` (enforced by a test) and needs neither the vector index nor the embedding model.
- **Fusion (new, small):** weighted RRF over the two ranked chunk lists; scores are never compared across paths, only ranks.

There is no database migration, no new table and no new dependency.

I deliberately made the graph path **conservative and simple**, and in several places I recommend deviating from the Phase 0 sketch in `docs/architecture.md` §4. Please confirm or overrule each:

| # | Decision | My recommendation | Why (critical view) |
|---|---|---|---|
| D1 | Entity linking | **Exact, token-boundary matching of normalised names and aliases inside the query; no fuzzy matching, no LLM extraction** | Fuzzy linking turns typos into wrong neighbourhoods; the vector path already covers soft matches. Cost: lower graph recall. Detailed rules in 4.2 (word boundaries, possessives, short names, overlaps, ambiguity). |
| D2 | Traversal direction | Request `graph.direction` = **`outgoing` \| `incoming` \| `both`** (default `both`, 1 hop; 2 hops opt-in); stored direction is returned on every evidence item and **never reversed** | The query's intended direction cannot be inferred without parsing or an LLM, so it is an explicit parameter, not a guess. Degree centrality is not used for ranking (hub dominance). |
| D3 | Graph ranking | **Withdrawn the vector tie-break.** Deterministic, graph-only lexicographic formula; the final tie-breaker is the chunk ID (4.4) | Graph retrieval must work with FAISS, vectors and the embedding model absent, and its ranking must not change when the vector index changes. RRF combines the two independent rankings afterwards. |
| D4 | Mention-only chunks | **Off by default** (opt-in `include_mentions`) | Chunks that merely mention an entity are keyword retrieval, which the vector path does better. The unique graph value is relationship evidence. |
| D5 | Endpoints | `POST /search/hybrid` **and** a thin `POST /search/graph` (graph only, same result schema); `/search/vector` is untouched | `/search/graph` was already listed as planned in the API table; it is the graph-only arm needed for later comparison. |
| D6 | Dedup / token budget | Dedup by `chunk_id` only. **Near-duplicate and overlap handling and the token budget are deferred to Phase 7** (context assembly) | Adjacent chunks overlap by up to 100 characters but have different IDs; collapsing them is a context-building concern and would complicate evaluation by chunk ID. |
| D7 | Ambiguous entity names | **Report and skip — never guess.** The only resolution is explicit `graph.entity_ids` (the earlier "include all candidates" mode is removed) | A name shared by two entities of different types must not silently pick one, nor silently expand both. |

## 1. What exists today (inspected, to be reused, not duplicated)

| Need | Existing piece | Fit |
|---|---|---|
| Embed query, search, filter, rounded scores | `services/retrieval/vector_search.py::search_vectors` → `VectorSearchOutcome` | Reuse directly. Already uses one immutable snapshot, handles `index_stale`, completed-documents-only, unknown document IDs, vanished chunks. |
| Stale/building behaviour | `VectorService.ensure_fresh()` (`VectorIndexBuilding`, `VectorIndexUnavailable`) | Hybrid maps these to path status, not to a failure of the whole request. |
| Independence | `services/graph/*` imports nothing from `services/embeddings` or `vector_search`, and the vector package imports nothing from the graph (checked with grep). `GraphService.initialize()` and `VectorService.initialize()` each catch their own failures and never raise at start-up | The two paths can be built, run and fail independently. |
| Graph snapshot | `GraphService.snapshot()` → immutable `GraphSnapshot(graph, revision)`; reads **self-heal synchronously** (rebuild when SQLite moved on) | Graph path can never serve a stale graph; it may block briefly while it rebuilds. |
| Traversal with limits and direction | `graph/queries.py::traverse(seeds, hops, direction, rel_types, max_nodes, max_edges, induced)`, `neighbors`, `shortest_path` | Reuse. Direction (`out`/`in`/`both`) already keeps stored edge direction and labels how an edge was reached. |
| Node/edge data | Nodes: `canonical_name`, `aliases`, `entity_type`, `normalized_key`, `source_document_ids`. Edges (keyed by relationship type): `supporting_evidence[]` (`supporting_text`, `document_id`, `chunk_id`, `provenance`, `confidence`), `source_chunk_ids`, `source_document_ids` | The graph already carries relationship → chunk provenance in memory. |
| Name normalisation | `extraction/normalization.py::normalize_key` (+ org-suffix stripping) | Reuse to build the query-side alias index, so linking uses the same rules as ingestion. |
| Entity mentions | `entity_mentions(entity_id, document_id, chunk_id, surface_form)` indexed by `entity_id` | Only needed for opt-in mention chunks (D4). |
| Chunk/document metadata | `DocumentChunk` (`content`, `page_number`, `section`, `chunk_index`), `Document.filename`, `processing_status` | One final SQLite lookup resolves every candidate. |
| API conventions | `/api/v1`, `AppError(status, code, message)`, sync endpoints, Pydantic request/response, `*Dep` annotations, `app.state.{graph,vectors}` | Followed. |

Not in the graph snapshot: the **chunks that mention** an entity (only counts/documents). That is why mention chunks need SQLite (D4) and relationship evidence does not.

Things the existing code does **not** provide and Phase 6 must add: a query-side entity linker (`find_nodes` matches the query *inside* a name, i.e. the wrong direction), chunk-level graph ranking, fusion, and a combined response model.

## 2. Architecture and data flow

```
POST /api/v1/search/hybrid
  │  validate (limits, document_ids, params)
  ▼
HybridRetriever.search(request)            services/retrieval/hybrid.py   (orchestration only)
  │
  ├─► Vector path ── search_vectors(top_k=vector_k, threshold, document_ids)      [existing]
  │      → ranked VectorHit list (already hydrated from SQLite by search_vectors)
  │      status: ok | stale | building | unavailable | skipped
  │
  ├─► Graph path ── GraphRetriever.search(...)                                     [new]
  │      1. graph.snapshot()                       (self-heals; revision recorded)
  │      2. EntityLinker.link(query)               (alias index cached per revision)
  │      3. traverse(seeds, hops, direction, rel_types, caps)
  │      4. collect evidence chunks (+ optional mentions) → candidate chunks
  │      5. rank with the graph-only integer formula (4.4) → ranked GraphHit list (no vector access)
  │      status: ok | no_entities | ambiguous_only | unavailable | skipped
  │
  ├─► Hydrate graph candidates ── ONE SQLite read for the graph chunk ids not already
  │      hydrated by the vector path: drop vanished chunks / non-completed or filtered
  │      documents, attach content, filename, page, section; count what was dropped;
  │      graph ranks are renumbered contiguously (filtering preserves order)
  │
  └─► Fusion ── rrf_fuse(vector_ranks, graph_ranks, k, weights)                    [new, pure]
         over chunk ids only; then the fused top `limit` is assembled
  ▼
HybridSearchResponse (fused top `limit`, per-path ranks/scores/evidence, path status, warnings)
```

Why this shape:
- **Independent paths, pure fusion.** Each path is testable alone; fusion is a pure function over ranks.
- **SQLite verifies everything last.** Both derived artifacts (FAISS file, graph snapshot) can lag SQLite; the hydration step (which runs before fusion, so fused ranks never include a vanished chunk) is the single place that decides a chunk still exists and is searchable. A chunk that vanished (deleted document, race with ingestion) is dropped and counted, never invented.
- **Sequential execution.** The vector path is ~19 ms end to end (Phase 5 benchmark, preliminary); the graph path is in-memory. A thread pool would add risk for no demonstrated benefit. I will measure before reconsidering.

## 3. New and modified modules

New (all under `backend/app/`):

| Module | Responsibility |
|---|---|
| `services/retrieval/entity_linker.py` | `AliasIndex` (built per graph revision, cached) and `EntityLinker.link(query)`; returns linked / ambiguous mentions. Imports only `graph` and `extraction.normalization` |
| `services/retrieval/graph_search.py` | `GraphRetriever`: link → traverse → evidence collection → ranking; returns `GraphHit` list + `GraphSearchOutcome`. **Must not import `services/embeddings` or `vector_search`** (import-boundary test, like the existing FAISS-boundary test) |
| `services/retrieval/scope.py` | neutral helpers shared by both paths: `resolve_document_scope()` and `lookup_chunks()` (SQLite). Contains no vector or graph logic |
| `services/retrieval/fusion.py` | `rrf_fuse()` — pure, no I/O |
| `services/retrieval/hybrid.py` | `HybridRetriever`: orchestration, partial-failure handling, hydration, response assembly (service-level result keeps the **full** per-path lists for Phase 8) |
| `schemas/hybrid.py` | request/response Pydantic models |

Modified (small, additive, behaviour-preserving):

| Module | Change |
|---|---|
| `services/retrieval/vector_search.py` | replace its inline "allowed documents" block with a call to `scope.resolve_document_scope()` (behaviour-preserving). **No other change** — no new outcome fields, no signature change. |
| `api/search.py` | add `/search/hybrid` and `/search/graph` routes; `/search/vector` unchanged |
| `api/deps.py`, `main.py` | construct one `HybridRetriever` at start-up and expose `get_hybrid` |
| `core/config.py`, `.env.example` | a few `GRAPHRAG_HYBRID_*` settings (section 4.6); `.env.example` test keeps them honest |
| `docs/architecture.md` §4, §6, `docs/implementation-plan.md` | replace the Phase 0 sketch with the implemented design; API table |

Not touched: FAISS engine/snapshot code, graph builder/store/service, extraction, ingestion, migrations.

## 4. Retrieval and ranking algorithms

### 4.1 Vector path
Exactly `search_vectors()` with `top_k = vector_k` (default 20, ≤ `vector_max_top_k`), optional threshold and document filter. Output: ranked chunks with `similarity` (cosine, 6 decimals), source metadata, `index_stale`, revisions. Rank = 1-based position in that list.

### 4.2 Entity linking (graph path, step 2)

Goal: find graph nodes that the query *explicitly names*, anywhere inside a natural-language sentence; do nothing otherwise. Exact matching only — no fuzzy matching, no stemming, no LLM.

1. **Normalisation** (the same functions used at ingestion, so query and corpus agree): NFKC and control-character removal, whitespace collapsed, case-folded, punctuation handled as in `normalize_key`. Matching is therefore case-insensitive and whitespace-insensitive.
2. **Token-boundary matching.** The query is split into tokens, and a name matches only as a **whole sequence of consecutive tokens** (n-grams, up to the longest indexed name, capped at `hybrid_max_name_tokens` = 8). There is no substring, prefix or regex search anywhere, so word boundaries hold by construction: "Acme" is not found in "Acmeville", "Al" not in "Alice", "Rust" not in "trust", "Smith" is not extracted from "Bob Smith" as a separate entity.
3. **Names and aliases.** Index (per graph revision, cached): for every node, `normalize_key(canonical_name, type)` and `normalize_key(alias, type)` for each alias → node IDs. `match_type` is `exact_name` when the matched key is the canonical name's key, else `exact_alias`.
4. **Corporate suffix.** An n-gram is also tried with trailing corporate suffixes removed ("Acme Inc", "Acme Corp" → "acme"), but only against **ORGANIZATION** nodes, mirroring `normalize_key`. Nothing else is rewritten.
5. **Possessive.** A trailing `'s` / `’s` on a query token is stripped ("Alice's manager" → "alice"). This is the only morphological allowance; plurals, stems and hyphen-splitting are not applied.
6. **Short-name guard.** Single-token names shorter than 4 characters ("Go", "AI", "C#") link only when the query has the same capitalisation as the stored name or alias (case-sensitive), so "where do I go" never links an entity called "Go". Longer and multi-token names stay case-insensitive.
7. **Span resolution.**
   - a span fully contained in a longer matched span is dropped (longest wins; dropped spans are listed under `explain`);
   - two spans that partially overlap, neither containing the other, are reported as an ambiguous group (`overlapping_spans`) and **neither is linked**;
   - one span matching more than one node (for example the same name with two entity types) is ambiguous (`multiple_entities`).
8. **Outcomes per span:** exactly one node → **linked**; several → **ambiguous**: reported with each candidate's `entity_id`, name, type and document count, and **not expanded**; none → nothing.
9. **Explicit `graph.entity_ids`.** These IDs are always seeds (validated against the graph; unknown → 422 `entity_not_found`). They also resolve ambiguity: if an ambiguous mention has a candidate listed in `entity_ids`, the mention is resolved to the listed candidate(s) (reported as `resolved_by: "entity_ids"`) and the other candidates stay unused. `graph.link_entities=false` turns automatic linking off entirely (explicit seeds only), which also gives evaluation a way to run the graph path with known seeds.
10. **Not done, deliberately:** fuzzy or edit-distance matching, compact-key matching ("open ai" ~ "openai"), substring matching, entity-type guessing, pronouns/coreference, LLM or NER-based query extraction.

Expected behaviour on representative inputs (these become tests; they are derived from the rules above, not measurements):

| Query | Entities in graph | Expected |
|---|---|---|
| "Who works at Acme Inc?" | Acme (ORG, key "acme") | linked: span "Acme Inc" → Acme (`exact_name`) |
| "who WORKS at  acme?" | same | linked: case/whitespace-insensitive |
| "Tell me about Acmeville" | Acme | nothing: no partial-word match |
| "Where is Alice's manager?" | Alice (PERSON) | linked: "Alice's" → Alice |
| "Did Bob Smith join?" | Bob Smith, and a separate node "Smith" | linked: Bob Smith (longest wins); "Smith" suppressed, listed in `explain` |
| "Where do I go next?" | an entity named "Go" | nothing: short-name case guard |
| "Is Go fast?" | an entity named "Go" | linked: capitalisation matches |
| "Tell me about Mercury" | Mercury (PLANET), Mercury (ELEMENT) | ambiguous (`multiple_entities`), nothing expanded; with `entity_ids=[<planet id>]` → resolved |
| "Acm Inc hires" | Acme | nothing: no fuzzy match |

Honest limit: a query that names an entity differently from the corpus ("the Big Apple" for a node called "New York") gets no graph contribution. By design the result is then "vector only", reported as `graph.status = "no_entities"`.

### 4.3 Traversal and evidence collection (steps 3–4)

**Direction.** `graph.direction` takes `outgoing`, `incoming` or `both` (default `both`). The shorter spellings `out` / `in` used by `/graph/entities/{id}/neighbors` are accepted as aliases and normalised; responses always echo the canonical word. Internally these map onto the existing `traverse(direction="out"|"in"|"both")`.

- Direction is always relative to the node being expanded: `outgoing` follows edges whose **source** is that node (node → other); `incoming` follows edges whose **target** is that node (other → node); `both` follows either. It is applied at every hop (with `both` and more than one hop a walk may go against a stored direction; each edge is labelled).
- **How it is selected:** only by the request. It is never inferred from the wording of the query ("who manages whom" is not parsed), because that needs a parser or an LLM. Default `both` at one hop is the conservative choice when the intended direction is unknown, and `outgoing` / `incoming` let a caller or an evaluation run narrow it.
- **How it is exposed:** the request field above; `paths.graph.direction` echoes the value used; each evidence item carries `traversal` (`outgoing` / `incoming` / `induced`: how the walk reached that edge) and `reached_from` (the entity it was reached from).
- **Stored direction is never altered.** Every evidence item reports `source` and `target` exactly as stored (`source → target`), whatever the traversal direction. Bidirectional traversal never reverses, mirrors or synthesises a relationship. Example, with the stored edge `Alice -WORKS_AT-> Acme Inc` and seed Acme Inc: `outgoing` returns no WORKS_AT edge; `incoming` returns it; `both` returns it; in the last two the evidence reads `source = Alice, target = Acme Inc, traversal = incoming`. Direction labels play no role in ranking.

Other rules:
- `traverse(seeds, hops, direction, rel_types, max_nodes, max_edges)` from `graph/queries.py`. Defaults: `hops=1`; hard cap `hybrid_graph_max_hops` (default 2, never above `graph_max_hops`). Existing node/edge caps apply and `truncated` is passed through. `relationship_types` optionally restricts the walk.
- With ≥ 2 seeds, `induced=True` also surfaces edges *between* the seeds ("how are A and B related?").
- **Edge distance** (used for ranking) is defined purely from the returned nodes: `min(depth(source), depth(target)) + 1`, i.e. one more than the distance of the nearer endpoint to a seed. It does not depend on the walk order.
- **Evidence chunks:** each `supporting_evidence[]` item of each returned edge yields a candidate chunk (`chunk_id`, `supporting_text`, relationship, endpoints, provenance, confidence-if-any). One chunk can be supported by several edges; they merge into one candidate.
- **Mention chunks (D4, opt-in):** `entity_mentions` rows for the *seeds only*, capped per entity (`mentions_per_entity`, default 5). These need SQLite; everything else comes from the in-memory graph.
- Scope: completed documents only and the optional `document_ids` filter via `scope.resolve_document_scope()` (SQLite `documents` table; no vector involvement).

### 4.4 Graph ranking — exact formula and tie resolution (step 5)

The graph list is ranked using **only** graph/evidence data: no embeddings, no FAISS, no cosine similarity, no query vector.

For each candidate chunk `c`, let `E(c)` be the set of traversed edges that have at least one evidence item in `c`. Features, all integers or booleans:

| Key | Definition | Direction |
|---|---|---|
| f1 `seed_coverage` | number of distinct seeds that are an endpoint of some edge in `E(c)` (plus, with `include_mentions`, seeds mentioned in `c`) | higher first |
| f2 `connects_seeds` | 1 if some edge in `E(c)` has **both** endpoints among the seeds | 1 first |
| f3 `relationship_evidence` | 1 if `E(c)` is non-empty, 0 for mention-only chunks | 1 first |
| f4 `distance` | `min` edge distance over `E(c)` (0 for mention-only) | lower first |
| f5 `support` | `|E(c)|`: the number of **distinct supporting relationships** (distinct source–type–target) evidenced in this chunk | higher first |
| f6 `corroboration` | `max` over `E(c)` of the number of distinct documents supporting that edge anywhere in the corpus (`len(edge.source_document_ids)`) | higher first |
| f7 `provenance_known` | 1 if any evidence item of `E(c)` has a provenance other than `"unknown"` | 1 first |
| f8 `chunk_id` | the chunk's ID | ascending (final tie-breaker) |

**Sort key** (a lexicographic tuple, compared left to right, no weighted sum, no floats):

```
( -f1, -f2, -f3, f4, -f5, -f6, -f7, f8 )
```

Graph rank = 1-based position in that order. `chunk_id` is unique, so the key is a **total order: ties cannot survive the last component**. The same graph snapshot, query, seeds and parameters always produce the same list.

**Tie resolution, step by step:** two chunks tied on f1…f7 are ordered by `chunk_id` (a SHA-1-derived string, stable across runs and across re-ingestion of the same file). This final step is arbitrary with respect to relevance. That is stated, not hidden: the response returns every feature value (`graph.features`), so a reader can see which keys decided the order and whether the chunk-ID step was used (`graph.features.decided_by`).

**Why these keys:** f1–f2 favour chunks that relate several things the query names; f3–f4 favour direct relationship evidence close to a seed; f5 follows your "distinct supporting relationship count"; f6 favours a fact stated in several documents; f7 is the only use of provenance (see below).

**What is deliberately not a key:**
- *Confidence:* it is NULL for every row today (rule-based extractor) and the project policy is to ignore uncalibrated model-reported confidence.
- *Evidence quality / provenance beyond known-vs-unknown:* provenance values are extractor names (`rule_based`, `mock`, later `llm`) or `unknown` for legacy rows. There is no measured basis to rank one extractor above another, so I do not encode one. f7 only separates "known extractor" from "unknown", is placed late, and can be dropped if you prefer.
- *Degree centrality:* it would reward hubs.
- *Anything that reads the vector index or embedding model.*

**Illustration** (derived from the sample-corpus demo output; will be asserted in tests, not a measurement). Query "Who works at Acme Inc?", seed Acme Inc, `both`, 1 hop: the Team chunk of `backend_stack.md` (Alice WORKS_AT Acme Inc; Acme Inc LOCATED_IN Berlin) and the `team_notes.txt` chunk (Bob Smith WORKS_AT Acme; Acme LOCATED_IN Berlin) both have f1=1, f2=0, f3=1, f4=1, f5=2, f6=2 (the LOCATED_IN edge is supported by both documents), f7=1 — fully tied until f8, so their order is decided by chunk ID. This is the honest situation for symmetric evidence; in the hybrid list RRF then uses the vector rank for those chunks.

The graph list is truncated to `graph.max_chunks` (default 20, cap `hybrid_graph_max_chunks` = 50) **after** ranking and hydration filtering, so ranks stay contiguous after vanished chunks are dropped (filtering preserves order).

### 4.5 Fusion — weighted Reciprocal Rank Fusion
For each chunk `c` appearing in either list:

```
fused(c) = Σ over paths p containing c  of  weight_p / (rrf_k + rank_p(c))
```
`rrf_k` default 60, `vector_weight` = `graph_weight` = 1.0 (all configurable per request, bounded; weights ≥ 0, at least one > 0).

- **Why RRF first:** cosine similarity and the graph's integer ranking features are on unrelated scales; RRF consumes only the two independent ranks, so nothing incomparable is ever added. It needs no training data, which we do not have. The graph ranking is computed first, on its own, and never sees vector information.
- **Duplicates:** matched by `chunk_id`; the merged result lists `retrieved_by: ["vector","graph"]`, keeps both original ranks, the cosine score and all graph evidence. A chunk found by both naturally outranks one found by a single path.
- **Order of equal fused scores:** compare scores rounded to 12 decimals (removes float noise), then the better (lower) of the path ranks, then vector-present before graph-only, then `chunk_id` ascending. Fusion never looks at content, filenames or similarity values, so it is a pure function of the two rank lists; the order is total, so identical inputs always give identical output.
- **One path empty/failed:** fusion degrades to that path's order; `fused_score` is still reported and the response says which paths contributed. It does not pretend to be hybrid.
- **What the fused score means:** a rank-based retrieval score, **not** a probability, relevance or answer confidence, and not comparable across queries or configurations. Documented in the response.
- **Not claimed:** that hybrid is better than vector-only. That requires a labelled dataset (Phase 8). The defaults (k=60, equal weights) are the textbook starting point, not tuned values.

### 4.6 Settings (new, all with safe defaults, validated)
`GRAPHRAG_HYBRID_RRF_K` (60), `GRAPHRAG_HYBRID_MAX_LIMIT` (50), `GRAPHRAG_HYBRID_GRAPH_MAX_CHUNKS` (50), `GRAPHRAG_HYBRID_GRAPH_MAX_HOPS` (2), `GRAPHRAG_HYBRID_MAX_NAME_TOKENS` (8). Per-request values can only be tighter than these caps.

## 5. API contracts

Conventions followed: `/api/v1` prefix, sync endpoints, `AppError` JSON envelope, 422 for invalid input with a stable `code`, 503 for retrieval backend problems, rounded scores, unknown document IDs reported not rejected.

### 5.1 `POST /api/v1/search/hybrid`

Request (every field except `query` optional):
```json
{
  "query": "Who works at Acme Inc?",
  "limit": 10,
  "document_ids": ["..."],
  "vector": { "top_k": 20, "similarity_threshold": null },
  "graph": {
    "hops": 1,
    "direction": "both",
    "relationship_types": null,
    "max_chunks": 20,
    "include_mentions": false,
    "link_entities": true,
    "entity_ids": null
  },
  "fusion": { "method": "rrf", "rrf_k": 60, "vector_weight": 1.0, "graph_weight": 1.0 },
  "explain": false
}
```
Validation: `query` 1–2000 non-blank chars; `limit` 1–`hybrid_max_limit`; `vector.top_k` ≤ `vector_max_top_k`; `graph.hops` ≤ `hybrid_graph_max_hops`; `graph.direction ∈ {outgoing, incoming, both}` (`out` / `in` accepted as aliases); `relationship_types` ⊆ known vocabulary; `rrf_k` 1–1000; weights 0–10 with at least one positive; `fusion.method` only `"rrf"` today (field exists so a second method can be added without a breaking change). Unknown fields are rejected.

Response `200`:
```json
{
  "query": "...",
  "results": [
    {
      "rank": 1,
      "chunk_id": "...", "document_id": "...", "content": "...",
      "source": { "filename": "...", "page_number": null, "section": "Team", "chunk_index": 1 },
      "retrieved_by": ["vector", "graph"],
      "fused_score": 0.032787,
      "vector": { "rank": 2, "similarity": 0.489922 },
      "graph": {
        "rank": 1,
        "features": { "seed_coverage": 1, "connects_seeds": false, "relationship_evidence": true,
                      "distance": 1, "support": 2, "corroboration": 2, "provenance_known": true,
                      "decided_by": "chunk_id" },
        "evidence": [
          { "kind": "relationship", "relationship_type": "WORKS_AT",
            "source": { "entity_id": "...", "name": "Alice", "type": "PERSON" },
            "target": { "entity_id": "...", "name": "Acme Inc", "type": "ORGANIZATION" },
            "traversal": "incoming", "reached_from": { "entity_id": "...", "name": "Acme Inc" },
            "distance": 1, "edge_id": "...", "relationship_id": "...",
            "supporting_text": "Alice works at Acme Inc", "provenance": "rule_based",
            "confidence": null }
        ]
      },
      "explanation": null
    }
  ],
  "linked_entities": [ { "span": "Acme Inc", "entity_id": "...", "name": "Acme Inc",
                         "type": "ORGANIZATION", "match_type": "exact_name", "resolved_by": "query" } ],
  "ambiguous_entities": [ { "span": "...", "candidates": [ {"entity_id":"...","name":"...","type":"..."} ],
                            "reason": "multiple_entities", "action": "skipped",
                            "resolve_with": "graph.entity_ids" } ],
  "paths": {
    "vector": { "status": "ok", "candidates": 20, "index_stale": false,
                "index_revision": 8, "database_revision": 8, "embedding_model": "...", "embedding_model_revision": "..." },
    "graph":  { "status": "ok", "candidates": 4, "graph_revision": 8, "database_revision": 8,
                "direction": "both", "hops": 1, "truncated": [] }
  },
  "fusion": { "method": "rrf", "rrf_k": 60, "vector_weight": 1.0, "graph_weight": 1.0,
              "score_meaning": "rank-based retrieval score, not a probability or answer confidence" },
  "limit": 10,
  "dropped": { "missing_chunks": 0, "documents_not_ready": 0 },
  "unknown_document_ids": [],
  "warnings": [],
  "timings_ms": { "vector": 18.2, "graph": 1.4, "fusion": 0.05, "hydrate": 0.9, "total": 21.0 }
}
```
- `vector`/`graph` blocks are `null` when that path did not retrieve the chunk.
- `graph.features.decided_by` names the first ranking key (f1…f8 in 4.4) on which this chunk differs from the chunk ranked just above it; `"chunk_id"` means every earlier key tied and the stable chunk-ID rule decided.
- `graph.rank`, `vector.rank` and `vector.similarity` are the paths' original ranks and scores, unchanged by fusion.
- `explanation` (only when `explain=true`) is a **templated, deterministic** structure, not generated text: `{ "summary": "...", "fusion": [{"path":"vector","rank":2,"weight":1.0,"contribution":0.016129}, ...], "linking": [...] }`.
- `graph.status ∈ ok | no_entities | ambiguous_only | unavailable | skipped`; `vector.status ∈ ok | stale | building | unavailable | skipped`. `no_entities` is a normal outcome, not an error.

### 5.2 `POST /api/v1/search/graph`
Same request minus `vector` and `fusion`, same result schema with `retrieved_by: ["graph"]`, `vector: null`, `fused_score: null`, `rank` = graph rank. This endpoint depends only on the graph service and SQLite: it does not use the vector service, the FAISS index or the embedding model, and works while they are down, rebuilding or misconfigured. If the graph path is unavailable → 503 `graph_unavailable` (existing code). Its ranked list is, by construction, identical to the `graph` list produced inside `/search/hybrid` for the same inputs.

### 5.3 Errors
| Case | Response |
|---|---|
| blank/over-long query, bad parameter, weights all 0 | 422 `validation_error` (existing FastAPI path) |
| over a configured limit | 422 `limit_exceeded` (existing code) |
| query too long for the embedder's input rules | 422 `invalid_query` (existing code) |
| unknown `entity_ids` | 422 `entity_not_found` |
| one path fails/unavailable, the other works | **200**, failed path `status: "unavailable"` with a machine-readable `error` code, plus a `warnings[]` entry |
| every requested path fails | 503 `retrieval_unavailable`; `details` lists each path with its status and machine-readable error code, e.g. `{"vector": "vector_index_building", "graph": "graph_unavailable"}` |
| `/search/graph` and the graph fails | 503 `graph_unavailable` |

`/search/vector`, `/graph/*`, `/vector/*` and all existing schemas are not modified.

## 6. Evidence and provenance design

Every result carries, without any extra lookup by the caller:
- `chunk_id`, `document_id`, `source.filename`, `page_number`, `section`, `chunk_index`, original `content`;
- `retrieved_by` (`vector`, `graph`, or both) and **both original ranks**; vector `similarity`; graph `features` (the exact integer values that produced the graph rank, plus `decided_by`);
- graph `evidence[]`: relationship type, both endpoint entities (id, name, type) with stored direction, how the walk reached it (`traversal`, `reached_from`), edge distance, `edge_id` and `relationship_id`, supporting text, extractor provenance, confidence (null unless an extractor genuinely reported one);
- top level: how every query span was linked/skipped, per-path status and the revisions each path read, fusion parameters, what was dropped.

Rules: nothing is fabricated; evidence comes only from `Relationship`/`EntityMention` rows (via the graph snapshot) whose chunk was re-verified in SQLite at hydration time; content always comes from `document_chunks`. Chunk IDs are `sha1(checksum:index)`, so they are stable when the same file is re-ingested — which makes them usable as relevance-label keys later.

## 7. Failure handling

| Situation | Behaviour |
|---|---|
| No entity in the query | `graph.status = "no_entities"`; hybrid returns the vector list (`retrieved_by: ["vector"]`); `/search/graph` returns an empty list with that status. Not an error. |
| Ambiguous name | skipped and reported with its candidates; never expanded; explicit `graph.entity_ids` resolves it |
| Vector path down (index building, unavailable, embedding failure, model missing offline) | hybrid returns the graph results (`vector.status` explains why); `/search/graph` is unaffected; `/search/vector` returns its own existing 503 |
| Graph path down (`GraphUnavailable`) | hybrid returns the vector results (`graph.status = "unavailable"`); `/search/vector` is unaffected |
| Linked entity has no edges | graph contributes nothing for it; reported in `linked_entities` with `edge_count: 0` |
| Edge evidence points to a chunk that no longer exists / its document is deleted or not `completed` | dropped at hydration, counted in `dropped`; if that empties a path, its status stays `ok` with `candidates: 0` and a warning |
| Vector index stale (ingestion in progress) | existing last-known-good behaviour; `paths.vector.status = "stale"` |
| Vector index building / embedding backend down | vector path `building` / `unavailable`; graph path still answers; response `200` + warning |
| Graph stale | cannot happen silently: `GraphService.snapshot()` rebuilds from SQLite first; the revision actually used is reported. Revision *skew between paths* (vector stale, graph current) is visible in `paths.*` |
| Graph rebuild takes long | the request waits (existing behaviour); noted as a latency risk, not hidden |
| Both paths unavailable | 503 `retrieval_unavailable` |
| Unknown `document_ids` | reported in `unknown_document_ids` (same as `/search/vector`) |
| Unexpected exception inside one path | caught at the orchestrator boundary, logged with a stack trace, reported as `unavailable` with a generic code (no internals leaked); the other path's result is still returned |
| Concurrent ingestion/deletion during a request | each path reads one immutable snapshot; hydration is the final authority |

Where a caught failure would hide a bug, a test asserts that the real exception class is mapped (not a bare `except Exception` swallow of programming errors in the pure functions: linking, ranking and fusion have no catch).

## 8. Test strategy

All in the default suite using the existing deterministic `HashingEmbedder` stub and the rule-based extractor — no network, no model download, no real-model dependency. Existing tests are untouched and must keep passing.

**Unit (pure, hand-computed expectations)**
- `fusion`: exact RRF values for hand-built lists (including k, weights); item in both lists sums contributions; item in one list; empty list; equal-score tie order; **permutation invariance** (shuffling input order never changes output); weights 0/∞ validation; float-noise ties (rounding to 12 decimals).
- `entity_linker`: every row of the table in 4.2 as a test; plus names and aliases inside long natural-language sentences, a name at the start/middle/end of the query, repeated mentions, punctuation around names, possessive handling, corporate-suffix variants, longest-match containment, **partial-overlap ambiguity**, same-name-two-types ambiguity, `entity_ids` resolving ambiguity (one candidate pinned, both pinned, unknown ID), `link_entities=false`, unicode and mixed case; **negative tests** proving no partial-word, prefix, substring, fuzzy or stemmed links ("Acm", "acme-ish", "Acmeville", "trust" vs "Rust", "Alicia" vs "Alice").
- graph ranking formula (4.4): each key f1…f7 decides alone while the earlier keys are tied (one test per key, hand-built candidates); combined cases; two fully tied candidates are ordered by `chunk_id`; `decided_by` is correct; the same candidates in any input order give the same list; features are integers/booleans only.
- **Independence:** an import-boundary test fails if `entity_linker.py`, `graph_search.py` or `scope.py` import `app.services.embeddings` or `vector_search` (same technique as the existing FAISS-boundary test); `GraphRetriever`'s constructor takes no vector argument.
- Directed traversal: with a hand-built `MultiDiGraph`, `outgoing` / `incoming` / `both` return the correct edge sets (A→B vs B→A asymmetry, a node with only incoming and only outgoing edges, mutual edges); the evidence's `source` and `target` equal the stored ones in **all three** modes (never swapped, never duplicated as a mirrored edge); `out`/`in` aliases normalise; relationship-type filter; hop cap; truncation flags; edge `distance` computed per the definition in 4.3.

**Integration (real SQLite, real pipeline, FastAPI client)**
- Vector-only behaviour via hybrid when the query has no entity; graph-only via `/search/graph`; hybrid with overlap (duplicate chunk merged once with both ranks/evidence); chunk found only by graph; chunk found only by vector.
- Missing entities; ambiguous entities (reported and skipped; resolved by explicit `entity_ids`); unknown `entity_ids`.
- **Graph independent of vectors:** with the vector service replaced by one whose every method raises, or with the index file deleted and the embedder failing, `/search/graph` still returns 200 and the **same ranked list** as with a healthy vector index (list equality, including features).
- **Vector independent of graph:** with `GraphService.snapshot` raising `GraphUnavailable`, `/search/vector` is unchanged and `/search/hybrid` returns the vector results with `graph.status="unavailable"` and a warning.
- **No cross-contamination:** the surviving path's ranked list and scores are identical to its standalone output when the other path fails (asserted by equality); nothing from the failed path appears in results; a failed path leaves no partial state (alias-index cache published only after a complete build).
- **Both fail:** 503 `retrieval_unavailable` with per-path codes in `details`; `/search/graph` alone → `graph_unavailable`.
- **Traceability invariant** (run over every result of every integration scenario): each result has a `chunk_id` that exists in SQLite, with its document and filename; at least one of `vector` / `graph` is non-null; every graph evidence item has a `chunk_id` equal to the result's, and its `supporting_text` (or the mention's `surface_form`) occurs in the returned `content`.
- Evidence integrity: every graph evidence item's `chunk_id` resolves to a returned chunk whose content contains the `supporting_text`.
- Deleted documents: delete a document, then hybrid/graph/vector never return its chunks or evidence; a **simulated race** (graph snapshot built, then chunk row removed) yields a dropped-chunk count, not a 500.
- Stale indexes (vector index lagging while graph is current, and the reverse): hold the vector sync with the existing `BlockingEmbedder` → `status: stale` and graph still answers; bump the graph revision → next request rebuilds and reports the new revision.
- Partial failures: monkeypatch each path to raise (`VectorIndexBuilding`, `VectorIndexUnavailable`, `EmbeddingError`, `GraphUnavailable`, unexpected `RuntimeError`); assert 200 + warning for one failed path, 503 for both.
- Determinism: same request ×N and across a re-ingest in a different document order → identical result list; stable tie-breaking asserted with engineered equal scores.
- Parameter validation matrix (limits, weights, hops, direction, unknown fields) and error envelope shape.
- Non-regression: `/search/vector`, `/graph/*`, `/vector/*` responses unchanged (existing tests plus a schema-snapshot check).
- Settings: new keys validated and documented (`test_env_example` already enforces `.env.example` ↔ `Settings`).

**Opt-in slow:** one real-model hybrid test on the sample corpus checking that evidence and ranks are present and consistent. No assertion about retrieval *quality*.

**Mutation checks (as in Phase 5):** reverse edge direction in evidence; make `both` swap source/target; drop dedup; fuse by raw scores; remove the post-hydration existence check; ignore weights; make graph ranking read the vector index; allow substring/prefix name matches; guess one candidate for an ambiguous name; drop the `chunk_id` tie-breaker — each must make at least one test fail.

**Gates before I report completion:** Ruff check + format, full suite, coverage (target ≥ the current 96 % overall; new modules ≥ 95 %), migration check (head unchanged, still `0006`), live-server smoke with the real model, demo script, git audit.

## 9. Migration requirements

**None.** No new tables or columns: everything is derived from existing `entities`, `entity_mentions`, `relationships`, `document_chunks`, `documents` and the in-memory graph/FAISS snapshots. Alembic stays at head `0006`. (If Phase 8 later wants to persist retrieval runs, that is a Phase 8 migration.)

## 10. Evaluation readiness (Phase 8 hooks, no claims now)

- The service-level result (`HybridOutcome`) keeps the **complete** vector list, graph list and fused list with per-path ranks and scores, independent of the API `limit`. Phase 8 can compute Precision@K, Recall@K, MRR and Hit@K for vector-only, graph-only and hybrid from one run by cutting each list, at chunk level (and document level by mapping chunks to documents).
- Chunk IDs are stable across re-ingestion of identical files, so labels survive.
- The graph list is independent of the vector list, so the vector-only, graph-only and fused rankings of one query are three separately attributable lists; none is derived from another.
- Every response echoes the full configuration (`rrf_k`, weights, hops, direction, limits, model + revision, index/graph revisions), so a result can be attributed to exact settings.
- Phase 6 will publish **no** "hybrid is better" statement. The Phase 6 report will show example queries and mechanics only, labelled as such.
- A small `scripts/hybrid_demo.py` (like the vector/graph demos) will print the three views for the sample corpus.

## 11. Implementation sequence and milestones

Each milestone ends with its tests green, Ruff clean and a short checkpoint; I will stop for your review after M3 and again at the end.

| # | Milestone | Content | Done when |
|---|---|---|---|
| M0 | Foundations | settings + `.env.example`; new neutral `scope.py` (`resolve_document_scope()`, `lookup_chunks()`); `vector_search.py` switches to `resolve_document_scope()` with no other change | all existing tests unchanged and green |
| M1 | Fusion | `fusion.py` + unit tests incl. permutation/tie tests | hand-computed RRF values pass |
| M2 | Entity linker | alias index per revision + linker + unit tests | exact/ambiguous/negative matrix passes |
| M3 | Graph retrieval | `graph_search.py` (traverse with `outgoing`/`incoming`/`both`, evidence, formula ranking), `/search/graph`, import-boundary test | graph-only tests pass **including with the vector service disabled**; **checkpoint for your review** |
| M4 | Hybrid orchestration | `hybrid.py`, partial-failure paths, hydration/drop accounting, `/search/hybrid` | hybrid integration, deletion, stale, failure tests pass |
| M5 | Explainability | `explain=true`, `warnings`, config echo | explanation tests pass |
| M6 | Hardening | determinism and mutation checks, schema non-regression, latency measurement of the new path | all checks run and reported |
| M7 | Docs and verification | architecture §4/§6, implementation plan, `hybrid_demo.py`, full 5C-style verification, `docs/phase-6-final-report.md` | report written; stop for approval |

## 12. Verification of the independence and failure requirements

Checked against the current code where possible, and mapped to tests that will enforce each one.

| Requirement | What the code shows today | How the design guarantees it | Enforced by |
|---|---|---|---|
| Graph-only retrieval works when the vector index is unavailable | `services/graph/*` has no import of `services/embeddings` or `vector_search` (grep); `GraphService.initialize()` never raises and does not touch vectors | `/search/graph` depends only on `get_graph`, settings and the session factory; `GraphRetriever` takes no vector argument; ranking reads only the graph snapshot (4.4); document scope reads the `documents` table | import-boundary test; "vector service raises everywhere → `/search/graph` 200 and identical list" test; mutation: ranking reads the vector index |
| Vector-only retrieval works when the graph is unavailable | `vector_search.py` and `services/embeddings/*` do not import the graph (grep); `/search/vector` uses `get_vectors` only | `/search/vector` is not modified; in hybrid a `GraphUnavailable` becomes `graph.status="unavailable"` | "`GraphService.snapshot` raises → `/search/vector` unchanged, `/search/hybrid` returns vector results" test |
| One failing path does not corrupt the other | the two services own separate immutable snapshots and separate revision counters (`graph_state`, `vector_state`) | no shared mutable state in the retrieval layer; the orchestrator catches per path and only reads results; the alias index is published only after a complete build | equality test: surviving path output with and without the other failing |
| Both paths failing gives a clear `retrieval_unavailable` | `AppError` already supports `details` (`core/errors.py`) | 503 `retrieval_unavailable` with per-path status and error code in `details` | both-fail integration test |
| Every returned chunk has traceable source evidence | every `Relationship` has a NOT NULL `source_chunk_id` and `source_document_id` with FK cascade; extraction validation already requires the supporting text to occur in the chunk; vector hits come from `document_chunks` rows | every result carries `chunk_id`, document, filename, page/section; graph results carry edge evidence with `relationship_id`; hydration verifies the chunk in SQLite | traceability invariant run over all integration scenarios |
| RRF is deterministic and independently testable | — | `fusion.py` is a pure function of two ranked ID lists and four parameters: no I/O, no clock, no randomness, no imports from other retrieval modules, total tie order ending in `chunk_id` | hand-computed unit tests; permutation-invariance test; mutation "fuse by raw scores" |
| Evaluation metadata preserves original rankings and scores | vector hits already carry rank + rounded cosine similarity | each result keeps `vector.rank`, `vector.similarity`, `graph.rank`, `graph.features`; the service result keeps the full per-path lists; the response echoes all parameters and the revisions read | schema tests asserting these fields; test that fused rank/score reproduce from the two stored ranks |

What I could not verify yet (needs the implementation): actual latency of the graph path, and behaviour under concurrent ingestion for the new code; both are in the M6 plan and will be measured, not assumed.

## 13. Technical risks and trade-offs

| Risk | Impact | Mitigation / honest position |
|---|---|---|
| Graph recall is low (exact linking; extractor is rule-based with an ~80-term gazetteer) | In the current corpus the graph path will often be `no_entities` or small | By design (D1). Real gains depend on the Phase 7 LLM extractor and are unmeasured. |
| Hub entities (or symmetric evidence) produce many candidates tied on f1…f7 | Their relative order is decided by `chunk_id`: stable but arbitrary | `max_chunks` cap, 1-hop default, features exposed, and RRF lets the independent vector rank reorder them in hybrid mode. Graph-only order among such ties is a stated limitation; I will not hide it behind a vector tie-break. |
| Fixed RRF defaults are untuned | May under/over-weight a path | Parameters are configurable and echoed; **no tuning claims** until Phase 8 data exists. |
| The graph formula has no relevance signal beyond graph structure | A chunk's graph rank says "close to the named entities with more distinct relationships", not "answers the question" | Documented; evaluation in Phase 8 decides whether f1–f7 are useful. The keys are integers with documented order, so changing them later is a small, testable edit. |
| Provenance key (f7) has no measured basis | Could mildly misorder | Weakest key, placed last before the chunk ID; can be removed with your agreement. |
| Graph snapshot rebuild blocks a request | Latency spike after writes | Existing behaviour; reported in timings; fine at the project scale, not at millions of rows. |
| Revision skew between paths | One path may reflect newer data | Both revisions reported; hydration prevents returning deleted content. |
| Query linking ignores relation words and direction ("who does Alice manage?") | Both directions returned, direction is shown not inferred | Honest; direction-aware query parsing would need an LLM/parser (out of scope). |
| Overlapping adjacent chunks can both appear | Redundant context | Deferred to Phase 7 (D6); IDs kept distinct for evaluation. |
| Single-process, in-memory alias index | Not shared across instances | Same deployment assumption as Phases 4–5. |
| Scope creep toward answer generation / context packing | — | Explicitly excluded; response is retrieval only. |

**Simpler alternatives I considered and rejected:** (1) *graph expansion of vector hits* (take vector top-k, add chunks linked through their entities) — simpler, but it can never surface evidence the vector path missed, which is the point of a graph path; (2) *score-based fusion* (normalise and add cosine and a graph score) — rejected because the scores are not commensurable; (3) *LLM entity extraction from the query* — out of scope (no LLM in this phase) and a new failure mode.

## 14. What happens next

I will not implement anything until you approve (or amend) this proposal. If you approve, my first action is M0–M1 (settings, shared helper, fusion), and I will pause for your review after M3.
