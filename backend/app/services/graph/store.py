"""Graph file persistence.

Format (documented, version 1): one JSON document

    {"format": "graphrag-knowledge-graph", "format_version": 1,
     "revision": <SQLite graph_state.revision this graph reflects>,
     "built_at": <ISO-8601 UTC>, "node_count": N, "edge_count": M,
     "sha256": <digest of the canonical JSON of "graph">,
     "graph": <networkx node-link data, multigraph, directed>}

JSON node-link rather than GraphML because edges carry nested evidence lists that GraphML
can only hold as opaque strings. The checksum and counts make truncation and bit-rot
detectable. Writes go to a temp file in the same directory and are swapped in with
``os.replace`` (atomic on the same volume), so an interrupted write can never damage the
previous file.
"""

import hashlib
import json
import logging
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

import networkx as nx

logger = logging.getLogger(__name__)

FORMAT = "graphrag-knowledge-graph"
FORMAT_VERSION = 1


class GraphLoadError(Exception):
    """The graph file is missing, unreadable, corrupt or not in the expected format."""


class GraphPersistenceError(Exception):
    """The graph could not be written; the previous file (if any) is untouched."""


def _canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def checksum(payload: dict) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def save_graph(graph: nx.MultiDiGraph, revision: int, path: Path) -> dict:
    """Atomically write ``graph``; returns the envelope metadata (without the payload)."""
    payload = nx.node_link_data(graph, edges="edges")
    envelope = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "revision": revision,
        "built_at": datetime.now(UTC).isoformat(),
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "sha256": checksum(payload),
        "graph": payload,
    }
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(envelope, sort_keys=True, ensure_ascii=False).encode()
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except (OSError, TypeError, ValueError) as exc:
        tmp.unlink(missing_ok=True)
        raise GraphPersistenceError(f"could not write graph file: {type(exc).__name__}") from exc
    return {k: v for k, v in envelope.items() if k != "graph"}


def load_graph(path: Path) -> tuple[nx.MultiDiGraph, int, dict]:
    """Load and fully validate a graph file; raise GraphLoadError on any problem."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise GraphLoadError("graph file does not exist") from None
    except OSError as exc:
        raise GraphLoadError(f"graph file unreadable: {type(exc).__name__}") from exc
    try:
        envelope = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GraphLoadError("graph file is not valid JSON (corrupt or truncated)") from exc

    if not isinstance(envelope, dict) or envelope.get("format") != FORMAT:
        raise GraphLoadError("not a knowledge graph file")
    if envelope.get("format_version") != FORMAT_VERSION:
        raise GraphLoadError(f"unsupported graph format version {envelope.get('format_version')!r}")
    payload = envelope.get("graph")
    revision = envelope.get("revision")
    if not isinstance(payload, dict) or not isinstance(revision, int):
        raise GraphLoadError("graph file is missing required fields")
    if checksum(payload) != envelope.get("sha256"):
        raise GraphLoadError("graph checksum mismatch (file was modified or damaged)")
    try:
        graph = nx.node_link_graph(payload, edges="edges")
    except Exception as exc:  # malformed node-link structure
        raise GraphLoadError("graph structure is invalid") from exc
    if not (graph.is_directed() and graph.is_multigraph()):
        raise GraphLoadError("graph must be a directed multigraph")
    if graph.number_of_nodes() != envelope.get(
        "node_count"
    ) or graph.number_of_edges() != envelope.get("edge_count"):
        raise GraphLoadError("graph element counts do not match the file header")
    meta = {k: v for k, v in envelope.items() if k != "graph"}
    return nx.freeze(graph), revision, meta
