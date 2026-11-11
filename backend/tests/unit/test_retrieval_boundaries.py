"""Import boundaries that keep the retrieval paths independent of each other.

* the graph path must work with FAISS, vectors and the embedding model absent;
* the vector path must work with NetworkX and the graph absent;
* the shared helpers and the pure modules know neither.
"""

import ast
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[2] / "app"

GRAPH_SIDE = ("networkx", "app.services.graph")
VECTOR_SIDE = ("faiss", "numpy", "sentence_transformers", "app.services.embeddings")
VECTOR_SEARCH = "app.services.retrieval.vector_search"


def imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text("utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found |= {f"{node.module}.{a.name}" for a in node.names}
    return found


def matching(imports: set[str], forbidden: tuple[str, ...]) -> list[str]:
    return sorted(i for i in imports if any(i == f or i.startswith(f + ".") for f in forbidden))


def package_files(relative: str) -> list[Path]:
    return sorted((APP / relative).glob("*.py"))


@pytest.mark.parametrize(
    "relative", ["services/retrieval/scope.py", "services/retrieval/path_status.py"]
)
def test_shared_helpers_know_neither_the_vector_nor_the_graph_side(relative):
    imports = imports_of(APP / relative)
    assert not matching(imports, GRAPH_SIDE + VECTOR_SIDE + (VECTOR_SEARCH,)), imports


@pytest.mark.parametrize(
    "relative",
    [
        "services/retrieval/entity_linker.py",
        "services/retrieval/graph_search.py",
        "schemas/graph_search.py",
    ],
)
def test_the_graph_path_does_not_depend_on_faiss_vectors_or_embeddings(relative):
    """Graph retrieval must work with FAISS, vectors and the embedding model absent."""
    imports = imports_of(APP / relative)
    assert not matching(imports, VECTOR_SIDE + (VECTOR_SEARCH, "app.api.deps")), imports
    assert not [i for i in imports if "vector" in i.lower() and "schemas.vector" not in i], imports


def test_fusion_imports_only_the_standard_library():
    imports = imports_of(APP / "services/retrieval/fusion.py")
    assert not [i for i in imports if i.split(".")[0] in {"app", "numpy", "networkx", "faiss"}]


@pytest.mark.parametrize("path", package_files("services/embeddings"), ids=lambda p: p.name)
def test_the_embedding_and_vector_modules_do_not_depend_on_the_graph(path):
    assert not matching(imports_of(path), GRAPH_SIDE), path.name


def test_vector_search_does_not_depend_on_the_graph():
    imports = imports_of(APP / "services/retrieval/vector_search.py")
    assert not matching(imports, GRAPH_SIDE)


@pytest.mark.parametrize("path", package_files("services/graph"), ids=lambda p: p.name)
def test_the_graph_modules_do_not_depend_on_vectors_or_embeddings(path):
    imports = imports_of(path)
    assert not matching(imports, VECTOR_SIDE + (VECTOR_SEARCH,)), path.name
    assert "faiss" not in imports


def test_the_boundary_check_actually_detects_a_violation(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("import networkx\nfrom app.services.graph import queries\n", "utf-8")
    assert matching(imports_of(bad), GRAPH_SIDE) == [
        "app.services.graph",
        "app.services.graph.queries",
        "networkx",
    ]


# ------------------------------------------------------- the orchestrator is the only bridge

VECTOR_PATH = ("app.services.embeddings", VECTOR_SEARCH)
GRAPH_PATH = ("app.services.graph", "app.services.retrieval.graph_search", "networkx")


@pytest.mark.parametrize("path", package_files("services/retrieval"), ids=lambda p: p.name)
def test_only_the_hybrid_orchestrator_imports_both_paths(path):
    imports = imports_of(path)
    knows_vector, knows_graph = matching(imports, VECTOR_PATH), matching(imports, GRAPH_PATH)
    if knows_vector and knows_graph:
        assert path.name == "hybrid.py", (path.name, knows_vector, knows_graph)


@pytest.mark.parametrize(
    "relative",
    [
        "services/retrieval/vector_search.py",
        "services/retrieval/graph_search.py",
        "services/retrieval/entity_linker.py",
        "services/retrieval/fusion.py",
        "services/retrieval/scope.py",
        "services/retrieval/path_status.py",
    ],
)
def test_no_path_or_helper_depends_on_the_orchestrator(relative):
    assert not matching(imports_of(APP / relative), ("app.services.retrieval.hybrid",))


def test_the_hybrid_orchestrator_uses_the_public_entry_points_of_each_path():
    imports = imports_of(APP / "services/retrieval/hybrid.py")
    assert "app.services.retrieval.vector_search.search_vectors" in imports
    assert "app.services.retrieval.graph_search.GraphRetriever" in imports
    assert "app.services.retrieval.fusion.rrf_fuse" in imports
    assert not matching(imports, ("faiss", "numpy", "networkx"))  # no direct engine access


def test_the_hybrid_schemas_import_no_service_code():
    imports = imports_of(APP / "schemas/hybrid.py")
    assert not matching(imports, ("app.services", "faiss", "networkx"))
