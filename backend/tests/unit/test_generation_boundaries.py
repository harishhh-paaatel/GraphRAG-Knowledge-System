"""Import and logging boundaries of the answer-generation code (Phase 7, M0–M1).

* Generation depends on neither FAISS, NetworkX nor the embedding stack: those belong to
  retrieval, and answer generation consumes retrieval results through its public service.
* ``httpx`` is allowed ONLY inside ``providers/`` (the HTTP client lives there); every
  other generation module is network-free.
* The privacy module is standard-library only.
* No log statement in generation code may include prompt, question, answer or document text.
"""

import ast
import re
import sys
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[2] / "app"
GENERATION_FILES = sorted((APP / "services/generation").rglob("*.py"))
PROVIDER_FILES = sorted((APP / "services/generation/providers").rglob("*.py"))
NON_PROVIDER_GENERATION_FILES = [f for f in GENERATION_FILES if f not in PROVIDER_FILES]


def imports_of(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text("utf-8"))):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found |= {f"{node.module}.{a.name}" for a in node.names}
    return found


def hits(imports: set[str], forbidden: tuple[str, ...]) -> list[str]:
    return sorted(i for i in imports if any(i == f or i.startswith(f + ".") for f in forbidden))


@pytest.mark.parametrize("path", GENERATION_FILES, ids=lambda p: str(p.relative_to(APP)))
def test_generation_code_does_not_reach_into_retrieval_engines(path):
    forbidden = (
        "faiss",
        "numpy",
        "networkx",
        "sentence_transformers",
        "torch",
        "app.services.embeddings",
        "app.services.graph",
    )
    assert not hits(imports_of(path), forbidden)


@pytest.mark.parametrize(
    "path", NON_PROVIDER_GENERATION_FILES, ids=lambda p: str(p.relative_to(APP))
)
def test_non_provider_modules_do_not_import_an_http_client(path):
    """httpx (and other HTTP clients) are allowed only inside providers/."""
    assert not hits(
        imports_of(path), ("httpx", "requests", "urllib3", "aiohttp", "openai", "ollama")
    )


def test_the_privacy_module_uses_only_the_standard_library():
    stdlib = set(sys.stdlib_module_names)
    for name in imports_of(APP / "core/privacy.py"):
        assert name.split(".")[0] in stdlib, name


def test_the_query_schemas_import_only_schemas_pydantic_and_the_standard_library():
    stdlib = set(sys.stdlib_module_names)
    for name in imports_of(APP / "schemas/query.py"):
        root = name.split(".")[0]
        assert root in stdlib or root == "pydantic" or name.startswith("app.schemas"), name


LOG_CALL = re.compile(r"logger\.(debug|info|warning|error|exception|critical)\((?P<args>[^\n]*)")
SENSITIVE = re.compile(r"\b(prompt|question|answer|messages?|content|text|excerpt|key)\b", re.I)


@pytest.mark.parametrize("path", GENERATION_FILES, ids=lambda p: str(p.relative_to(APP)))
def test_log_statements_never_include_prompt_question_answer_or_document_text(path):
    source = path.read_text("utf-8")
    for match in LOG_CALL.finditer(source):
        args = match.group("args")
        # a log call may MENTION a count ("%d chars") but must not interpolate sensitive values
        interpolated = re.findall(r"(?:%[sdr]|\{[^}]*\})", args)
        if interpolated:
            assert not SENSITIVE.search(args.replace("chars", "")), (path.name, args)


def test_the_boundary_checks_detect_a_violation(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("import faiss\nimport httpx\nfrom app.services.graph import queries\n", "utf-8")
    assert hits(imports_of(bad), ("faiss",)) == ["faiss"]
    assert hits(imports_of(bad), ("httpx",)) == ["httpx"]
    assert hits(imports_of(bad), ("app.services.graph",)) == [
        "app.services.graph",
        "app.services.graph.queries",
    ]
    leak = 'logger.info(f"prompt was {prompt}")'
    assert LOG_CALL.search(leak) and SENSITIVE.search(LOG_CALL.search(leak).group("args"))
