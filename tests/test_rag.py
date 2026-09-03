"""
RAG pipeline tests.

These deliberately use DeterministicFakeEmbeddings instead of the real
Hugging Face Inference API: tests must be fast, free, and runnable with no
network access or API token (e.g. in CI). We're testing our own retrieval
*plumbing* here (does the vector store index/search/persist correctly,
does metadata survive the round trip) — not the semantic quality of a
particular embedding model, which belongs in the Phase 11 evaluation
suite against the real embedding provider.
"""

import json

import pytest
from langchain_core.documents import Document

from app.rag.embeddings import DeterministicFakeEmbeddings
from app.rag.loader import departments_to_documents, load_departments
from app.rag.retriever import format_context, retrieve_teams
from app.rag.vectorstore import build_vectorstore, load_vectorstore


# ---------- loader tests ----------

def test_departments_to_documents_creates_one_doc_per_team():
    departments = [
        {
            "department": "IT",
            "teams": [
                {"team_id": 101, "name": "IT Support", "description": "desc", "examples": ["a", "b"]},
                {"team_id": 102, "name": "DevOps", "description": "desc2", "examples": ["c"]},
            ],
        }
    ]
    docs = departments_to_documents(departments)
    assert len(docs) == 2
    assert docs[0].metadata == {"department": "IT", "team": "IT Support", "team_id": 101}
    assert "IT Support" in docs[0].page_content
    assert "a" in docs[0].page_content


def test_load_departments_rejects_duplicate_team_ids(tmp_path):
    bad_data = [
        {"department": "IT", "teams": [{"team_id": 101, "name": "A", "description": "x", "examples": []}]},
        {"department": "HR", "teams": [{"team_id": 101, "name": "B", "description": "y", "examples": []}]},
    ]
    path = tmp_path / "departments.json"
    path.write_text(json.dumps(bad_data))

    with pytest.raises(ValueError, match="Duplicate team_id"):
        load_departments(path)


def test_load_departments_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_departments(tmp_path / "does_not_exist.json")


# ---------- vector store / retriever tests ----------

@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(
            page_content="Team: IT Support. Handles login problems and password resets.",
            metadata={"department": "IT", "team": "IT Support", "team_id": 101},
        ),
        Document(
            page_content="Team: Billing. Handles refunds and payment failures.",
            metadata={"department": "Finance", "team": "Billing", "team_id": 201},
        ),
        Document(
            page_content="Team: Recruiting. Handles job applications and interview scheduling.",
            metadata={"department": "HR", "team": "Recruiting", "team_id": 302},
        ),
    ]


def test_build_and_retrieve_returns_most_relevant_team(tmp_path, sample_documents):
    embedding = DeterministicFakeEmbeddings()
    vectorstore = build_vectorstore(
        sample_documents, embedding, persist_directory=str(tmp_path / "chroma")
    )

    results = retrieve_teams(vectorstore, "I need a password reset for my login", k=1)

    assert len(results) == 1
    top_doc, _score = results[0]
    assert top_doc.metadata["team_id"] == 101


def test_retrieve_respects_k(tmp_path, sample_documents):
    embedding = DeterministicFakeEmbeddings()
    vectorstore = build_vectorstore(
        sample_documents, embedding, persist_directory=str(tmp_path / "chroma")
    )

    results = retrieve_teams(vectorstore, "job applications", k=2)
    assert len(results) == 2


def test_load_vectorstore_after_build_requires_no_reembedding(tmp_path, sample_documents):
    embedding = DeterministicFakeEmbeddings()
    persist_dir = str(tmp_path / "chroma")
    build_vectorstore(sample_documents, embedding, persist_directory=persist_dir)

    # Re-open as a fresh object, simulating what a second process (the
    # running API) does: load an already-built index, don't rebuild it.
    reopened = load_vectorstore(embedding, persist_directory=persist_dir)
    results = retrieve_teams(reopened, "refund for a failed payment", k=1)

    assert results[0][0].metadata["team_id"] == 201


def test_load_vectorstore_missing_index_raises(tmp_path):
    embedding = DeterministicFakeEmbeddings()
    with pytest.raises(FileNotFoundError):
        load_vectorstore(embedding, persist_directory=str(tmp_path / "nonexistent"))


def test_format_context_includes_team_ids(sample_documents):
    results = [(sample_documents[0], 0.1), (sample_documents[1], 0.3)]
    context = format_context(results)

    assert "[team_id: 101]" in context
    assert "[team_id: 201]" in context
    assert "IT Support" in context
