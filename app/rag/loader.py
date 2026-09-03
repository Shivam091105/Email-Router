"""
Organizational knowledge base loader.

Turns data/departments.json into a list of LangChain `Document` objects —
one document per TEAM (not per department), because the classification
decision we ultimately need is "which team," and retrieval works best when
each document maps 1:1 to the thing we want to retrieve.

Each Document's `page_content` is a natural-language blob combining the
team's description and its example issues — this is what gets embedded
and searched against. Each Document's `metadata` carries the structured
fields (department, team name, team_id) that the LLM will need later to
avoid ever inventing a team_id itself: the ID always comes from here, not
from the model's imagination.
"""

import json
import logging
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

DEFAULT_DEPARTMENTS_PATH = Path(__file__).resolve().parents[2] / "data" / "departments.json"


def load_departments(path: Path = DEFAULT_DEPARTMENTS_PATH) -> list[dict]:
    """Loads and lightly validates the raw departments.json structure."""
    if not path.exists():
        raise FileNotFoundError(f"Organizational knowledge base not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError("departments.json must be a non-empty list of department objects")

    seen_team_ids: set[int] = set()
    for dept in data:
        for team in dept.get("teams", []):
            team_id = team["team_id"]
            if team_id in seen_team_ids:
                raise ValueError(f"Duplicate team_id found in knowledge base: {team_id}")
            seen_team_ids.add(team_id)

    return data


def departments_to_documents(departments: list[dict]) -> list[Document]:
    """
    Converts department/team records into one Document per team.

    The page_content format is deliberately simple and readable — it's what
    gets embedded, so it should read the way a real description of the
    team's job would read, including realistic example phrasings of the
    issues they handle (this is what lets retrieval match a new email's
    wording to the right team even when the exact words differ).
    """
    documents: list[Document] = []

    for dept in departments:
        department_name = dept["department"]
        for team in dept["teams"]:
            examples_block = "\n".join(f"- {ex}" for ex in team.get("examples", []))
            content = (
                f"Department: {department_name}\n"
                f"Team: {team['name']}\n"
                f"Responsibilities: {team['description']}\n"
                f"Example issues this team handles:\n{examples_block}"
            )
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "department": department_name,
                        "team": team["name"],
                        "team_id": team["team_id"],
                    },
                )
            )

    logger.info("Loaded %d team documents from knowledge base", len(documents))
    return documents


def load_team_documents(path: Path = DEFAULT_DEPARTMENTS_PATH) -> list[Document]:
    """Convenience one-shot: load the JSON and convert it to Documents."""
    return departments_to_documents(load_departments(path))
