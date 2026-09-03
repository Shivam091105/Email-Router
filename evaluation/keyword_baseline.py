"""
Approach 1: keyword/rule-based baseline.

No embeddings, no LLM. Scores each team by counting how many of its
description/example words appear in the email text, picks the
highest-scoring team. This is the "dumbest reasonable thing" a team might
have shipped before considering RAG/LLM at all — the point of comparing
against it is to show whether the added complexity of RAG + LLM is
actually earning its keep.
"""

import re
from collections import Counter

STOPWORDS = {
    "the", "a", "an", "and", "or", "is", "are", "to", "of", "in", "for",
    "on", "with", "this", "that", "it", "i", "my", "me", "can", "you",
    "please", "hi", "hello", "help", "team", "handles",
}


def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z']+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def build_team_keyword_index(departments: list[dict]) -> list[dict]:
    """Precomputes a bag-of-words keyword set per team from departments.json."""
    index = []
    for dept in departments:
        for team in dept["teams"]:
            text = team["description"] + " " + " ".join(team.get("examples", []))
            keywords = Counter(_tokenize(text))
            index.append(
                {
                    "department": dept["department"],
                    "team": team["name"],
                    "team_id": team["team_id"],
                    "keywords": keywords,
                }
            )
    return index


def classify_keyword_baseline(email_text: str, team_index: list[dict]) -> dict:
    """
    Returns the best-scoring team as a dict shaped like a (partial)
    classification result: {department, team, team_id, confidence}.
    `confidence` here is just normalized keyword-overlap score, not a
    calibrated probability — same caveat as the LLM's confidence score.
    """
    email_words = Counter(_tokenize(email_text))

    best = None
    best_score = -1.0
    for entry in team_index:
        overlap = sum(min(count, entry["keywords"].get(word, 0)) for word, count in email_words.items())
        if overlap > best_score:
            best_score = overlap
            best = entry

    total_email_words = sum(email_words.values()) or 1
    confidence = min(best_score / total_email_words, 1.0) if best else 0.0

    if best is None:
        return {"department": "UNKNOWN", "team": "UNKNOWN", "team_id": -1, "confidence": 0.0}

    return {
        "department": best["department"],
        "team": best["team"],
        "team_id": best["team_id"],
        "confidence": round(confidence, 4),
    }
