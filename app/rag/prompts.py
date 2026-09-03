"""
Prompt templates for classification and summarization.

Kept as plain functions (not a templating library) — the prompts are
short enough that f-strings are clearer than adding a dependency.
"""

CLASSIFICATION_INSTRUCTIONS = """You are an email routing assistant for an organization.

You will be given an email and a list of candidate teams retrieved from \
the organization's knowledge base. Your job is to decide which ONE team \
should handle this email.

CRITICAL RULES:
- You MUST choose a team_id that appears in the "Candidate teams" list below. \
Never invent a team_id that is not listed.
- If none of the candidates seem like a strong fit, still pick the closest one \
and reflect your uncertainty with a LOW confidence score.
- confidence must be a number between 0.0 and 1.0 representing how certain you \
are, based only on how well the email matches the candidate team's description \
and examples.
- Respond with ONLY a single JSON object, no other text, no markdown code fences.

Required JSON shape:
{{
  "department": "<department name>",
  "team": "<team name>",
  "team_id": <integer, must be one of the candidate team_ids>,
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<one or two sentences explaining the decision>"
}}

Candidate teams:
{context}

Email:
Subject: {subject}
Body: {body}

JSON response:"""


def build_classification_prompt(subject: str, body: str, context: str) -> str:
    return CLASSIFICATION_INSTRUCTIONS.format(subject=subject, body=body, context=context)


SUMMARY_INSTRUCTIONS = """Summarize the following email in 1-2 concise sentences \
for someone triaging support tickets. Focus on what the sender needs, not \
pleasantries. Respond with only the summary text, no preamble.

Email:
Subject: {subject}
Body: {body}

Summary:"""


def build_summary_prompt(subject: str, body: str) -> str:
    return SUMMARY_INSTRUCTIONS.format(subject=subject, body=body)
