"""
Assembles the LangGraph workflow described in the project README:

START -> preprocess_email -> retrieve_context -> classify_email
      -> [confidence_check] -> route_email        -> generate_summary -> save_result -> END
                             -> human_review       -> generate_summary -> save_result -> END
                             -> (failed)           -> save_result -> END

`decide_after_classification` is the conditional edge function: it reads
`state["status"]`/`state["confidence"]` and returns which branch to take,
without itself being a node (it doesn't return state updates).
"""

from langchain_chroma import Chroma
from langgraph.graph import END, StateGraph

from app.graph.nodes import (
    decide_after_classification,
    make_classify_email,
    make_generate_summary,
    make_human_review,
    make_preprocess_email,
    make_retrieve_context,
    make_route_email,
    make_save_result,
)
from app.graph.state import EmailState
from app.llm.client import LLMClient


def build_workflow(llm_client: LLMClient, vectorstore: Chroma, session_factory):
    graph = StateGraph(EmailState)

    graph.add_node("preprocess_email", make_preprocess_email())
    graph.add_node("retrieve_context", make_retrieve_context(vectorstore))
    graph.add_node("classify_email", make_classify_email(llm_client))
    graph.add_node("route_email", make_route_email())
    graph.add_node("human_review", make_human_review())
    graph.add_node("generate_summary", make_generate_summary(llm_client))
    graph.add_node("save_result", make_save_result(session_factory))

    graph.set_entry_point("preprocess_email")

    # If preprocessing itself failed (empty email), skip straight to saving
    # the FAILED status rather than calling retrieval/classification on
    # nothing.
    graph.add_conditional_edges(
        "preprocess_email",
        lambda state: "failed" if state.get("status") == "FAILED" else "ok",
        {"failed": "save_result", "ok": "retrieve_context"},
    )

    graph.add_conditional_edges(
        "retrieve_context",
        lambda state: "failed" if state.get("status") == "FAILED" else "ok",
        {"failed": "save_result", "ok": "classify_email"},
    )

    graph.add_conditional_edges(
        "classify_email",
        decide_after_classification,
        {
            "failed": "save_result",
            "high_confidence": "route_email",
            "low_confidence": "human_review",
        },
    )

    graph.add_edge("route_email", "generate_summary")
    graph.add_edge("human_review", "generate_summary")
    graph.add_edge("generate_summary", "save_result")
    graph.add_edge("save_result", END)

    return graph.compile()


def run_email_workflow(
    llm_client: LLMClient,
    vectorstore: Chroma,
    session_factory,
    email_id: int,
    sender: str,
    subject: str,
    body: str,
    attachment_text: str = "",
) -> EmailState:
    """Convenience wrapper: build the graph fresh and run it for one email."""
    workflow = build_workflow(llm_client, vectorstore, session_factory)
    initial_state: EmailState = {
        "email_id": email_id,
        "sender": sender,
        "subject": subject,
        "body": body,
        "attachment_text": attachment_text,
    }
    return workflow.invoke(initial_state)
