"""
Streamlit demo UI.

Deliberately simple, per the project requirements: functionality over
polish. This is a thin client — every action is a plain `requests` call
to the FastAPI backend; no business logic lives here. That separation
means the API is fully usable (and testable) without Streamlit at all,
and the UI could be swapped for something else without touching the
backend.
"""

import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Email Routing & Triage", layout="wide")
st.title("AI-Powered Enterprise Email Routing & Triage System")

tab_submit, tab_history, tab_review, tab_analytics = st.tabs(
    ["Submit Email", "Email History", "Human Review", "Analytics"]
)


# ---------------- Submit Email ----------------
with tab_submit:
    st.subheader("Submit a new email")

    with st.form("submit_email_form"):
        sender = st.text_input("Sender", placeholder="jane.doe@customer.com")
        subject = st.text_input("Subject", placeholder="Can't access my account")
        body = st.text_area("Body", height=150, placeholder="Describe the issue...")
        submitted = st.form_submit_button("Process Email")

    if submitted:
        if not sender or not body:
            st.error("Sender and body are required.")
        else:
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/emails",
                    json={"sender": sender, "subject": subject, "body": body},
                    timeout=10,
                )
                if resp.status_code == 201:
                    email = resp.json()
                    st.success(
                        f"Email #{email['id']} submitted (status: {email['status']}). "
                        f"Processing runs in the background — check Email History shortly."
                    )
                else:
                    st.error(f"Submission failed: {resp.status_code} — {resp.text}")
            except requests.RequestException as exc:
                st.error(f"Could not reach the API at {API_BASE_URL}: {exc}")

    st.divider()
    st.subheader("Or load from sample dataset")
    st.caption("Loads example emails from data/sample_emails.json for a quick demo.")
    if st.button("Load 3 sample emails"):
        import json
        from pathlib import Path

        sample_path = Path(__file__).resolve().parents[1] / "data" / "sample_emails.json"
        samples = json.loads(sample_path.read_text())[:3]
        for sample in samples:
            try:
                resp = requests.post(f"{API_BASE_URL}/emails", json=sample, timeout=10)
                if resp.status_code == 201:
                    st.write(f"Submitted: {sample['subject']} (id={resp.json()['id']})")
                else:
                    st.write(f"Failed to submit '{sample['subject']}': {resp.status_code}")
            except requests.RequestException as exc:
                st.error(f"Could not reach the API: {exc}")
                break


# ---------------- Email History ----------------
with tab_history:
    st.subheader("Email history")
    if st.button("Refresh", key="refresh_history"):
        st.rerun()

    try:
        resp = requests.get(f"{API_BASE_URL}/emails", timeout=10)
        resp.raise_for_status()
        emails = resp.json()
    except requests.RequestException as exc:
        st.error(f"Could not reach the API at {API_BASE_URL}: {exc}")
        emails = []

    if not emails:
        st.info("No emails yet. Submit one in the 'Submit Email' tab.")
    else:
        for email in emails:
            with st.expander(f"#{email['id']} — {email['subject']} ({email['status']})"):
                st.write(f"**Sender:** {email['sender']}")
                st.write(f"**Body:** {email['body']}")
                st.write(f"**Status:** {email['status']}")

                detail_resp = requests.get(f"{API_BASE_URL}/emails/{email['id']}", timeout=10)
                if detail_resp.status_code == 200:
                    result = detail_resp.json().get("routing_result")
                    if result:
                        st.write("---")
                        st.write(f"**Department:** {result['department']}")
                        st.write(f"**Team:** {result['team']} (team_id={result['team_id']})")
                        st.write(f"**Confidence:** {result['confidence']:.2f}")
                        st.write(f"**Reasoning:** {result['reasoning']}")
                        st.write(f"**Summary:** {result['summary']}")
                        with st.popover("Retrieved context"):
                            st.text(result["retrieved_context"])


# ---------------- Human Review ----------------
with tab_review:
    st.subheader("Pending human reviews")

    try:
        resp = requests.get(f"{API_BASE_URL}/reviews/pending", timeout=10)
        resp.raise_for_status()
        pending = resp.json()
    except requests.RequestException as exc:
        st.error(f"Could not reach the API at {API_BASE_URL}: {exc}")
        pending = []

    if not pending:
        st.info("No emails currently need human review.")
    else:
        for review in pending:
            email_id = review["email_id"]
            email_resp = requests.get(f"{API_BASE_URL}/emails/{email_id}", timeout=10)
            email = email_resp.json() if email_resp.status_code == 200 else {}

            with st.container(border=True):
                st.write(f"**Email #{email_id}:** {email.get('subject', '')}")
                st.write(email.get("body", ""))
                st.write(
                    f"**AI predicted:** {review['predicted_team']} "
                    f"(confidence: {review['predicted_confidence']:.2f})"
                )

                col1, col2 = st.columns(2)
                with col1:
                    final_team = st.text_input(
                        "Correct team name", value=review["predicted_team"], key=f"team_{email_id}"
                    )
                with col2:
                    final_team_id = st.number_input(
                        "Correct team_id", value=review["predicted_team_id"], key=f"team_id_{email_id}"
                    )

                if st.button("Submit decision", key=f"submit_{email_id}"):
                    decision_resp = requests.post(
                        f"{API_BASE_URL}/reviews/{email_id}",
                        json={"final_team": final_team, "final_team_id": int(final_team_id)},
                        timeout=10,
                    )
                    if decision_resp.status_code == 200:
                        st.success("Decision recorded.")
                        st.rerun()
                    else:
                        st.error(f"Failed: {decision_resp.status_code} — {decision_resp.text}")


# ---------------- Analytics ----------------
with tab_analytics:
    st.subheader("Analytics")
    try:
        resp = requests.get(f"{API_BASE_URL}/analytics", timeout=10)
        resp.raise_for_status()
        analytics = resp.json()
    except requests.RequestException as exc:
        st.error(f"Could not reach the API at {API_BASE_URL}: {exc}")
        analytics = None

    if analytics:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Emails", analytics["total_emails"])
        col2.metric("Auto-Routed", analytics["auto_routed"])
        col3.metric("Review Required", analytics["review_required"])
        col4.metric("Avg. Confidence", f"{analytics['average_confidence']:.2f}")

        st.write(f"**Human reviewed (resolved):** {analytics['human_reviewed']}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**By department**")
            st.bar_chart(analytics["by_department"])
        with col_b:
            st.write("**By team**")
            st.bar_chart(analytics["by_team"])
