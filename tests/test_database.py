"""
Repository layer tests, run against SQLite instead of Postgres.

The schema uses no Postgres-specific types, so SQLite is a faithful
enough stand-in for testing query logic and relationships — it avoids
requiring a running Postgres instance just to run the test suite.
"""

from app.database import repositories


def test_create_and_get_email(db_session):
    email = repositories.create_email(db_session, sender="a@b.com", subject="Hi", body="Help me")
    assert email.id is not None
    assert email.status == "PENDING"

    fetched = repositories.get_email(db_session, email.id)
    assert fetched.sender == "a@b.com"


def test_list_emails_orders_newest_first(db_session):
    e1 = repositories.create_email(db_session, sender="a@b.com", subject="1", body="x")
    e2 = repositories.create_email(db_session, sender="a@b.com", subject="2", body="x")

    emails = repositories.list_emails(db_session)
    assert emails[0].id == e2.id
    assert emails[1].id == e1.id


def test_update_email_status(db_session):
    email = repositories.create_email(db_session, sender="a@b.com", subject="1", body="x")
    updated = repositories.update_email_status(db_session, email.id, "ROUTED")
    assert updated.status == "ROUTED"


def test_create_routing_result_and_fetch(db_session):
    email = repositories.create_email(db_session, sender="a@b.com", subject="1", body="x")
    result = repositories.create_routing_result(
        db_session,
        email_id=email.id,
        department="IT",
        team="IT Support",
        team_id=101,
        confidence=0.9,
        reasoning="reason",
    )
    fetched = repositories.get_routing_result(db_session, email.id)
    assert fetched.id == result.id
    assert fetched.team_id == 101


def test_review_lifecycle(db_session):
    email = repositories.create_email(db_session, sender="a@b.com", subject="1", body="x")
    review = repositories.create_review(
        db_session, email_id=email.id, predicted_team="IT Support", predicted_team_id=101, predicted_confidence=0.4
    )
    assert review.reviewer_decision == "PENDING"

    pending = repositories.list_pending_reviews(db_session)
    assert len(pending) == 1

    decided = repositories.submit_review_decision(
        db_session, email_id=email.id, final_team="Billing", final_team_id=201, decision="CORRECTED"
    )
    assert decided.final_team_id == 201
    assert decided.reviewer_decision == "CORRECTED"
    assert decided.reviewed_at is not None

    assert repositories.list_pending_reviews(db_session) == []


def test_analytics_summary(db_session):
    e1 = repositories.create_email(db_session, sender="a@b.com", subject="1", body="x")
    repositories.update_email_status(db_session, e1.id, "ROUTED")
    repositories.create_routing_result(
        db_session, email_id=e1.id, department="IT", team="IT Support", team_id=101, confidence=0.9, reasoning="r"
    )

    e2 = repositories.create_email(db_session, sender="a@b.com", subject="2", body="y")
    repositories.update_email_status(db_session, e2.id, "REVIEW_REQUIRED")
    repositories.create_routing_result(
        db_session, email_id=e2.id, department="Finance", team="Billing", team_id=201, confidence=0.5, reasoning="r"
    )

    summary = repositories.get_analytics_summary(db_session)
    assert summary["total_emails"] == 2
    assert summary["auto_routed"] == 1
    assert summary["review_required"] == 1
    assert summary["by_department"] == {"IT": 1, "Finance": 1}
