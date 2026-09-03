"""
Shared fixtures for API and database tests.

`client` gives tests a fully wired FastAPI TestClient backed by an
isolated in-memory SQLite database (StaticPool keeps the same in-memory
DB alive across the multiple connections SQLAlchemy opens — a plain
`sqlite:///:memory:` engine would otherwise hand out a fresh empty
database per connection). Real Postgres is not required to run these
tests.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import models  # noqa: F401 - registers models on Base
from app.database.database import Base, get_db
from app.main import app


@pytest.fixture
def test_session_factory():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


@pytest.fixture
def db_session(test_session_factory):
    db = test_session_factory()
    yield db
    db.close()


@pytest.fixture
def client(test_session_factory):
    def override_get_db():
        db = test_session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    # Background tasks (app.services.email_service.process_email) use the
    # module-level SessionLocal directly, not the get_db dependency, since
    # they run outside the request's DI scope. Patch it at the source so
    # background-task DB writes land in the same test database.
    with patch("app.services.email_service.SessionLocal", test_session_factory):
        with TestClient(app) as c:
            yield c

    app.dependency_overrides.clear()
