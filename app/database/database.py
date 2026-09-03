"""
Database engine and session management.

This module owns exactly three things:
1. `engine`      - the SQLAlchemy Engine, which manages a pool of actual
                    TCP connections to Postgres.
2. `SessionLocal` - a factory that produces new Session objects, each
                    representing one unit-of-work (roughly: one request).
3. `Base`         - the declarative base class that all ORM models (Phase 5)
                    will inherit from, so SQLAlchemy knows what tables exist.

`get_db()` is a FastAPI dependency: each request gets its own Session,
and the session is guaranteed to be closed afterwards (even if the request
raised an exception), via the try/finally + generator pattern FastAPI
expects from a dependency.

Why SQLAlchemy: it gives us the ORM convenience of writing Python classes
instead of raw SQL for CRUD operations, while still allowing raw SQL/Core
queries when needed for anything more complex (e.g. analytics aggregates
in later phases).
"""

import logging

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings

logger = logging.getLogger(__name__)

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,  # detects and recycles dead connections automatically
    future=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base class every ORM model (Email, RoutingResult, Review, ...) inherits from."""

    pass


def get_db():
    """FastAPI dependency that yields a request-scoped DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db_connection() -> bool:
    """
    Runs a trivial query to confirm we can actually reach Postgres.

    Used by the /health endpoint. We don't want /health to just say "the
    FastAPI process is alive" — we want it to say "the app can actually
    do its job," and its job requires the database.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # noqa: BLE001 - we deliberately want to catch anything here
        logger.error("Database connectivity check failed: %s", exc)
        return False
