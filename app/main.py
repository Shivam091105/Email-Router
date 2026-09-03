"""
FastAPI application entrypoint: app bootstrapping, DB table creation on
startup, and route registration for /emails, /reviews, /analytics, plus
/health and /.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import analytics, emails, reviews
from app.core.config import settings
from app.core.logging import configure_logging
from app.database.database import Base, check_db_connection, engine
from app.database import models  # noqa: F401 - ensures models are registered before create_all

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting application in '%s' environment", settings.app_env)
    if check_db_connection():
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables verified/created")
    else:
        logger.warning("Database unreachable at startup; tables were not created")
    yield
    # Shutdown (nothing to clean up yet; SQLAlchemy's engine pool handles
    # connection teardown on process exit)
    logger.info("Shutting down application")


app = FastAPI(
    title="AI-Powered Enterprise Email Routing & Triage System",
    description="Classifies incoming organizational emails and routes them to the correct team.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(emails.router)
app.include_router(reviews.router)
app.include_router(analytics.router)


@app.exception_handler(FileNotFoundError)
async def vectorstore_not_found_handler(request: Request, exc: FileNotFoundError) -> JSONResponse:
    """
    The most common cause of this specific exception in this app is
    querying the RAG endpoints before `scripts/build_index.py` has been
    run. Without this handler, that surfaces as an opaque "Internal
    Server Error" — this turns it into a 503 with the actual fix to run,
    which is what a real API consumer (or you, six months from now) needs.
    """
    logger.error("FileNotFoundError on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=503,
        content={
            "detail": (
                "The classification index is not ready. Run "
                "'python -m scripts.build_index' to build it, then retry."
            )
        },
    )


@app.get("/health", tags=["system"])
def health_check() -> dict:
    """
    Reports service liveness AND database connectivity.

    A /health endpoint that only checks "is the process running" is
    misleading in a system where almost every real request needs the
    database. So we check both, and report them separately, so it's
    obvious from the response alone which part is broken if something
    is down.
    """
    db_ok = check_db_connection()
    return {
        "status": "ok" if db_ok else "degraded",
        "app_env": settings.app_env,
        "database": "connected" if db_ok else "unreachable",
    }


@app.get("/", tags=["system"])
def root() -> dict:
    return {"message": "Email Routing & Triage System API. See /docs for API documentation."}
