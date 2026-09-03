"""
Phase 1 smoke tests.

These just confirm the app boots and the /health endpoint responds with
the correct shape. They intentionally don't assert on database
connectivity status, since whether a DB is reachable depends on the
environment running the test (CI vs local vs Docker).
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint_shape():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"status", "app_env", "database"}
    assert body["status"] in {"ok", "degraded"}
    assert body["database"] in {"connected", "unreachable"}
