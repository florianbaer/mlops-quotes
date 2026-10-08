"""Two tests for the service. Run: uv run train.py --no-mlflow --export model.joblib && uv run pytest"""

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_a_known_author():
    r = client.post("/predict", json={"text": "Imagination is more important than knowledge."})
    assert r.status_code == 200
    body = r.json()
    assert body["author"]
    assert 0.0 < body["confidence"] <= 1.0
