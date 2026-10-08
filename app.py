"""The quotes service.

    GET  /health   -> {"status": "ok", "version": "..."}           (readiness probe)
    POST /predict  {"text": "..."} -> {"author", "confidence", "version"}

Configuration from the environment, the container way:
    MODEL_PATH   the joblib file written by `train.py --export` (default: model.joblib)
    MODEL_URI    optional, e.g. models:/quotes-author@champion - serve straight from the
                 MLflow registry instead of the file (local runs only: MLflow is a dev
                 dependency and is not in the image)
    APP_VERSION  baked in at image build time (v1 / v2), shown in every response
"""

import os
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "model.joblib"))
MODEL_URI = os.environ.get("MODEL_URI")
VERSION = os.environ.get("APP_VERSION", "dev")


def load_model():
    """The registry model if MODEL_URI is set, else the exported file. Returns (model, source)."""
    if not MODEL_URI:
        return joblib.load(MODEL_PATH), str(MODEL_PATH)

    import mlflow
    import mlflow.sklearn

    source = MODEL_URI
    if "@" in MODEL_URI:  # models:/<name>@<alias> - say which version the alias points at
        name, alias = MODEL_URI.removeprefix("models:/").split("@", 1)
        version = mlflow.MlflowClient().get_model_version_by_alias(name, alias).version
        source = f"{MODEL_URI} (version {version})"
    return mlflow.sklearn.load_model(MODEL_URI), source


model, MODEL_SOURCE = load_model()
print(f"serving {MODEL_SOURCE}", flush=True)

app = FastAPI(title="quotes-service", version=VERSION)


class Quote(BaseModel):
    text: str = Field(min_length=1, examples=["Be yourself; everyone else is already taken."])


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": VERSION}


@app.post("/predict")
def predict(quote: Quote) -> dict:
    try:
        probabilities = model.predict_proba([quote.text])[0]
    except Exception as exc:  # a model that cannot score is a 500, not a crash loop
        raise HTTPException(status_code=500, detail=f"model error: {exc}") from exc
    best = probabilities.argmax()
    return {
        "author": str(model.classes_[best]),
        "confidence": round(float(probabilities[best]), 3),
        "version": VERSION,
    }
