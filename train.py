"""Train the quotes model and track it with MLflow.

A TF-IDF + logistic-regression pipeline that guesses the author of a quote
(a toy: 48 quotes, 7 authors - the tracking is the point, not the accuracy).

    uv run train.py                         # two runs, two registered versions, best one -> "champion"
    uv run train.py --export model.joblib   # additionally save the best model for app.py
    uv run train.py --no-mlflow --export /app/model.joblib   # what the Dockerfile runs
    uv run train.py --data /data/quotes.csv                  # train on another CSV

Every run logs its hyperparameters, the git commit, the metrics and the model
itself; every model is registered as a version of "quotes-author", and the alias
"champion" points at the best one. Compare the runs with `uv run mlflow ui`.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from prepare import prepare

DATA = Path(__file__).parent / "data" / "quotes.csv"
EXPERIMENT = "quotes-author"
MODEL_NAME = "quotes-author"
CONFIGS = [{"C": 1.0, "ngram_max": 1}, {"C": 10.0, "ngram_max": 2}]


def load_data(path: Path = DATA):
    # prepare() is idempotent: on the raw scrape (data-v1) it filters to the 7 authors,
    # on the training set (data-v2) it changes nothing - either data version trains.
    df = prepare(pd.read_csv(path))
    return train_test_split(
        df["text"], df["author"], test_size=0.25, random_state=42, stratify=df["author"]
    )


def build_model(C: float, ngram_max: int) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, ngram_max), sublinear_tf=True)),
            ("clf", LogisticRegression(C=C, max_iter=1000)),
        ]
    )


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-git"


def train_and_log(
    C: float, ngram_max: int, use_mlflow: bool, data: Path = DATA
) -> tuple[Pipeline, float, str | None]:
    """Train one configuration; with MLflow, log params, commit, metrics and the model.

    Returns (model, f1_macro, model_uri). model_uri is None without MLflow.
    """
    X_train, X_test, y_train, y_test = load_data(data)
    model = build_model(C, ngram_max)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"C={C:<5} ngram_max={ngram_max}  accuracy={acc:.3f}  f1_macro={f1:.3f}")

    if not use_mlflow:
        return model, f1, None

    import mlflow
    import mlflow.sklearn

    with mlflow.start_run():
        mlflow.log_param("C", C)
        mlflow.log_param("ngram_max", ngram_max)
        mlflow.log_param("git_commit", git_commit())
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)
        # THIS is the line that makes the run reproducible and deployable:
        info = mlflow.sklearn.log_model(model, name="model")
        return model, f1, info.model_uri


def register(model_uri: str) -> str:
    """Register the logged model as a new version of MODEL_NAME; return the version."""
    import mlflow

    result = mlflow.register_model(model_uri, MODEL_NAME)
    return result.version


def promote_champion(version: str) -> None:
    """Move the alias "champion" to `version` - this, not a new file, is a promotion."""
    import mlflow

    mlflow.MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", version)
    print(f"alias 'champion' -> {MODEL_NAME} v{version}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", metavar="PATH", help="save the best model with joblib (for app.py / the image)")
    parser.add_argument("--data", type=Path, default=DATA, help="training CSV (default: data/quotes.csv)")
    parser.add_argument("--no-mlflow", action="store_true", help="train only, no tracking (used in the Docker build)")
    args = parser.parse_args()

    use_mlflow = not args.no_mlflow
    if use_mlflow:
        import mlflow

        mlflow.set_experiment(EXPERIMENT)

    best = None
    for cfg in CONFIGS:
        model, f1, model_uri = train_and_log(cfg["C"], cfg["ngram_max"], use_mlflow, args.data)
        version = register(model_uri) if use_mlflow else None
        if version:
            print(f"  -> registered {MODEL_NAME} v{version}")
        if best is None or f1 > best[1]:
            best = (model, f1, version, cfg)

    model, f1, version, cfg = best
    print(f"best: {cfg} (f1_macro={f1:.3f})")
    if version:
        promote_champion(version)
        print("Compare the runs: uv run mlflow ui  ->  http://localhost:5000")
    if args.export:
        out = Path(args.export)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out)
        print(f"exported {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
