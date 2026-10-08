# The quotes service as a container image.
#
#   builder  python:3.14-slim + uv. The dependency layer is keyed on pyproject.toml
#            and uv.lock only; the model is trained from data/quotes.csv at build
#            time, so every image carries the model that belongs to its commit.
#   final    python:3.14-slim, non-root (uid 1001), only the dependencies, app.py
#            and model.joblib - no uv, no training code, no MLflow.
#
# Build:   docker build -t quotes-api:v1 .
#          docker build -t quotes-api:v2 --build-arg APP_VERSION=v2 .
# Run:     docker run -d --rm --name quotes -p 8000:8000 quotes-api:v1
# Verify:  curl localhost:8000/health

# ---- builder: resolve deps, train the model -------------------------------
FROM python:3.14-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev

COPY train.py prepare.py ./
COPY data/ data/
RUN uv run --no-sync python train.py --no-mlflow --export /app/model.joblib

# The runtime dependencies as a plain directory, not a .venv (a .venv is not
# relocatable: absolute paths and its own interpreter).
RUN uv export --locked --no-dev --no-emit-project -o /tmp/req.txt && \
    uv pip install --system --target /app/deps -r /tmp/req.txt

# ---- final: slim runtime, no uv, no training code, non-root ---------------
FROM python:3.14-slim

ARG APP_VERSION=v1
ENV APP_VERSION=$APP_VERSION \
    PYTHONPATH=/app/deps \
    MODEL_PATH=/app/model.joblib

RUN groupadd --gid 1001 app && useradd --uid 1001 --gid app --no-create-home app

WORKDIR /app
COPY --from=builder --chown=app:app /app/deps /app/deps
COPY --from=builder --chown=app:app /app/model.joblib /app/model.joblib
COPY --chown=app:app app.py ./

USER 1001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
