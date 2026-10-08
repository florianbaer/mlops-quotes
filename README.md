# mlops-quotes

A small ML service that guesses the author of a quote - the course project of the MLOps
lecture, finished: the data is versioned, the model is tracked, the service ships as a
container image, and a pipeline tests and publishes it on every push.

```
data/quotes.csv          100 quotes (text, author, tags) from quotes.toscrape.com
prepare.py               turns the raw scrape into the training set (48 quotes, 7 authors)
train.py                 TF-IDF + logistic regression, tracked with MLflow
app.py                   FastAPI: GET /health, POST /predict
Dockerfile               multi-stage, trains the model at build time
.github/workflows/ci.yml test, then build and push the image to ghcr.io
```

## Use the image

The pipeline publishes a public multi-platform image (`amd64` and `arm64`); no login needed:

```bash
docker run -d --rm --name quotes -p 8000:8000 ghcr.io/florianbaer/mlops-quotes:v1
curl localhost:8000/health
# {"status":"ok","version":"v1"}
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' \
     -d '{"text": "Imagination is more important than knowledge."}'
docker stop quotes
```

`:v1` and `:v2` are built from the same commit and differ only in `APP_VERSION`, which every
response reports - two tags to practise a rolling update and a rollback with.

## Run it locally

Needs [`uv`](https://docs.astral.sh/uv/).

```bash
uv run train.py --export model.joblib   # two tracked runs, the better one is exported
uv run uvicorn app:app --port 8000      # serving model.joblib
uv run pytest -q
uv run mlflow ui                        # compare the runs at http://localhost:5000
```

To serve straight from the MLflow registry instead of the exported file:

```bash
MODEL_URI=models:/quotes-author@champion uv run uvicorn app:app --port 8000
```

## Build it yourself

```bash
docker build -t quotes-api:v1 .
docker build -t quotes-api:v2 --build-arg APP_VERSION=v2 .
```

## The pipeline

`.github/workflows/ci.yml` has two jobs:

- **test** - on every push and pull request: `uv sync --locked`, train a model, run the tests.
- **publish** - on `main`, after the tests pass: build for `amd64` and `arm64` and push
  `ghcr.io/florianbaer/mlops-quotes:v1` and `:v2`.

The push needs no secret of your own: the workflow logs in to `ghcr.io` with the
`GITHUB_TOKEN` and `permissions: packages: write`.
