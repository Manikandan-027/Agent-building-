# ARA backend + worker image (CPU). Model inference is delegated to the isolated
# colpali-service container so GPU workloads scale independently.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ARA_ENV=production

RUN groupadd -r ara && useradd -r -g ara -d /app ara

WORKDIR /app

COPY pyproject.toml ./
COPY ara/ ara/
COPY colpali_service/ colpali_service/
COPY scripts/ scripts/
COPY evals/ evals/
COPY tests/ tests/
COPY Makefile ./

RUN pip install --no-cache-dir -e ".[dev]" \
 && apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

USER ara

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "ara.service.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
