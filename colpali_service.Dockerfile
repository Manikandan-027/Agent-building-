# ColPali-family visual retrieval service.
# Default: mock mode (CPU, deterministic multi-vector lexical embeddings).
# For REAL ColQwen2 inference, build with --build-arg REAL=1 on a CUDA base image
# and install requirements-colpali.txt (see docker-compose profile "gpu").
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    COLPALI_MODE=mock

ARG REAL=0

RUN groupadd -r colpali && useradd -r -g colpali -d /app colpali

WORKDIR /app

# For REAL=1 builds on a CUDA host image, also:
#   pip install --no-cache-dir -r requirements-colpali.txt
COPY requirements-colpali.txt ./
COPY colpali_service/ colpali_service/
COPY ara/ ara/
COPY pyproject.toml ./

RUN pip install --no-cache-dir fastapi uvicorn[standard] pydantic pillow httpx pypdf \
 && if [ "$REAL" = "1" ]; then pip install --no-cache-dir -r requirements-colpali.txt; fi

USER colpali

EXPOSE 8100
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8100/health')" || exit 1

CMD ["uvicorn", "colpali_service.main:app", "--host", "0.0.0.0", "--port", "8100"]
