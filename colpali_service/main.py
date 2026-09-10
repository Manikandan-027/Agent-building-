"""ColPali-family visual retrieval microservice.

Isolated from the backend so it can move to GPU infrastructure independently.

Modes (COLPALI_MODE):
  mock : deterministic multi-vector lexical embeddings (CPU, CI/demo, no deps).
  real : ColQwen2/ColPali via `colpali-engine` + torch. Requires the
         `requirements-colpali.txt` extras and a GPU host. Fails CLOSED (503)
         if the model stack is unavailable — never silently falls back.

API:
  GET  /health         -> {status, mode, model}
  POST /embed/queries  -> {queries: [str]}            -> multi-vectors
  POST /embed/pages    -> {pages: [{id, image_b64, text}]} -> multi-vectors per page
"""
from __future__ import annotations

import base64
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="colpali-service", version="0.1.0")

MODE = os.environ.get("COLPALI_MODE", "mock")
_model: Any = None
_processor: Any = None


def _load_real_model() -> None:  # pragma: no cover - requires GPU stack
    global _model, _processor
    from colpali_engine.models import ColQwen2, ColQwen2Processor
    import torch

    name = os.environ.get("COLPALI_MODEL", "vidore/colqwen2-v1.0")
    device = os.environ.get("COLPALI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    _model = ColQwen2.from_pretrained(name, torch_dtype=torch.bfloat16, device_map=device).eval()
    _processor = ColQwen2Processor.from_pretrained(name)


@app.on_event("startup")
def startup() -> None:
    if MODE == "real":
        try:
            _load_real_model()
        except Exception as exc:  # noqa: BLE001 — fail closed, loudly
            raise RuntimeError(f"COLPALI_MODE=real but model stack unavailable: {exc}") from exc


class QueriesRequest(BaseModel):
    queries: list[str]


class PageIn(BaseModel):
    id: str
    image_b64: str | None = None
    text: str = ""


class PagesRequest(BaseModel):
    pages: list[PageIn]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "mode": MODE,
            "model": os.environ.get("COLPALI_MODEL", "vidore/colqwen2-v1.0") if MODE == "real" else "mock-hashing-multivec"}


def _embed_queries_real(queries: list[str]) -> list[list[list[float]]]:  # pragma: no cover
    import torch

    with torch.no_grad():
        batches = _processor.process_queries(queries).to(_model.device)
        emb = _model(**batches).to(torch.float32)
    return [[row.tolist() for row in q] for q in emb]


def _embed_pages_real(pages: list[PageIn]) -> list[list[list[float]]]:  # pragma: no cover
    import torch
    from io import BytesIO

    from PIL import Image

    images = [Image.open(BytesIO(base64.b64decode(p.image_b64))).convert("RGB") for p in pages]
    with torch.no_grad():
        batches = _processor.process_images(images).to(_model.device)
        emb = _model(**batches).to(torch.float32)
    return [[row.tolist() for row in img] for img in emb]


@app.post("/embed/queries")
def embed_queries(req: QueriesRequest) -> dict:
    if not req.queries or len(req.queries) > 32:
        raise HTTPException(422, "queries must contain 1..32 items")
    if MODE == "real":
        if _model is None:
            raise HTTPException(503, "model not loaded")
        vectors = _embed_queries_real(req.queries)
    else:
        from ara.retrieval import mockcolpali

        vectors = [mockcolpali.embed_query(q) for q in req.queries]
    return {"mode": MODE, "embeddings": vectors}


@app.post("/embed/pages")
def embed_pages(req: PagesRequest) -> dict:
    if not req.pages or len(req.pages) > 64:
        raise HTTPException(422, "pages must contain 1..64 items")
    if MODE == "real":
        if _model is None:
            raise HTTPException(503, "model not loaded")
        if any(p.image_b64 is None for p in req.pages):
            raise HTTPException(422, "real mode requires page images")
        vectors = _embed_pages_real(req.pages)
    else:
        # mock mode embeds page TEXT (CI/demo); images accepted but not pixel-understood
        from ara.retrieval import mockcolpali

        vectors = [mockcolpali.embed_page_chunks(p.text or p.id) for p in req.pages]
    return {"mode": MODE, "embeddings": vectors}
