"""Deterministic mock ColPali embedding — the same logic runs in-process (tests/CI)
and inside the colpali-service (mock mode), so retrieval quality is consistent.

It is a multi-vector LEXICAL embedding (hashing bag-of-n-grams, one vector per
page chunk / query token, MaxSim scoring). It intentionally preserves the exact
API shape of real ColPali/ColQwen embeddings (multi-vector per page, MaxSim) so
swapping in the real model changes NO downstream code.

HONEST LIMITATION: mock mode does not understand pixels. Layout/table/figure
understanding requires COLPALI_MODE=real (GPU) — see colpali_service/real.py.
"""
from __future__ import annotations

import hashlib
import re

DIM = 96
CHUNK_TOKENS = 20

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = set("a an and are as at be by for from has have in is it its of on or that the to was were will with".split())


def _norm_tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 1]


def _hash_vec(token: str) -> list[float]:
    """Stable signed hashing-trick vector for one token."""
    v = [0.0] * DIM
    for gram in {token, token[:5]}:
        h = int.from_bytes(hashlib.sha1(gram.encode()).digest()[:8], "big")
        idx = h % DIM
        sign = 1.0 if (h >> 63) & 1 else -1.0
        v[idx] += sign
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


def embed_page_chunks(page_text: str) -> list[list[float]]:
    """Page -> multi-vector: one vector per ~CHUNK_TOKENS-token chunk (simulates
    ColPali's ~one-vector-per-image-patch layout)."""
    toks = _norm_tokens(page_text)
    if not toks:
        toks = ["empty"]
    chunks = [toks[i : i + CHUNK_TOKENS] for i in range(0, len(toks), CHUNK_TOKENS)]
    vectors = []
    for chunk in chunks:
        vec = [0.0] * DIM
        for tok in chunk:
            tv = _hash_vec(tok)
            vec = [a + b for a, b in zip(vec, tv)]
        n = sum(x * x for x in vec) ** 0.5 or 1.0
        vectors.append([x / n for x in vec])
    return vectors


def embed_query(text: str) -> list[list[float]]:
    """Query -> multi-vector: full-query vector + one vector per content token."""
    toks = _norm_tokens(text)
    if not toks:
        toks = ["empty"]
    full = [0.0] * DIM
    for tok in toks:
        full = [a + b for a, b in zip(full, _hash_vec(tok))]
    n = sum(x * x for x in full) ** 0.5 or 1.0
    return [[x / n for x in full]] + [_hash_vec(t) for t in toks]


def maxsim(query_vectors: list[list[float]], page_vectors: list[list[float]]) -> float:
    """ColPali scoring: every query vector's best-matching page vector, averaged."""
    if not query_vectors or not page_vectors:
        return 0.0
    total = 0.0
    for qv in query_vectors:
        best = max(sum(a * b for a, b in zip(qv, pv)) for pv in page_vectors)
        total += best
    return total / len(query_vectors)
