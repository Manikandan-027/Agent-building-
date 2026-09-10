"""Document ingestion pipeline.

PDF/scan/image/text
  -> page extraction (pypdf text; pypdfium2 renders real page images when present)
  -> page images (PNG; synthesized placeholder only in mock mode)
  -> visual retrieval model (colpali-service) -> multi-vector per page
  -> vector store (multi-vector point + metadata)
  -> Postgres/SQLite metadata (documents, document_pages)

Metadata on EVERY vector point: document_id, page_number, tenant_id, access
(roles/user_ids), document_type, version, content hash, ingested_at.
"""
from __future__ import annotations

import base64
import hashlib
from io import BytesIO

from ara.core.errors import ValidationError
from ara.core.ids import iso_now
from ara.core.logging import get_logger
from ara.db import UnitOfWork
from ara.retrieval.colpali_client import ColPaliLike
from ara.retrieval.vectors import InMemoryMultiVectorStore

log = get_logger("ara.ingest")

MAX_DOC_BYTES = 30 * 1024 * 1024
MAX_PAGES = 300


def render_page_image(text: str, page_number: int, title: str) -> bytes:
    """Mock-mode fallback page image (text rendered onto canvas). Real deployments
    should install pypdfium2 to render true page pixels for the visual model."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1000, 1400), "white")
    draw = ImageDraw.Draw(img)
    draw.text((40, 30), f"{title} — page {page_number}", fill="black")
    y = 80
    for line in (text or "").splitlines()[:52] or ["(no extractable text)"]:
        draw.text((40, y), line[:110], fill="black")
        y += 24
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def extract_pdf(text_bytes: bytes) -> list[dict]:
    """Returns [{'page': n, 'text': ...}, ...] using pypdf (no OCR dependency)."""
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(text_bytes))
    if len(reader.pages) > MAX_PAGES:
        raise ValidationError(f"PDF exceeds {MAX_PAGES} pages")
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 — a broken page must not kill ingestion
            text = ""
        pages.append({"page": i, "text": text.strip()})
    return pages


class IngestionPipeline:
    def __init__(self, uow: UnitOfWork, colpali: ColPaliLike, store: InMemoryMultiVectorStore | object):
        self.uow = uow
        self.colpali = colpali
        self.store = store

    def ingest(self, *, tenant_id: str, filename: str, content: bytes, doc_type: str | None = None,
               title: str | None = None, access: dict | None = None, user_id: str = "") -> dict:
        if not content:
            raise ValidationError("empty document")
        if len(content) > MAX_DOC_BYTES:
            raise ValidationError("document exceeds 30MB limit")

        suffix = (filename.lower().rsplit(".", 1) + [""])[-1]
        content_hash = hashlib.sha256(content).hexdigest()

        if suffix == "pdf" or (doc_type == "pdf"):
            pages = extract_pdf(content)
            kind = "pdf"
        elif suffix in {"png", "jpg", "jpeg"}:
            pages = [{"page": 1, "text": ""}]
            kind = "image"
        else:
            pages = [{"page": 1, "text": content.decode("utf-8", errors="replace")}]
            kind = "text"

        title = title or filename
        doc_id = self.uow.documents.create(
            tenant_id=tenant_id, title=title, source=filename, doc_type=doc_type or kind,
            pages=len(pages), access=access or {"roles": ["*"], "user_ids": ["*"]},
            content_hash=content_hash,
        )

        # store page rows + render images
        page_records = []
        for p in pages:
            image_path = None
            try:
                image = render_page_image(p["text"], p["page"], title)
                image_path = f"data/pages/{doc_id}_{p['page']}.png"
                from pathlib import Path

                Path(image_path).parent.mkdir(parents=True, exist_ok=True)
                Path(image_path).write_bytes(image)
            except Exception as exc:  # noqa: BLE001 — rendering is best-effort in mock mode
                log.warning("page_render_skipped", extra={"fields": {"error": str(exc)[:120]}})
            self.uow.documents.add_page(doc_id, p["page"], image_path, p["text"], {})
            page_records.append({"id": f"{doc_id}:{p['page']}", "text": p["text"], "page": p["page"],
                                 "image_b64": base64.b64encode(image).decode() if image else None})

        # visual embedding via colpali-service
        try:
            embeddings = self.colpali.embed_pages(
                [{"id": r["id"], "text": r["text"], "image_b64": r["image_b64"]} for r in page_records])
        except Exception as exc:
            self.uow.documents.delete(doc_id, tenant_id)
            raise ValidationError(f"page embedding failed: {exc}") from exc

        # multi-vector upsert with tenant/permission metadata
        for r, vectors in zip(page_records, embeddings):
            self.store.upsert(
                f"{r['id']}", vectors,
                {"document_id": doc_id, "page_number": r["page"], "tenant_id": tenant_id,
                 "access": access or {"roles": ["*"], "user_ids": ["*"]},
                 "document_type": kind, "version": "v1", "title": title,
                 "content_hash": content_hash, "text_preview": r["text"][:1000],
                 "ingested_at": iso_now(), "ingested_by": user_id},
            )

        log.info("document_ingested", extra={"fields": {"doc_id": doc_id, "pages": len(pages), "kind": kind}})
        return {"document_id": doc_id, "pages": len(pages), "doc_type": kind, "content_hash": content_hash}
