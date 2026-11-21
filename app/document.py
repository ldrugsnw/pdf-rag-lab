import hashlib

from app.parser import extract_pages
from app.chunker import chunk_pages


def generate_document_id(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()


def prepare_document(
    pdf_bytes: bytes,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[dict]:
    document_id = generate_document_id(pdf_bytes)

    pages = extract_pages(pdf_bytes)
    chunks = chunk_pages(
        pages,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    for chunk in chunks:
        chunk["document_id"] = document_id

    return chunks