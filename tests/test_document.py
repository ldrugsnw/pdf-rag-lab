from app.document import generate_document_id, prepare_document
from pathlib import Path

def test_same_content_has_same_document_id():
    pdf_bytes = b"sample pdf bytes"

    first_id = generate_document_id(pdf_bytes)
    second_id = generate_document_id(pdf_bytes)

    assert first_id == second_id


def test_different_content_has_different_document_id():
    first_id = generate_document_id(b"document A")
    second_id = generate_document_id(b"document B")

    assert first_id != second_id


def test_prepare_document_adds_document_id():
    pdf_path = Path("samples/rag_paper.pdf")

    pdf_bytes = pdf_path.read_bytes()

    chunks = prepare_document(pdf_bytes)

    expected_id = generate_document_id(pdf_bytes)

    assert len(chunks) > 0

    assert all(
        chunk["document_id"] == expected_id
        for chunk in chunks
    )

    