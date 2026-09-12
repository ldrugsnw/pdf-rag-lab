import pytest
import fitz

from parser import extract_pages, PdfParsingError, normalize_text

def create_pdf_bytes(text: str) -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)

    pdf_bytes = document.tobytes()
    document.close()

    return pdf_bytes


def test_empty_file_raises_error():
    with pytest.raises(PdfParsingError):
        extract_pages(b"")

def test_non_pdf_data_raises_error():
    with pytest.raises(PdfParsingError):
        extract_pages(b"this is not a pdf!!!")


def test_extracts_text_from_pdf():
    pdf_bytes = create_pdf_bytes("Hello PDF")

    pages = extract_pages(pdf_bytes)

    assert len(pages) == 1
    assert pages[0]["page_number"] == 1
    assert "Hello PDF" in pages[0]["text"]

def create_blank_pdf_bytes() -> bytes:
    document = fitz.open()
    document.new_page()

    pdf_bytes = document.tobytes()
    document.close()

    return pdf_bytes

def test_blank_pdf_raises_error():
    pdf_bytes = create_blank_pdf_bytes()

    with pytest.raises(PdfParsingError):
        extract_pages(pdf_bytes)

def test_normalize_text_removes_empty_lines_and_spaces():
    text = "   첫 번째 줄 \n\n 두 번째 줄 \n"

    normalized = normalize_text(text)

    assert normalized == "첫 번째 줄\n두 번째 줄"