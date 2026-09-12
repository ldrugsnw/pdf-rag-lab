from fastapi.testclient import TestClient
import fitz

from main import app

client = TestClient(app)

def test_reject_non_pdf_file():
    response = client.post(
        "/papers",
        files = {
            "file": (
                "example.txt",
                b"this is not a pdf",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail" : "PDF 파일만 업로드할 수 있습니다."
    }


def create_pdf_bytes(text: str) -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)

    pdf_bytes = document.tobytes()
    document.close()

    return pdf_bytes

def test_uploads_pdf_and_returns_pages():
    pdf_bytes = create_pdf_bytes("Hello API")

    response = client.post(
        "/papers",
        files={
            "file": (
                "example.pdf",
                pdf_bytes,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["filename"] == "example.pdf"
    assert data["total_pages"] == 1
    assert data["pages"][0]["page_number"] == 1
    assert "Hello API" in data["pages"][0]["text"]

    assert data["total_chunks"] == 1
    assert data["chunks"][0]["chunk_index"] == 0
    assert data["chunks"][0]["page_numbers"] == [1]
    assert "Hello API" in data["chunks"][0]["text"] 