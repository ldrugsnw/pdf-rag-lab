import fitz

class PdfParsingError(Exception):
    pass

def extract_pages(pdf_bytes: bytes): # Parser

    if not pdf_bytes:
        raise PdfParsingError("파일이 비어 있습니다.")
    
    try:
        doc = fitz.open(
            stream = pdf_bytes,
            filetype= "pdf"
        )

    except Exception as error:
        raise PdfParsingError("유효한 PDF가 아닙니다.") from error

    pages = []

    for index, page in enumerate(doc):
        raw_text = page.get_text("text", sort = True)
        text = normalize_text(raw_text)

        pages.append({
            "page_number": index + 1,
            "text": text
        })

    if not any(page_data["text"] for page_data in pages):
        raise PdfParsingError("PDF에서 텍스트를 찾을 수 없습니다.")
    
    return pages


def normalize_text(text: str) -> str:
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    return "\n".join(lines)
