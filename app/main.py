from fastapi import FastAPI, UploadFile, File, HTTPException

from app.chunker import chunk_pages
from app.parser import extract_pages, PdfParsingError

app = FastAPI()

@app.post("/papers")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code = 400,
            detail = "PDF 파일만 업로드할 수 있습니다.",
        )
    
    pdf_bytes = await file.read()
    try:
        pages = extract_pages(pdf_bytes)
    except PdfParsingError as error:
        raise HTTPException(
            status_code = 400,
            detail=str(error),
        ) from error

    chunks = chunk_pages(pages)
    
    return {
        "filename" : file.filename,
        "total_pages" : len(pages),
         "total_chunks": len(chunks),
        "pages": pages,
        "chunks": chunks,
    }
