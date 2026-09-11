from fastapi import FastAPI, UploadFile, File, HTTPException

from parser import extract_pages, PdfParsingError

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

    return {
        "filename" : file.filename,
        "total_pages" : len(pages),
        "pages": pages,
    }