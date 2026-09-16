def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")    

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        if end >= len(text):
            break

        start += chunk_size - overlap

    return chunks

def combine_pages(
    pages: list[dict],
) -> tuple[str, list[dict]]:
    document_parts = []
    page_ranges = []
    current_position = 0

    for index, page in enumerate(pages):
        page_text = page["text"]

        start = current_position

        document_parts.append(page_text)
        current_position += len(page_text)

        end = current_position

        page_ranges.append({
            "page_number": page["page_number"],
            "start": start,
            "end": end,
        })

        if index < len(pages) - 1:
            document_parts.append("\n")
            current_position += 1

    document_text = "".join(document_parts)

    return document_text, page_ranges


def find_page_numbers(
    chunk_start: int,
    chunk_end: int,
    page_ranges: list[dict],
) -> list[int]:
    page_numbers = []

    for page_range in page_ranges:
        overlaps = (
            chunk_start < page_range["end"]
            and chunk_end > page_range["start"]
        )

        if overlaps:
            page_numbers.append(page_range["page_number"])

    return page_numbers

def chunk_pages(
    pages: list[dict],
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[dict]:
    document_text, page_ranges = combine_pages(pages)

    text_chunks = chunk_text(
        document_text,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    chunks = []
    step = chunk_size - overlap

    for chunk_index, text_chunk in enumerate(text_chunks):
        chunk_start = chunk_index * step
        chunk_end = chunk_start + len(text_chunk)

        page_numbers = find_page_numbers(
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            page_ranges=page_ranges,
        )

        chunks.append({
            "chunk_index": chunk_index,
            "page_numbers": page_numbers,
            "text": text_chunk,
        })

    return chunks
