import pytest

from chunker import (
    chunk_pages,
    chunk_text,
    combine_pages,
    find_page_numbers,
)

def test_chunk_text_with_overlap():
    text = "abcdefghij"

    chunks = chunk_text(
        text,
        chunk_size=4,
        overlap=1,
    )

    assert chunks == [
        "abcd",
        "defg",
        "ghij",
    ]


def test_chunk_text_returns_empty_list_for_empty_text():
    chunks = chunk_text("", chunk_size=4, overlap=1)

    assert chunks == []


def test_chunk_text_returns_one_chunk_for_short_text():
    chunks = chunk_text("abc", chunk_size=4, overlap=1)

    assert chunks == ["abc"]


def test_chunk_text_rejects_non_positive_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("abcdef", chunk_size=0, overlap=0)


def test_chunk_text_rejects_overlap_equal_to_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("abcdef", chunk_size=4, overlap=4)

def test_combine_pages_records_page_ranges():
    pages = [
        {"page_number": 1, "text": "abc"},
        {"page_number": 2, "text": "def"},
    ]

    document_text, page_ranges = combine_pages(pages)

    assert document_text == "abc\ndef"
    assert page_ranges == [
        {"page_number": 1, "start": 0, "end": 3},
        {"page_number": 2, "start": 4, "end": 7},
    ]

    assert document_text[0:3] == "abc"
    assert document_text[4:7] == "def"

def test_find_page_numbers_for_chunk_spanning_pages():
    page_ranges = [
        {"page_number": 1, "start": 0, "end": 10},
        {"page_number": 2, "start": 11, "end": 20},
    ]

    page_numbers = find_page_numbers(
        chunk_start=8,
        chunk_end=15,
        page_ranges=page_ranges,
    )

    assert page_numbers == [1, 2]


def test_chunk_pages_tracks_pages_across_boundary():
    pages = [
        {"page_number": 1, "text": "abcdef"},
        {"page_number": 2, "text": "ghijkl"},
    ]

    chunks = chunk_pages(
        pages,
        chunk_size=8,
        overlap=2,
    )

    assert chunks == [
        {
            "chunk_index": 0,
            "page_numbers": [1, 2],
            "text": "abcdef\ng",
        },
        {
            "chunk_index": 1,
            "page_numbers": [2],
            "text": "\nghijkl",
        },
    ]