"""
test_document_processing.py — Tests for document extraction.
Scenarios 4-8 from spec: short/long notes, empty input, invalid PDF, scanned PDF.
No real API calls.
"""
from __future__ import annotations

import io
import struct
import pytest

from backend.core.pipeline import (
    extract_text_from_txt,
    extract_text_from_pdf,
    extract_text,
    prepare_notes,
)
from backend.utils.validators import validate_text_input, validate_uploaded_file


# ---------------------------------------------------------------------------
# Fake file helper
# ---------------------------------------------------------------------------

class _FakeFile:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._buf = io.BytesIO(content)

    def seek(self, *args):
        self._buf.seek(*args)

    def tell(self):
        return self._buf.tell()

    def read(self, *args):
        return self._buf.read(*args)


# ---------------------------------------------------------------------------
# Scenario 4: Very short notes
# ---------------------------------------------------------------------------

class TestScenario4ShortNotes:
    def test_empty_string_rejected(self):
        ok, msg = validate_text_input("")
        assert not ok
        assert "empty" in msg.lower()

    def test_whitespace_only_rejected(self):
        ok, msg = validate_text_input("   \t\n  ")
        assert not ok

    def test_49_chars_rejected(self):
        ok, msg = validate_text_input("a" * 49)
        assert not ok
        assert "short" in msg.lower()

    def test_50_chars_accepted(self):
        ok, msg = validate_text_input("a" * 50)
        assert ok


# ---------------------------------------------------------------------------
# Scenario 5: Long notes
# ---------------------------------------------------------------------------

class TestScenario5LongNotes:
    def test_15001_chars_rejected(self):
        ok, msg = validate_text_input("w" * 15_001)
        assert not ok
        assert "long" in msg.lower()

    def test_prepare_notes_truncates_at_15000(self):
        from backend.config import MAX_TEXT_CHARS
        long_text = "word " * 4000   # ~20 000 chars
        result = prepare_notes(long_text)
        assert len(result) <= MAX_TEXT_CHARS + 200   # +200 for truncation message
        assert "truncated" in result.lower()

    def test_prepare_notes_cuts_at_sentence_boundary(self):
        # Build text with clear sentence boundaries
        sentences = ["This is sentence number " + str(i) + ". " for i in range(2000)]
        text = "".join(sentences)
        result = prepare_notes(text)
        # Should end close to a period
        assert result.rstrip().endswith(".") or "truncated" in result.lower()


# ---------------------------------------------------------------------------
# Scenario 6: Empty file input
# ---------------------------------------------------------------------------

class TestScenario6EmptyInput:
    def test_empty_txt_file_rejected_by_validator(self):
        f = _FakeFile("notes.txt", b"")
        ok, msg = validate_uploaded_file(f)
        assert not ok
        assert "empty" in msg.lower()

    def test_empty_txt_extraction_returns_empty_string(self):
        f = _FakeFile("notes.txt", b"")
        result = extract_text_from_txt(f)
        assert result == ""


# ---------------------------------------------------------------------------
# Scenario 7: Invalid (corrupted) PDF
# ---------------------------------------------------------------------------

class TestScenario7InvalidPDF:
    def test_corrupted_pdf_raises_exception(self):
        corrupt = b"%PDF-1.4 THIS IS CORRUPT GARBAGE %%%"
        f = _FakeFile("corrupt.pdf", corrupt)
        with pytest.raises(Exception):
            extract_text_from_pdf(f)

    def test_non_pdf_bytes_raises_exception(self):
        f = _FakeFile("fake.pdf", b"Not a PDF file at all")
        with pytest.raises(Exception):
            extract_text_from_pdf(f)


# ---------------------------------------------------------------------------
# Scenario 8: Scanned PDF (no extractable text)
# ---------------------------------------------------------------------------

class TestScenario8ScannedPDF:
    def _make_minimal_pdf(self) -> bytes:
        """Create a minimal valid PDF with no text content (mimics scanned PDF)."""
        return b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << >> >>
endobj
4 0 obj
<< /Length 0 >>
stream
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
trailer
<< /Size 5 /Root 1 0 R >>
startxref
316
%%EOF"""

    def test_scanned_pdf_raises_value_error_with_message(self):
        pdf_bytes = self._make_minimal_pdf()
        f = _FakeFile("scanned.pdf", pdf_bytes)
        with pytest.raises(ValueError) as exc_info:
            extract_text_from_pdf(f)
        assert "scanned" in str(exc_info.value).lower() or "text" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# TXT encoding edge cases
# ---------------------------------------------------------------------------

class TestTXTEncoding:
    def test_utf8_bom_handled(self):
        # UTF-8 BOM prefix
        content = b"\xef\xbb\xbf" + "Biology notes about cells.".encode("utf-8")
        f = _FakeFile("notes.txt", content)
        result = extract_text_from_txt(f)
        assert "Biology" in result

    def test_latin1_encoding_decoded(self):
        content = "Caf\xe9 is a French word for coffee.".encode("latin-1")
        f = _FakeFile("notes.txt", content)
        result = extract_text_from_txt(f)
        assert "Caf" in result

    def test_cp1252_encoding_decoded(self):
        # Build cp1252 bytes directly: 0x96 is the en-dash in cp1252
        content = b"Smart " + bytes([0x96]) + b" quotes test."
        f = _FakeFile("notes.txt", content)
        result = extract_text_from_txt(f)
        assert "Smart" in result


# ---------------------------------------------------------------------------
# File validation edge cases
# ---------------------------------------------------------------------------

class TestFileValidation:
    def test_unsupported_extension_rejected(self):
        f = _FakeFile("lecture.docx", b"some content")
        ok, msg = validate_uploaded_file(f)
        assert not ok
        assert "unsupported" in msg.lower()

    def test_oversized_file_rejected(self):
        f = _FakeFile("big.txt", b"x" * (6 * 1024 * 1024))
        ok, msg = validate_uploaded_file(f)
        assert not ok
        assert "large" in msg.lower()

    def test_valid_pdf_extension_accepted(self):
        f = _FakeFile("lecture.pdf", b"%PDF fake but valid size")
        ok, msg = validate_uploaded_file(f)
        assert ok

    def test_none_file_rejected(self):
        ok, msg = validate_uploaded_file(None)
        assert not ok
