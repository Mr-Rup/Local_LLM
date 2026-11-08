"""Document parsers for extracting text and structure from PDF, DOCX, and TXT files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class PageContent(BaseModel):
    page_number: int
    text: str


class DocumentContent(BaseModel):
    file_name: str
    file_type: str
    file_size_bytes: int
    total_pages: int = 1
    pages: List[PageContent] = Field(default_factory=list)
    full_text: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentParsingError(Exception):
    """Raised when parsing a document fails."""

    pass


class PDFParser:
    """PDF parser using PyMuPDF for fast C-level text and page extraction."""

    @staticmethod
    def parse(file_path: Path) -> DocumentContent:
        try:
            import pymupdf
        except ImportError:
            try:
                import fitz as pymupdf
            except ImportError as exc:
                raise DocumentParsingError("PyMuPDF is not installed. Install via pip.") from exc

        try:
            doc = pymupdf.open(file_path)
            pages: List[PageContent] = []
            text_parts: List[str] = []

            for page_idx in range(len(doc)):
                page = doc.load_page(page_idx)
                page_text = page.get_text("text").strip()
                if page_text:
                    pages.append(PageContent(page_number=page_idx + 1, text=page_text))
                    text_parts.append(page_text)

            meta = doc.metadata or {}
            doc.close()

            return DocumentContent(
                file_name=file_path.name,
                file_type="pdf",
                file_size_bytes=file_path.stat().st_size,
                total_pages=len(pages),
                pages=pages,
                full_text="\n\n".join(text_parts),
                metadata=meta,
            )
        except Exception as exc:
            raise DocumentParsingError(f"Failed to parse PDF '{file_path.name}': {exc}") from exc


class DocxParser:
    """DOCX parser using python-docx to extract paragraphs and tables."""

    @staticmethod
    def parse(file_path: Path) -> DocumentContent:
        try:
            import docx
        except ImportError as exc:
            raise DocumentParsingError("python-docx is not installed. Install via pip.") from exc

        try:
            doc = docx.Document(file_path)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

            # Also extract text from any embedded tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(
                        cell.text.strip() for cell in row.cells if cell.text.strip()
                    )
                    if row_text:
                        paragraphs.append(row_text)

            full_text = "\n\n".join(paragraphs)
            return DocumentContent(
                file_name=file_path.name,
                file_type="docx",
                file_size_bytes=file_path.stat().st_size,
                total_pages=1,
                pages=[PageContent(page_number=1, text=full_text)],
                full_text=full_text,
                metadata={},
            )
        except Exception as exc:
            raise DocumentParsingError(f"Failed to parse DOCX '{file_path.name}': {exc}") from exc


class TextParser:
    """Plain text parser supporting UTF-8 and Latin-1 fallback."""

    @staticmethod
    def parse(file_path: Path) -> DocumentContent:
        try:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(file_path, "r", encoding="latin-1") as f:
                    text = f.read()

            text = text.strip()
            return DocumentContent(
                file_name=file_path.name,
                file_type="txt",
                file_size_bytes=file_path.stat().st_size,
                total_pages=1,
                pages=[PageContent(page_number=1, text=text)],
                full_text=text,
                metadata={},
            )
        except Exception as exc:
            raise DocumentParsingError(
                f"Failed to parse text file '{file_path.name}': {exc}"
            ) from exc


def load_document(file_path: Path | str) -> DocumentContent:
    """Factory loader dispatching to the appropriate parser based on file extension."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext == ".pdf":
        return PDFParser.parse(path)
    elif ext == ".docx":
        return DocxParser.parse(path)
    elif ext in [".txt", ".md", ".csv", ".json"]:
        return TextParser.parse(path)
    else:
        raise DocumentParsingError(
            f"Unsupported file format '{ext}'. Supported formats: .pdf, .docx, .txt, .md"
        )
