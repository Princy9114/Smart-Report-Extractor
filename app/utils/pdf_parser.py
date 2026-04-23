import pdfplumber
import io
from dataclasses import dataclass


@dataclass
class ParsedPDF:
    full_text: str
    pages: list[str]
    tables: list[list[list[str]]]  # page → table → rows → cells
    num_pages: int
    metadata: dict


def parse_pdf(file_bytes: bytes) -> ParsedPDF:
    """
    Extract text, tables, and metadata from PDF bytes using pdfplumber.
    pdfplumber is preferred over PyPDF2 because it handles tables far better
    and preserves spatial layout of text, which matters for invoices and statements.
    """
    pages_text = []
    all_tables = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        metadata = pdf.metadata or {}
        num_pages = len(pdf.pages)

        for page in pdf.pages:
            # Extract text preserving layout
            text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            pages_text.append(text)

            # Extract tables — critical for invoices and bank statements
            tables = page.extract_tables()
            for table in tables:
                clean_table = []
                for row in table:
                    clean_row = [str(cell).strip() if cell else "" for cell in row]
                    clean_table.append(clean_row)
                all_tables.append(clean_table)

    full_text = "\n\n--- PAGE BREAK ---\n\n".join(pages_text)

    return ParsedPDF(
        full_text=full_text,
        pages=pages_text,
        tables=all_tables,
        num_pages=num_pages,
        metadata=metadata,
    )


def extract_tables_as_text(tables: list[list[list[str]]]) -> str:
    """Convert extracted tables to a readable text representation for LLM context."""
    result = []
    for i, table in enumerate(tables):
        result.append(f"\n[TABLE {i + 1}]")
        for row in table:
            result.append(" | ".join(cell for cell in row if cell))
    return "\n".join(result)
