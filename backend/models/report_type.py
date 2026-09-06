"""
report_type.py
~~~~~~~~~~~~~~
Enum representing every document category the extractor can recognise.
"""

from enum import StrEnum


class ReportType(StrEnum):
    INVOICE = "invoice"
    BANK_STATEMENT = "bank_statement"
    RESUME = "resume"
    UNKNOWN = "unknown"
