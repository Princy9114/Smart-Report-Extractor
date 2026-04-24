"""
report_type.py
~~~~~~~~~~~~~~
Enum representing every document category the extractor can recognise.
"""

from enum import Enum


class ReportType(str, Enum):
    INVOICE = "invoice"
    BANK_STATEMENT = "bank_statement"
    RESUME = "resume"
    UNKNOWN = "unknown"
