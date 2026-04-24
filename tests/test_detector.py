import pytest

from backend.models.report_type import ReportType
from backend.services.detector import detect_report_type

def test_detect_invoice():
    text = "Invoice Number: 12345\nVendor: Acme Corp\nBill To: John\nTotal Due: $500\nAmount Due: $500\nSubtotal and tax applied."
    report_type, conf = detect_report_type(text)
    assert report_type == ReportType.INVOICE
    assert conf > 0.4

def test_detect_bank_statement():
    text = "Bank Statement\nAccount Number: 987654321\nAvailable Balance: 1000\nOpening Balance: 1000\nClosing Balance: 500\nWithdrawal Debit Credit"
    report_type, conf = detect_report_type(text)
    assert report_type == ReportType.BANK_STATEMENT
    assert conf > 0.4

def test_detect_resume():
    text = "John Doe\nCurriculum Vitae\nResume\nProfessional Summary\nEmployment History and Work Experience\nEducation: BSc Computer Science\nSkills: Python, FastAPI\nProjects and Languages."
    report_type, conf = detect_report_type(text)
    assert report_type == ReportType.RESUME
    assert conf > 0.4

def test_detect_unknown():
    text = "Just a random little text note about groceries: milk, eggs, bread."
    report_type, conf = detect_report_type(text)
    assert report_type == ReportType.UNKNOWN
    assert conf == 0.0
