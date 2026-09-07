"""
eval_dataset.py
~~~~~~~~~~~~~~~
Curated ground-truth benchmark dataset for quantitative pipeline evaluation.
Covers:
  - Invoices (Standard, GST, Scanned/OCR)
  - Resumes (Tech heavy, All-caps headers, Complex contact info)
  - Bank Statements (Account info, Balances, Transactions)
"""

from __future__ import annotations

from typing import Any
from backend.models.report_type import ReportType

def words_from_layout(lines: list[tuple[str, float, float]], page_idx: int = 0) -> list[dict[str, Any]]:
    """Generate spatial word dictionaries with bounding boxes from (text_line, x_start, y_top)."""
    spatial_words = []
    for line_text, x_start, y_top in lines:
        curr_x = x_start
        for word in line_text.split():
            w_width = max(len(word) * 6.5, 8.0)
            spatial_words.append({
                "text": word,
                "x0": round(curr_x, 1),
                "x1": round(curr_x + w_width, 1),
                "top": round(y_top, 1),
                "bottom": round(y_top + 12.0, 1),
                "page_idx": page_idx,
            })
            curr_x += w_width + 4.0
    return spatial_words


# ---------------------------------------------------------------------------
# BANK STATEMENT TEMPLATES WITH SPATIAL LAYOUTS
# ---------------------------------------------------------------------------

# Template 1: HDFC Inline Layout
_HDFC_INLINE_LINES = [
    ("HDFC BANK LIMITED", 40.0, 30.0),
    ("Account Statement", 40.0, 50.0),
    ("Account Holder: Rajesh Kumar Sharma", 40.0, 75.0),
    ("Account Number: 50100234567891", 40.0, 95.0),
    ("IFSC Code: HDFC0001234", 40.0, 115.0),
    ("Statement Period: 01/01/2024 to 31/01/2024", 40.0, 135.0),
    ("Email: rajesh.sharma@gmail.com | Phone: 9876543210", 40.0, 155.0),
    ("Opening Balance: INR 45,230.50", 40.0, 175.0),
    ("Closing Balance: INR 82,150.00", 40.0, 195.0),
]

# Template 2: ICICI Stacked 2-Line Layout (Label line 1, Value line 2 directly below)
_ICICI_STACKED_LINES = [
    ("ICICI BANK LIMITED", 40.0, 30.0),
    ("Retail Banking Statement", 40.0, 48.0),
    ("Account Number", 40.0, 70.0),
    ("102938475610", 40.0, 84.0),
    ("Account Name", 40.0, 106.0),
    ("Priya Patel", 40.0, 120.0),
    ("IFSC Code", 40.0, 142.0),
    ("ICIC0000921", 40.0, 156.0),
    ("Opening Balance", 40.0, 178.0),
    ("24,500.00", 40.0, 192.0),
    ("Closing Balance", 40.0, 214.0),
    ("68,900.50", 40.0, 228.0),
    ("Statement Period", 40.0, 250.0),
    ("01/02/2024 to 28/02/2024", 40.0, 264.0),
]

# Template 3: SBI 2-Column Layout (Side-by-side columns with horizontal gap)
_SBI_2COL_LINES = [
    ("STATE BANK OF INDIA", 40.0, 30.0),
    ("Account Summary", 40.0, 48.0),
    # Left column (x: 40.0)
    ("Account No: 20495810293", 40.0, 75.0),
    ("Customer Name: Amit Verma", 40.0, 95.0),
    ("Opening Balance: 15,200.00", 40.0, 115.0),
    ("IFSC: SBIN0004521", 40.0, 135.0),
    # Right column (x: 320.0)
    ("CIF Number: 883920194", 320.0, 75.0),
    ("Account Type: Savings", 320.0, 95.0),
    ("Closing Balance: 35,400.00", 320.0, 115.0),
    ("Statement Period: 01/03/2024 to 31/03/2024", 320.0, 135.0),
]

# Template 4: Axis Bank Boxed Grid Layout (Grid header row with value row directly beneath)
_AXIS_GRID_LINES = [
    ("AXIS BANK LIMITED", 40.0, 30.0),
    ("Statement of Account", 40.0, 48.0),
    # Grid Header Row 1 (y: 75.0)
    ("Account No", 40.0, 75.0),
    ("Account Holder", 160.0, 75.0),
    ("Open Balance", 300.0, 75.0),
    ("Close Balance", 420.0, 75.0),
    # Grid Value Row 1 (y: 92.0)
    ("91827364510", 40.0, 92.0),
    ("Sneha Joshi", 160.0, 92.0),
    ("52,100.00", 300.0, 92.0),
    ("94,300.00", 420.0, 92.0),
    # Grid Header Row 2 (y: 120.0)
    ("Statement Period", 40.0, 120.0),
    ("IFSC Code", 260.0, 120.0),
    # Grid Value Row 2 (y: 137.0)
    ("01/04/2024 to 30/04/2024", 40.0, 137.0),
    ("UTIB0000843", 260.0, 137.0),
]


BENCHMARK_DATASET: list[dict[str, Any]] = [
    # -----------------------------------------------------------------------
    # RESUME BENCHMARKS
    # -----------------------------------------------------------------------
    {
        "id": "resume_princy_patel",
        "category": "Resume",
        "report_type": ReportType.RESUME,
        "text": """PRINCY PATEL
princy.patel@gmail.com | +91 93282 12345 | linkedin.com/in/princypatel | github.com/Princy9114

EDUCATION
Gujarat Technological University
Bachelor of Technology in Computer Engineering
2021 - 2025

TECHNICAL SKILLS
Languages: Python, JavaScript, SQL, C++
Frameworks: FastAPI, PyTorch, TensorFlow, React
Tools & Libraries: Docker, OpenCV, YOLOv8, Git, Linux

EXPERIENCE
AI Engineer Intern
NexGen Technologies
May 2024 - Jul 2024
- Built an automated object tracking pipeline using YOLOv8 and OpenCV with 94% mAP.
- Deployed inference microservices with FastAPI and Docker.

PROJECTS
Smart Report Extractor
Jan 2025 - Present
- Multi-layer OCR and entity extraction engine for financial documents.
""",
        "tables": [],
        "ground_truth": {
            "name": "PRINCY PATEL",
            "email": "princy.patel@gmail.com",
            "phone": "+91 93282 12345",
            "profile_url": "linkedin.com/in/princypatel",
            "organizations": [
                "Gujarat Technological University",
                "NexGen Technologies",
            ],
            "dates": [
                "2021 - 2025",
                "May 2024 - Jul 2024",
                "Jan 2025 - Present",
            ],
        },
    },
    {
        "id": "resume_sarah_connor",
        "category": "Resume",
        "report_type": ReportType.RESUME,
        "text": """Sarah Connor | Data Scientist
sarah.connor@example.com · +1 (555) 342-8910 · github.com/sconnor

PROFESSIONAL SUMMARY
Data scientist with 3+ years experience in NLP and computer vision.

WORK EXPERIENCE
Senior Machine Learning Engineer
Cyberdyne Systems
Jun 2022 - Aug 2024
- Trained BERT and Transformer models for text classification.

Data Analyst
Acme Corporation
Aug 2020 - May 2022
- Maintained Tableau dashboards and SQL ETL pipelines.

EDUCATION
Stanford University
M.S. in Computer Science
2018 - 2020
""",
        "tables": [],
        "ground_truth": {
            "name": "Sarah Connor",
            "email": "sarah.connor@example.com",
            "phone": "+1 (555) 342-8910",
            "profile_url": "github.com/sconnor",
            "organizations": [
                "Cyberdyne Systems",
                "Acme Corporation",
                "Stanford University",
            ],
            "dates": [
                "Jun 2022 - Aug 2024",
                "Aug 2020 - May 2022",
                "2018 - 2020",
            ],
        },
    },

    # -----------------------------------------------------------------------
    # INVOICE BENCHMARKS
    # -----------------------------------------------------------------------
    {
        "id": "invoice_apex_solutions",
        "category": "Invoice",
        "report_type": ReportType.INVOICE,
        "text": """TAX INVOICE
Apex Solutions Pvt Ltd
Plot 45, MIDC Industrial Area, Mumbai 400604
GSTIN: 27AABCU9603R1ZM | PAN: AABCU9603R
Email: billing@apexsolutions.com | Phone: 022-25820309

Invoice No: INV-2024-8891
Date: 15/04/2024

Bill To:
Metro Retailers Ltd
Ahmedabad, Gujarat

Description       Qty    Unit Price    Total
Cloud Hosting       1       5000.00   5000.00
API Gateway        10        250.00   2500.00
---------------------------------------------
Subtotal                              7500.00
IGST @ 18%                            1350.00
Grand Total: ₹8,850.00
""",
        "tables": [
            [
                ["Description", "Qty", "Unit Price", "Total"],
                ["Cloud Hosting", "1", "5000.00", "5000.00"],
                ["API Gateway", "10", "250.00", "2500.00"],
            ]
        ],
        "ground_truth": {
            "invoice_number": "INV-2024-8891",
            "date": "15/04/2024",
            "gstin": "27AABCU9603R1ZM",
            "pan": "AABCU9603R",
            "email": "billing@apexsolutions.com",
            "phone": "022-25820309",
            "total_amount": "8,850.00",
            "vendor": "Apex Solutions Pvt Ltd",
        },
    },
    {
        "id": "invoice_tech_supplies",
        "category": "Invoice",
        "report_type": ReportType.INVOICE,
        "text": """INVOICE
Tech Supplies Global Inc
GSTIN: 29ABCDE1234F1Z5
Email: sales@techsupplies.com
Invoice Number: TS-90210
Invoice Date: 2024-03-20

Total Amount: USD 1,450.00
""",
        "tables": [],
        "ground_truth": {
            "invoice_number": "TS-90210",
            "date": "2024-03-20",
            "gstin": "29ABCDE1234F1Z5",
            "email": "sales@techsupplies.com",
            "total_amount": "1,450.00",
        },
    },

    # -----------------------------------------------------------------------
    # BANK STATEMENT BENCHMARKS (4 DISTINCT TEMPLATES)
    # -----------------------------------------------------------------------
    # Template 1: HDFC Inline Layout
    {
        "id": "bank_hdfc_inline",
        "category": "Bank Statement (Inline)",
        "report_type": ReportType.BANK_STATEMENT,
        "text": """HDFC BANK LIMITED
Account Statement

Account Holder: Rajesh Kumar Sharma
Account Number: 50100234567891
IFSC Code: HDFC0001234
Statement Period: 01/01/2024 to 31/01/2024
Email: rajesh.sharma@gmail.com | Phone: 9876543210
Opening Balance: INR 45,230.50
Closing Balance: INR 82,150.00
""",
        "spatial_words": words_from_layout(_HDFC_INLINE_LINES),
        "tables": [
            [
                ["Date", "Description", "Debit", "Credit", "Balance"],
                ["05/01/2024", "Salary Credit", "", "50000.00", "95230.50"],
                ["12/01/2024", "Electricity Bill", "3080.50", "", "92150.00"],
                ["20/01/2024", "Rent Payment", "10000.00", "", "82150.00"],
            ]
        ],
        "ground_truth": {
            "account_number": "50100234567891",
            "account_name": "Rajesh Kumar Sharma",
            "ifsc": "HDFC0001234",
            "email": "rajesh.sharma@gmail.com",
            "phone": "9876543210",
            "opening_balance": "45,230.50",
            "closing_balance": "82,150.00",
            "statement_period": "01/01/2024 to 31/01/2024",
        },
    },

    # Template 2: ICICI Stacked 2-Line Layout
    {
        "id": "bank_icici_stacked",
        "category": "Bank Statement (Stacked)",
        "report_type": ReportType.BANK_STATEMENT,
        "text": """ICICI BANK LIMITED
Retail Banking Statement

Account Number
102938475610
Account Name
Priya Patel
IFSC Code
ICIC0000921
Opening Balance
24,500.00
Closing Balance
68,900.50
Statement Period
01/02/2024 to 28/02/2024
""",
        "spatial_words": words_from_layout(_ICICI_STACKED_LINES),
        "tables": [],
        "ground_truth": {
            "account_number": "102938475610",
            "account_name": "Priya Patel",
            "ifsc": "ICIC0000921",
            "opening_balance": "24,500.00",
            "closing_balance": "68,900.50",
            "statement_period": "01/02/2024 to 28/02/2024",
        },
    },

    # Template 3: SBI 2-Column Side-by-Side Layout
    {
        "id": "bank_sbi_2column",
        "category": "Bank Statement (2-Column)",
        "report_type": ReportType.BANK_STATEMENT,
        "text": """STATE BANK OF INDIA
Account Summary

Account No: 20495810293         CIF Number: 883920194
Customer Name: Amit Verma       Account Type: Savings
Opening Balance: 15,200.00      Closing Balance: 35,400.00
IFSC: SBIN0004521               Statement Period: 01/03/2024 to 31/03/2024
""",
        "spatial_words": words_from_layout(_SBI_2COL_LINES),
        "tables": [],
        "ground_truth": {
            "account_number": "20495810293",
            "account_name": "Amit Verma",
            "ifsc": "SBIN0004521",
            "opening_balance": "15,200.00",
            "closing_balance": "35,400.00",
            "statement_period": "01/03/2024 to 31/03/2024",
        },
    },

    # Template 4: Axis Bank Boxed Grid Layout
    {
        "id": "bank_axis_boxed_grid",
        "category": "Bank Statement (Grid)",
        "report_type": ReportType.BANK_STATEMENT,
        "text": """AXIS BANK LIMITED
Statement of Account

Account No          Account Holder          Open Balance          Close Balance
91827364510         Sneha Joshi             52,100.00             94,300.00

Statement Period                            IFSC Code
01/04/2024 to 30/04/2024                    UTIB0000843
""",
        "spatial_words": words_from_layout(_AXIS_GRID_LINES),
        "tables": [],
        "ground_truth": {
            "account_number": "91827364510",
            "account_name": "Sneha Joshi",
            "ifsc": "UTIB0000843",
            "opening_balance": "52,100.00",
            "closing_balance": "94,300.00",
            "statement_period": "01/04/2024 to 30/04/2024",
        },
    },
]

