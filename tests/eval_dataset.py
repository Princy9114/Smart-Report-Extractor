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
    # BANK STATEMENT BENCHMARKS
    # -----------------------------------------------------------------------
    {
        "id": "bank_hdfc_statement",
        "category": "Bank Statement",
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
            "ifsc": "HDFC0001234",
            "email": "rajesh.sharma@gmail.com",
            "phone": "9876543210",
            "opening_balance": "45,230.50",
            "closing_balance": "82,150.00",
        },
    },
]
