from typing import Optional
from pydantic import BaseModel
from .base import BaseExtraction


class Transaction(BaseModel):
    date: Optional[str] = None
    description: Optional[str] = None
    debit: Optional[float] = None
    credit: Optional[float] = None
    balance: Optional[float] = None


class BankStatementExtraction(BaseExtraction):
    bank_name: Optional[str] = None
    account_holder: Optional[str] = None
    account_number_masked: Optional[str] = None
    account_type: Optional[str] = None
    statement_period_start: Optional[str] = None
    statement_period_end: Optional[str] = None
    opening_balance: Optional[float] = None
    closing_balance: Optional[float] = None
    total_credits: Optional[float] = None
    total_debits: Optional[float] = None
    transactions: list[Transaction] = []
    currency: Optional[str] = None
