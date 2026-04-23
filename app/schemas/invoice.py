from typing import Optional
from pydantic import BaseModel
from .base import BaseExtraction


class LineItem(BaseModel):
    description: Optional[str] = None
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    amount: Optional[float] = None


class InvoiceExtraction(BaseExtraction):
    vendor_name: Optional[str] = None
    vendor_address: Optional[str] = None
    client_name: Optional[str] = None
    client_address: Optional[str] = None
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    line_items: list[LineItem] = []
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    discount: Optional[float] = None
    total_amount: Optional[float] = None
    currency: Optional[str] = None
    payment_terms: Optional[str] = None
    notes: Optional[str] = None
