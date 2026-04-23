from enum import Enum
from pydantic import BaseModel, ConfigDict
from typing import Optional


class DocumentType(str, Enum):
    INVOICE = "invoice"
    RESUME = "resume"
    BANK_STATEMENT = "bank_statement"
    UNKNOWN = "unknown"


class ExtractionMethod(str, Enum):
    LLM = "llm"
    RULE_BASED = "rule_based"
    HYBRID = "hybrid"
    FAILED = "failed"


class BaseExtraction(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    document_type: DocumentType
    confidence: float  # 0.0 – 1.0 classifier confidence
    extraction_method: ExtractionMethod
    summary: Optional[str] = None
    warnings: list[str] = []
    raw_text_length: int = 0
