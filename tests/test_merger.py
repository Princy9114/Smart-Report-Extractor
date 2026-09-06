import pytest
from backend.models.field_result import FieldResult
from backend.services.merger import merge


def test_merger_resolves_invoice_number_to_single_clean_value():
    layer_results = {
        "layer3_regex": {
            # List of raw matches including phone number, zip code, and real invoice number
            "invoice_number": FieldResult(
                value=["02225820309", "400604", "INV-2024-998"],
                confidence=0.90,
                source="regex",
            )
        },
        "layer1_pdfplumber": {
            "invoice_number": FieldResult(
                value="INV-2024-998",
                confidence=0.85,
                source="label_adjacent",
            )
        },
    }

    merged = merge(layer_results)
    assert "invoice_number" in merged
    # Must be a single scalar string, NOT a list
    assert isinstance(merged["invoice_number"].value, str)
    assert merged["invoice_number"].value == "INV-2024-998"
    assert "consensus" in merged["invoice_number"].source


def test_merger_rejects_unit_of_measure_and_picks_monetary_total():
    layer_results = {
        "layer1_pdfplumber": {
            # Bogus quantity match
            "total": FieldResult(value="2 NOS", confidence=0.85, source="label_adjacent")
        },
        "layer3_regex": {
            # List of amounts
            "total_amount": FieldResult(value=["2", "830.00"], confidence=0.90, source="regex")
        },
        "layer2_spacy": {
            "total_amount": FieldResult(value="830.00", confidence=0.75, source="spacy_ner")
        }
    }

    merged = merge(layer_results)
    assert "total_amount" in merged
    assert isinstance(merged["total_amount"].value, str)
    assert merged["total_amount"].value == "830.00"
    assert merged["total_amount"].value != "2 NOS"
    assert merged["total_amount"].value != "2"


def test_merger_cleans_and_resolves_phone():
    layer_results = {
        "layer3_regex": {
            "phone": FieldResult(
                value=["456378\n3800", "9876543210"],
                confidence=0.90,
                source="regex",
            )
        }
    }

    merged = merge(layer_results)
    assert "phone" in merged
    assert isinstance(merged["phone"].value, str)
    assert merged["phone"].value == "9876543210"


def test_merger_preserves_legitimate_list_fields():
    layer_results = {
        "layer1_pdfplumber": {
            "line_items": FieldResult(
                value=[{"item": "Widget", "qty": "2", "price": "100"}],
                confidence=0.90,
                source="table",
            )
        }
    }

    merged = merge(layer_results)
    assert "line_items" in merged
    assert isinstance(merged["line_items"].value, list)
    assert len(merged["line_items"].value) == 1


def test_merger_resolves_resume_name_rejecting_yolov8():
    layer_results = {
        "layer2_spacy": {
            # spaCy misclassified project keyword as PERSON
            "name": FieldResult(value="YOLOv8", confidence=0.75, source="spacy_ner")
        },
        "layer1_pdfplumber": {
            # Layer 1 positional header found the actual name
            "name": FieldResult(value="Princy Patel", confidence=0.90, source="header_positional")
        }
    }

    merged = merge(layer_results)
    assert "name" in merged
    assert merged["name"].value == "Princy Patel"
    assert merged["name"].value != "YOLOv8"


def test_merger_sanitizes_organizations_and_dates():
    layer_results = {
        "layer1_pdfplumber": {
            "name": FieldResult(value="Princy Patel", confidence=0.90, source="header_positional"),
        },
        "layer2_spacy": {
            # Contains candidate name in all-caps, platform name, and real company
            "organizations": FieldResult(
                value=["PRINCY PATEL", "LinkedIn", "Google", "GitHub"],
                confidence=0.75,
                source="spacy_ner",
            ),
            # Contains phone fragment and real dates
            "dates": FieldResult(
                value=["93282", "May 2025", "2021-2025", "1"],
                confidence=0.75,
                source="spacy_ner",
            ),
        }
    }

    merged = merge(layer_results)
    assert "organizations" in merged
    orgs = merged["organizations"].value
    assert "PRINCY PATEL" not in orgs
    assert "LinkedIn" not in orgs
    assert "GitHub" not in orgs
    assert "Google" in orgs

    assert "dates" in merged
    dates = merged["dates"].value
    assert "93282" not in dates
    assert "1" not in dates
    assert "May 2025" in dates
    assert "2021-2025" in dates
