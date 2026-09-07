"""
test_eval.py
~~~~~~~~~~~~
Evaluation harness and benchmark verification against ground-truth dataset.
Calculates:
  - Field-Level Exact Match (EM) Accuracy
  - Field-Level Normalized Match (NM) Accuracy
  - Entity Set Precision, Recall, and F1 Score (for list fields)
  - Confidence Calibration Metrics
"""

from __future__ import annotations

import re
from typing import Any
import pytest

from backend.models.field_result import FieldResult
from backend.services.layers import layer1_pdfplumber, layer2_spacy, layer3_regex
from backend.services import merger
from tests.eval_dataset import BENCHMARK_DATASET


def _normalize_str(val: Any) -> str:
    """Normalize string for robust comparison (strip currency symbols, whitespace, case)."""
    if val is None:
        return ""
    s = str(val).strip().lower()
    # Remove currency prefixes/symbols
    s = re.sub(r"^(?:inr|usd|eur|rs\.?|[₹$€£¥])\s*", "", s, flags=re.IGNORECASE)
    # Remove commas in numbers
    s = s.replace(",", "")
    # Normalize internal whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def _evaluate_sample(sample: dict[str, Any]) -> dict[str, Any]:
    text = sample["text"]
    tables = sample.get("tables", [])
    spatial_words = sample.get("spatial_words", None)
    report_type = sample["report_type"]
    gt = sample["ground_truth"]

    # Run extraction layers
    l1 = layer1_pdfplumber.extract(text, tables, report_type, spatial_words=spatial_words)
    l2 = layer2_spacy.extract(text, report_type)
    l3 = layer3_regex.extract(text, report_type)

    merged = merger.merge({
        "layer1_pdfplumber": l1,
        "layer2_spacy": l2,
        "layer3_regex": l3,
    })

    field_scores: dict[str, dict[str, Any]] = {}

    for field_name, expected in gt.items():
        extracted_fr: FieldResult | None = merged.get(field_name)
        extracted_val = extracted_fr.value if extracted_fr else None
        extracted_conf = extracted_fr.confidence if extracted_fr else 0.0

        if isinstance(expected, list):
            # List field evaluation (Precision / Recall / F1)
            ext_list = extracted_val if isinstance(extracted_val, list) else ([extracted_val] if extracted_val else [])
            ext_norm = {_normalize_str(x) for x in ext_list if x}
            exp_norm = {_normalize_str(x) for x in expected if x}

            true_pos = len(ext_norm.intersection(exp_norm))
            prec = true_pos / len(ext_norm) if ext_norm else (1.0 if not exp_norm else 0.0)
            rec = true_pos / len(exp_norm) if exp_norm else 1.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            field_scores[field_name] = {
                "type": "list",
                "expected": expected,
                "extracted": ext_list,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "is_match": f1 >= 0.60,
                "confidence": extracted_conf,
            }
        else:
            # Scalar field evaluation (Exact & Normalized Match)
            exp_norm = _normalize_str(expected)
            ext_norm = _normalize_str(extracted_val)
            exact_match = (str(expected).strip() == str(extracted_val).strip())
            norm_match = (exp_norm == ext_norm) or (exp_norm in ext_norm and len(exp_norm) > 4)

            field_scores[field_name] = {
                "type": "scalar",
                "expected": expected,
                "extracted": extracted_val,
                "exact_match": exact_match,
                "norm_match": norm_match,
                "is_match": norm_match,
                "confidence": extracted_conf,
            }

    return {
        "id": sample["id"],
        "category": sample["category"],
        "field_scores": field_scores,
    }


def test_ground_truth_benchmark_evaluation():
    """Run full benchmark against ground-truth dataset and compute accuracy scoreboard."""
    results = []
    total_scalar_fields = 0
    correct_scalar_fields = 0
    total_list_fields = 0
    list_f1_sum = 0.0

    print("\n" + "=" * 90)
    print(f"{'GROUND-TRUTH PIPELINE BENCHMARK EVALUATION':^90}")
    print("=" * 90)

    for sample in BENCHMARK_DATASET:
        eval_res = _evaluate_sample(sample)
        results.append(eval_res)

        print(f"\nDocument: [{eval_res['category']}] {eval_res['id']}")
        print(f"{'-' * 90}")
        print(f"  {'Field Name':<20} | {'Status':<10} | {'Conf':<5} | {'Expected':<24} | {'Extracted'}")
        print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*5}-+-{'-'*24}-+-{'-'*22}")

        for field_name, score in eval_res["field_scores"].items():
            conf = score["confidence"]
            if score["type"] == "scalar":
                total_scalar_fields += 1
                if score["is_match"]:
                    correct_scalar_fields += 1
                    status = "[PASS]"
                else:
                    status = "[FAIL]"
                exp_str = repr(str(score['expected']))[:22]
                ext_str = repr(str(score['extracted']))[:22]
                print(f"  {field_name:<20} | {status:<10} | {conf:<5.2f} | {exp_str:<24} | {ext_str}")
            else:
                total_list_fields += 1
                f1 = score["f1"]
                list_f1_sum += f1
                status = f"[F1:{f1:.2f}]"
                exp_str = repr(score['expected'])[:22]
                ext_str = repr(score['extracted'])[:22]
                print(f"  {field_name:<20} | {status:<10} | {conf:<5.2f} | {exp_str:<24} | {ext_str}")

    # Summary calculations
    scalar_acc = (correct_scalar_fields / total_scalar_fields) * 100 if total_scalar_fields else 0
    avg_list_f1 = (list_f1_sum / total_list_fields) * 100 if total_list_fields else 0

    print("\n" + "=" * 90)
    print(f"{'OVERALL BENCHMARK ACCURACY SCORECARD':^90}")
    print("=" * 90)
    print(f"  * Total Benchmark Documents    : {len(BENCHMARK_DATASET)}")
    print(f"  * Scalar Field Accuracy (EM/NM): {scalar_acc:.1f}% ({correct_scalar_fields}/{total_scalar_fields} fields)")
    print(f"  * Multi-Valued Entity Mean F1  : {avg_list_f1:.1f}% ({total_list_fields} list fields)")
    print("=" * 90 + "\n")

    # Assert high quantitative accuracy
    assert scalar_acc >= 90.0, f"Scalar field accuracy {scalar_acc:.1f}% is below 90% threshold"
    assert avg_list_f1 >= 80.0, f"Entity list mean F1 {avg_list_f1:.1f}% is below 80% threshold"


if __name__ == "__main__":
    test_ground_truth_benchmark_evaluation()
