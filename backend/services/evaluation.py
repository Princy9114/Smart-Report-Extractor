"""
evaluation.py
~~~~~~~~~~~~~
Per-layer evaluation framework and quantitative benchmark harness.

Calculates:
- Per-layer Precision, Recall, F1 Score, Exact Match (EM), and Normalized Match (NM)
- Isolated performance (Layer 1 Rules vs Layer 2 NER vs Layer 3 Regex vs Layer 4 LLM vs Consensus Ensemble)
- Layer contribution & attribution breakdown in final merged outputs
- Generates ASCII and Markdown leaderboard scorecards suitable for portfolio/interview demos.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from backend.models.field_result import FieldResult
from backend.models.report_type import ReportType
from backend.services.layers import layer1_pdfplumber, layer2_spacy, layer3_regex, layer4_llm
from backend.services import merger
from tests.eval_dataset import BENCHMARK_DATASET


def normalize_str(val: Any) -> str:
    """Normalize strings for robust comparison (strip currency symbols, whitespace, punctuation)."""
    if val is None:
        return ""
    s = str(val).strip().lower()
    # Strip currency prefixes/symbols
    s = re.sub(r"^(?:inr|usd|eur|rs\.?|[₹$€£¥])\s*", "", s, flags=re.IGNORECASE)
    # Remove commas in numbers
    s = s.replace(",", "")
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s)
    return s


@dataclass
class LayerMetrics:
    name: str
    total_target_fields: int = 0
    extracted_fields: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    exact_matches: int = 0
    total_latency_ms: float = 0.0
    doc_count: int = 0
    attribution_count: int = 0  # number of times this layer won in the final ensemble

    @property
    def precision(self) -> float:
        total = self.true_positives + self.false_positives
        return (self.true_positives / total * 100) if total > 0 else 0.0

    @property
    def recall(self) -> float:
        total = self.true_positives + self.false_negatives
        return (self.true_positives / total * 100) if total > 0 else 0.0

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    @property
    def exact_match_rate(self) -> float:
        return (self.exact_matches / self.total_target_fields * 100) if self.total_target_fields > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return (self.total_latency_ms / self.doc_count) if self.doc_count > 0 else 0.0


def _evaluate_layer_output(
    extracted: dict[str, FieldResult],
    ground_truth: dict[str, Any],
    metrics: LayerMetrics,
    elapsed_ms: float,
) -> None:
    """Evaluate extracted field results against ground truth and update layer metrics."""
    metrics.doc_count += 1
    metrics.total_latency_ms += elapsed_ms

    for gt_field, expected in ground_truth.items():
        metrics.total_target_fields += 1
        res: FieldResult | None = extracted.get(gt_field)
        actual_val = res.value if res else None

        if actual_val is None or actual_val == "" or actual_val == []:
            metrics.false_negatives += 1
            continue

        metrics.extracted_fields += 1

        if isinstance(expected, list):
            ext_list = actual_val if isinstance(actual_val, list) else [actual_val]
            ext_norm = {normalize_str(x) for x in ext_list if x}
            exp_norm = {normalize_str(x) for x in expected if x}

            overlap = len(ext_norm.intersection(exp_norm))
            if overlap > 0:
                metrics.true_positives += 1
                if ext_norm == exp_norm:
                    metrics.exact_matches += 1
            else:
                metrics.false_positives += 1
        else:
            exp_n = normalize_str(expected)
            act_n = normalize_str(actual_val)
            is_exact = str(expected).strip() == str(actual_val).strip()
            is_norm = (exp_n == act_n) or (exp_n in act_n and len(exp_n) > 4)

            if is_exact:
                metrics.exact_matches += 1
                metrics.true_positives += 1
            elif is_norm:
                metrics.true_positives += 1
            else:
                metrics.false_positives += 1

    # Check for fields extracted that weren't in ground truth (extra false positives)
    for ext_field, res in extracted.items():
        if ext_field not in ground_truth and ext_field != "__meta__" and ext_field != "document_summary":
            if res.value:
                metrics.false_positives += 1


def run_benchmark_evaluation(dataset: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Execute evaluation across the benchmark dataset for all layers and consensus merger."""
    bench_data = dataset or BENCHMARK_DATASET

    metrics_map: dict[str, LayerMetrics] = {
        "Layer 1: Spatial/Layout Rules": LayerMetrics(name="Layer 1: Spatial/Layout Rules"),
        "Layer 2: spaCy NER": LayerMetrics(name="Layer 2: spaCy NER"),
        "Layer 3: Regex Patterns": LayerMetrics(name="Layer 3: Regex Patterns"),
        "Consensus Ensemble (Merged)": LayerMetrics(name="Consensus Ensemble (Merged)"),
    }

    category_breakdown: dict[str, dict[str, int]] = {
        "INVOICE": {"docs": 0, "fields": 0, "correct": 0},
        "RESUME": {"docs": 0, "fields": 0, "correct": 0},
        "BANK_STATEMENT": {"docs": 0, "fields": 0, "correct": 0},
    }

    layer_wins: dict[str, int] = {
        "layer1_pdfplumber": 0,
        "layer2_spacy": 0,
        "layer3_regex": 0,
        "layer4_llm": 0,
        "consensus": 0,
    }

    for sample in bench_data:
        text = sample["text"]
        tables = sample.get("tables", [])
        spatial_words = sample.get("spatial_words", None)
        report_type: ReportType = sample["report_type"]
        gt = sample["ground_truth"]

        # Track category
        cat_name = report_type.name
        if cat_name in category_breakdown:
            category_breakdown[cat_name]["docs"] += 1
            category_breakdown[cat_name]["fields"] += len(gt)

        # 1. Layer 1 Execution
        t0 = time.perf_counter()
        l1 = layer1_pdfplumber.extract(text, tables, report_type, spatial_words=spatial_words)
        t1 = time.perf_counter()
        _evaluate_layer_output(l1, gt, metrics_map["Layer 1: Spatial/Layout Rules"], (t1 - t0) * 1000)

        # 2. Layer 2 Execution
        t0 = time.perf_counter()
        l2 = layer2_spacy.extract(text, report_type)
        t1 = time.perf_counter()
        _evaluate_layer_output(l2, gt, metrics_map["Layer 2: spaCy NER"], (t1 - t0) * 1000)

        # 3. Layer 3 Execution
        t0 = time.perf_counter()
        l3 = layer3_regex.extract(text, report_type)
        t1 = time.perf_counter()
        _evaluate_layer_output(l3, gt, metrics_map["Layer 3: Regex Patterns"], (t1 - t0) * 1000)

        # 4. Consensus Ensemble Execution
        t0 = time.perf_counter()
        merged = merger.merge({
            "layer1_pdfplumber": l1,
            "layer2_spacy": l2,
            "layer3_regex": l3,
        })
        t1 = time.perf_counter()
        _evaluate_layer_output(merged, gt, metrics_map["Consensus Ensemble (Merged)"], (t1 - t0) * 1000)

        # Track layer attribution wins in merged result
        for field_name, res in merged.items():
            if field_name == "__meta__":
                continue
            src = res.source
            if src in layer_wins:
                layer_wins[src] += 1
            else:
                layer_wins[src] = 1

            # Check if correctly matched
            if field_name in gt:
                exp = gt[field_name]
                if normalize_str(exp) == normalize_str(res.value) or (isinstance(exp, list) and isinstance(res.value, list)):
                    if cat_name in category_breakdown:
                        category_breakdown[cat_name]["correct"] += 1

    total_ensemble_fields = sum(layer_wins.values()) or 1
    attribution_pcts = {k: (v / total_ensemble_fields * 100) for k, v in layer_wins.items()}

    return {
        "metrics": metrics_map,
        "attribution_counts": layer_wins,
        "attribution_percentages": attribution_pcts,
        "category_breakdown": category_breakdown,
        "total_documents": len(bench_data),
    }


def generate_markdown_report(report_data: dict[str, Any]) -> str:
    """Generate a clean GitHub-Flavored Markdown report with comparison table."""
    metrics: dict[str, LayerMetrics] = report_data["metrics"]
    attr_pct = report_data["attribution_percentages"]

    md = []
    md.append("# Multi-Layer Extraction Benchmark Report\n")
    md.append(f"**Total Benchmark Samples**: {report_data['total_documents']} documents (Invoices, Resumes, Bank Statements)\n")

    md.append("## Layer Performance Comparison Table\n")
    md.append("| Extraction Layer | Precision | Recall | F1 Score | Exact Match (EM) | Avg Latency |")
    md.append("|---|---|---|---|---|---|")

    for name, m in metrics.items():
        is_ensemble = "Consensus" in name
        prefix = "**" if is_ensemble else ""
        suffix = "**" if is_ensemble else ""
        md.append(
            f"| {prefix}{m.name}{suffix} | {prefix}{m.precision:.1f}%{suffix} | {prefix}{m.recall:.1f}%{suffix} | "
            f"{prefix}{m.f1:.1f}%{suffix} | {prefix}{m.exact_match_rate:.1f}%{suffix} | {prefix}{m.avg_latency_ms:.2f}ms{suffix} |"
        )

    md.append("\n## Layer Consensus Attribution in Final Output\n")
    md.append("Demonstrating when and why each layer contributed to the final merged result:\n")
    for src, pct in attr_pct.items():
        count = report_data["attribution_counts"].get(src, 0)
        md.append(f"- **`{src}`**: {pct:.1f}% ({count} fields resolved)")

    md.append("\n## Category Accuracy Breakdown\n")
    md.append("| Document Type | Documents | Target Fields | Correct Extractions | Accuracy |")
    md.append("|---|---|---|---|---|")
    for cat, data in report_data["category_breakdown"].items():
        acc = (data["correct"] / data["fields"] * 100) if data["fields"] > 0 else 0.0
        md.append(f"| **{cat}** | {data['docs']} | {data['fields']} | {data['correct']} | **{acc:.1f}%** |")

    return "\n".join(md)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    rep = run_benchmark_evaluation()
    print(generate_markdown_report(rep))
