"""
eval.py
~~~~~~~
API route for retrieving live pipeline benchmark and per-layer evaluation metrics.
"""

from __future__ import annotations

from fastapi import APIRouter
from backend.services.evaluation import run_benchmark_evaluation, generate_markdown_report

router = APIRouter(prefix="/eval", tags=["Evaluation"])


@router.get("/benchmark", summary="Run ground-truth evaluation and return per-layer metrics")
async def get_benchmark_report():
    """Runs the quantitative benchmark against ground-truth documents and returns

    per-layer precision, recall, F1, exact match rate, latency, and attribution breakdown.
    """
    report_data = run_benchmark_evaluation()
    markdown_str = generate_markdown_report(report_data)

    metrics_summary = {}
    for name, m in report_data["metrics"].items():
        metrics_summary[name] = {
            "precision": round(m.precision, 1),
            "recall": round(m.recall, 1),
            "f1_score": round(m.f1, 1),
            "exact_match_rate": round(m.exact_match_rate, 1),
            "avg_latency_ms": round(m.avg_latency_ms, 2),
            "true_positives": m.true_positives,
            "false_positives": m.false_positives,
            "false_negatives": m.false_negatives,
        }

    return {
        "status": "success",
        "total_documents": report_data["total_documents"],
        "metrics": metrics_summary,
        "attribution_percentages": {k: round(v, 1) for k, v in report_data["attribution_percentages"].items()},
        "category_accuracy": {
            cat: f"{round((data['correct'] / data['fields'] * 100), 1)}%"
            for cat, data in report_data["category_breakdown"].items()
            if data["fields"] > 0
        },
        "markdown_report": markdown_str,
    }
