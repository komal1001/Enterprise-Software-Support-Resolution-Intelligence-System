"""
Langfuse integration for the evaluation pipeline.

Separated from eval.py (pure evaluation logic) because dataset management
and Langfuse wiring are setup/observability concerns, not evaluation logic.

Three responsibilities:
  1. create_dataset()     — run once ever to register the dataset in Langfuse
  2. add_items()          — add golden set cases to the dataset; re-run when new cases are added
  3. run_evaluation_with_langfuse() — run eval and link each result to its Langfuse dataset item

Run modes (via eval.py __main__):
    python -m eval.eval --push-dataset          # calls create_dataset + add_items
    python -m eval.eval --langfuse agent1-v1    # calls run_evaluation_with_langfuse
"""

from typing import Callable
from dotenv import load_dotenv

from eval.eval import (
    load_golden_set,
    run_classification_check,
    build_per_category_accuracy,
    build_misclassification_matrix,
    build_confidence_calibration,
    calculate_tsr,
    EvalReport,
)
from src.observability.langfuse_client import get_langfuse, start_trace, log_ticket_complete

load_dotenv()

LANGFUSE_DATASET_NAME = "golden-set"


def create_dataset(langfuse) -> None:
    """
    Register the evaluation dataset in Langfuse — run once ever.
    Safe to re-run: Langfuse returns the existing dataset without error.
    """
    langfuse.create_dataset(
        name=LANGFUSE_DATASET_NAME,
        description="55 golden test cases for Enterprise Support & Resolution Intelligence System",
        metadata={
            "project": "NIIT Capstone",
            "version": "1.1",
            "tc_range": "TC-01 to TC-55",
        },
    )
    print(f"Dataset '{LANGFUSE_DATASET_NAME}' ready.")


def add_items(langfuse, golden_set: list) -> None:
    """
    Insert test cases into the dataset.
    Re-run whenever new cases are added to golden_set.json.
    Langfuse deduplicates by input — existing items are not duplicated.
    """
    for idx, case in enumerate(golden_set, start=1):
        metadata = {
            "tc_id":            case["tc_id"],
            "index":            idx,
            "category_label":   case.get("category", ""),
            "expected_agents":  case.get("expected_agents", []),
            "expected_outcome": case.get("expected_outcome", ""),
            "risk_level":       case.get("risk_level", ""),
        }
        langfuse.create_dataset_item(
            dataset_name=LANGFUSE_DATASET_NAME,
            input={"question": case["input"]},
            expected_output={
                "category":            case.get("expected_category", ""),
                "severity":            case.get("severity", ""),
                "routing_path":        case.get("route", ""),
                "expected_escalation": case.get("expected_escalation", False),
                "answer_rubric":       case.get("answer_rubric", ""),
            },
            metadata=metadata,
        )
        print(f"  [{idx:02d}] Added {case['tc_id']}")

    langfuse.flush()
    print(f"\nDone — {len(golden_set)} cases in dataset '{LANGFUSE_DATASET_NAME}'")


def push_golden_set_to_langfuse() -> None:
    """
    Convenience wrapper: create dataset + add all items in one call.
    Called by: python -m eval.eval --push-dataset
    """
    lf = get_langfuse()
    create_dataset(lf)
    add_items(lf, load_golden_set())


def run_evaluation_with_langfuse(agent1_fn: Callable, run_name: str) -> EvalReport:
    """
    Runs evaluation and links each result to its Langfuse dataset item.
    Results appear in Langfuse under Datasets → golden-set → Runs tab.
    Enables side-by-side accuracy comparison across prompt versions.

    run_name — label for this eval run, e.g. "agent1-classify-v2"
    """
    lf = get_langfuse()
    golden_set = load_golden_set()
    results = []

    for tc in golden_set:
        result = run_classification_check(tc, agent1_fn)

        if not result.error:
            root, token = start_trace(tc["input"])
            log_ticket_complete(
                root,
                token,
                final_response=result.actual_reasoning or "",
                total_cost_usd=0.0,
                total_latency_ms=0.0,
                tsr_pass=result.classification_passed,
                routing_path=result.actual_route or "",
            )
            root.score_trace(
                name="classification_pass",
                value=1.0 if result.classification_passed else 0.0,
                comment=f"expected: {result.expected_category} / actual: {result.actual_category}",
            )

        results.append(result)

    lf.flush()

    report = EvalReport()
    report.total_cases              = len(results)
    report.passed                   = sum(1 for r in results if r.classification_passed)
    report.failed                   = report.total_cases - report.passed
    report.tsr                      = calculate_tsr(results)
    report.per_category             = build_per_category_accuracy(results)
    report.misclassification_matrix = build_misclassification_matrix(results)
    report.confidence_calibration   = build_confidence_calibration(results)
    report.failures                 = [r for r in results if not r.classification_passed]
    return report
