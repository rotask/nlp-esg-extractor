from __future__ import annotations
from typing import Any

from nlp_esg.config import EPSILON
from nlp_esg.types import KPIExtraction


def is_correct(pred: KPIExtraction, gold: dict[str, Any]) -> bool:
    p_value = pred.value
    g_value = gold.get("value")

    if p_value is None and g_value is None:
        # Both "not reported" — unit/year are irrelevant in this branch.
        return True
    if p_value is None or g_value is None:
        return False

    if pred.unit != gold.get("unit"):
        return False
    if pred.reporting_year != gold.get("reporting_year"):
        return False
    if g_value == 0:
        return abs(p_value - g_value) < EPSILON
    return abs(p_value - g_value) / abs(g_value) <= EPSILON


def evaluate(
    preds: list[KPIExtraction],
    golds: list[dict[str, Any]],
    extractor: str,
    kpi: str,
) -> dict[str, float]:
    """
    Compute precision / recall / F1 / coverage for a single (extractor, kpi) slice.

    Matches preds to golds by (company, report_year).
    - Precision: TP / (TP + FP). FP = pred has a value but is incorrect (wrong number,
      wrong unit, wrong year, or hallucinated when gold is 'not reported').
    - Recall: TP / (TP + FN). FN = gold has a value but pred is wrong or 'not reported'.
    - Coverage: fraction of matched (pred, gold) pairs where the pred has a
      non-null value. When preds and golds are generated as parallel rows
      per company-year (the coursework case) this equals
      non_null_preds / len(preds).
    """
    pred_by_key = {(p.company, p.report_year): p for p in preds}
    tp = fp = fn = tn = 0
    total_preds = 0
    non_null_preds = 0

    for g in golds:
        key = (g["company"], g["report_year"])
        p = pred_by_key.get(key)
        if p is None:
            if g["value"] is None:
                tn += 1
            else:
                fn += 1
            continue

        total_preds += 1
        if p.value is not None:
            non_null_preds += 1

        correct = is_correct(p, g)
        if correct and p.value is not None:
            tp += 1
        elif correct and p.value is None:
            tn += 1
        elif not correct and p.value is not None:
            fp += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    coverage = non_null_preds / total_preds if total_preds > 0 else 0.0

    return {
        "extractor": extractor,
        "kpi": kpi,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "coverage": coverage,
    }
