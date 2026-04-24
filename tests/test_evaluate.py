import pytest
from nlp_esg.evaluate import evaluate, is_correct
from nlp_esg.types import KPIExtraction


def _pred(value, unit="tCO2e", year=2024, flags=None, company="acme"):
    return KPIExtraction(
        company=company, report_year=2024, kpi="scope_1_emissions",
        value=value, unit=unit, reporting_year=year,
        source_snippet=None, source_page=None, confidence=1.0,
        extractor="baseline", flags=flags or [],
    )


def _gold(value, unit="tCO2e", year=2024, company="acme"):
    return {
        "company": company, "report_year": 2024, "kpi": "scope_1_emissions",
        "value": value, "unit": unit, "reporting_year": year,
    }


def test_is_correct_exact_match():
    assert is_correct(_pred(1000.0), _gold(1000.0))


def test_is_correct_within_epsilon():
    assert is_correct(_pred(1000.5), _gold(1000.0))
    assert is_correct(_pred(1005.0), _gold(1000.0))


def test_is_correct_outside_epsilon():
    assert not is_correct(_pred(1020.0), _gold(1000.0))


def test_is_correct_unit_mismatch():
    assert not is_correct(_pred(1000.0, unit="MWh"), _gold(1000.0, unit="tCO2e"))


def test_is_correct_year_mismatch():
    assert not is_correct(_pred(1000.0, year=2023), _gold(1000.0, year=2024))


def test_is_correct_both_not_reported():
    assert is_correct(_pred(None, unit=None, year=None), _gold(None, unit=None, year=None))


def test_is_correct_pred_hallucinates():
    assert not is_correct(_pred(1000.0), _gold(None, unit=None, year=None))


def test_is_correct_missed():
    assert not is_correct(_pred(None, unit=None, year=None), _gold(1000.0))


def test_evaluate_perfect_predictions():
    preds = [_pred(1000.0, company="acme_a"), _pred(2000.0, year=2024, company="acme_b")]
    golds = [_gold(1000.0, company="acme_a"), _gold(2000.0, company="acme_b")]
    m = evaluate(preds, golds, extractor="baseline", kpi="scope_1_emissions")
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_evaluate_one_false_positive():
    preds = [_pred(1000.0, company="acme_a"), _pred(42.0, company="acme_b")]
    golds = [_gold(1000.0, company="acme_a"),
             _gold(None, unit=None, year=None, company="acme_b")]
    m = evaluate(preds, golds, extractor="baseline", kpi="scope_1_emissions")
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(2 * 0.5 * 1.0 / 1.5)


def test_evaluate_coverage():
    preds = [_pred(1000.0, company="acme_a"),
             _pred(None, unit=None, year=None, company="acme_b"),
             _pred(2000.0, company="acme_c")]
    golds = [_gold(1000.0, company="acme_a"),
             _gold(2000.0, company="acme_b"),
             _gold(None, unit=None, year=None, company="acme_c")]
    m = evaluate(preds, golds, extractor="baseline", kpi="scope_1_emissions")
    assert m["coverage"] == pytest.approx(2 / 3)
