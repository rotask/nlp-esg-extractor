# Gold Labels — Conventions

This directory contains `gold_labels.csv`: hand-labeled ground truth for 5 reports
× 3 KPIs = 15 rows. The evaluation harness (`src/nlp_esg/evaluate.py`) scores
extractor predictions against these labels.

## Schema

| Column           | Type     | Notes |
| ---------------- | -------- | ----- |
| `company`        | string   | Must match the filename prefix in `data/reports/`. |
| `report_year`    | int      | Must match the year in the PDF filename. |
| `kpi`            | string   | One of: `scope_1_emissions`, `renewable_energy`, `water_consumption`. |
| `value`          | float    | The reported value **in the canonical unit** (tCO2e, MWh, m³). Blank = not reported. |
| `unit`           | string   | Canonical unit (`tCO2e`, `MWh`, `m3`) or blank. |
| `reporting_year` | int      | Year the value itself refers to (usually same as `report_year`). Blank if not reported. |
| `source_page`    | int      | PDF page where you found the value (1-indexed). |
| `notes`          | string   | Free-form. Useful for "why I chose this number". |

## Picking the canonical value

When a report states the KPI in multiple places (a narrative number and a performance
table), use the **performance table** value — that's considered the canonical form.

Convert to the canonical unit *yourself* before entering. The pipeline compares in
canonical units; an extractor that reads "1.2 GWh" and the gold labeled "1200 MWh"
should count as correct.

## Not reported vs. breakdown-only

If a company reports Scope 1 only as regional breakdowns (e.g., "EMEA: 12k,
APAC: 5k") with **no consolidated total**, label it as **not reported** (leave
`value`, `unit`, `reporting_year` blank). Do **not** sum the breakdowns —
the baseline extractor deliberately doesn't, so the eval has to use the same
convention.

## Out-of-range sanity check

If the value falls outside these plausibility ranges, double-check the unit:

| KPI                         | Plausible range (canonical) |
| --------------------------- | --------------------------- |
| Scope 1 emissions (tCO2e)   | 100 – 10⁹                   |
| Renewable energy (MWh)      | 100 – 10⁹                   |
| Water consumption (m³)      | 10 – 10¹⁰                   |
