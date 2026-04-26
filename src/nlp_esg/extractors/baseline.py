from __future__ import annotations
import logging
import re
from typing import Any

import numpy as np
from nlp_esg.config import KPIS, TAU_TABLE, TOP_K_SENTENCES
from nlp_esg.extractors.base import Extractor
from nlp_esg.normalize import (
    _NUMBER_RE,
    canonicalize_unit,
    normalize_co2,
    parse_number,
    parse_value,
    to_canonical_value,
)
from nlp_esg.retrieval import cosine_sim, embed_texts, top_k
from nlp_esg.types import KPIExtraction

log = logging.getLogger(__name__)

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _structural_score(headers: list[str], report_year: int) -> float:
    for h in headers:
        if _YEAR_RE.search(h or "") and str(report_year) in (h or ""):
            return 1.0
    return 0.5


def _find_year_col(headers: list[str], report_year: int) -> int | None:
    """Return the column index of the MOST RECENT year found in headers.

    ESG reports typically show the current reporting year as the leftmost data
    column.  Using the most-recent year rather than trying to match report_year
    (derived from the filename) avoids mismatches when a "2024"-named file
    actually covers fiscal year 2025.
    """
    best: tuple[int, int] | None = None
    for i, h in enumerate(headers):
        m = _YEAR_RE.search(h or "")
        if m:
            year = int(m.group(0))
            if best is None or year > best[1]:
                best = (i, year)
    return best[0] if best else None


def _effective_headers_and_rows(
    table: dict,
) -> tuple[list[str], list[list[str]]]:
    """Return (headers, rows), promoting row-0 to headers when no year is found
    in the nominal table headers (pdfplumber sometimes places year labels there)."""
    headers = table["headers"]
    rows = table["rows"]
    if not any(_YEAR_RE.search(h or "") for h in headers):
        if rows and any(_YEAR_RE.search(c or "") for c in rows[0]):
            return list(rows[0]), rows[1:]
    return headers, rows


def _row_score(query: str, query_tokens: set[str], row_label: str) -> float:
    """Score how well a table row label matches the KPI query.

    Prefers rows where the normalised query phrase appears verbatim.
    Falls back to token-overlap when no phrase match is found.
    """
    label_lower = row_label.lower()
    query_lower = query.lower()
    if query_lower in label_lower:
        return 1.0
    label_tokens = set(label_lower.split())
    overlap = len(query_tokens & label_tokens)
    if overlap == len(query_tokens):
        return 0.8
    return overlap / max(1, len(query_tokens)) * 0.6


def _infer_unit_from_row_or_header(
    headers: list[str], row: list[str], value_col: int, unit_family_canonicals: set[str]
) -> str | None:
    """Look for a unit in: the cell itself (trailing), a 'Unit' column, or header annotation."""
    # 1. Check the value cell itself ("45,678 tCO2e")
    cell = row[value_col] if value_col < len(row) else ""
    pv = parse_value(cell, kpi_unit_family=list(unit_family_canonicals))
    if pv:
        return pv[1]

    # 2. Look for a 'Unit' column
    for i, h in enumerate(headers):
        if (h or "").strip().lower() in ("unit", "units"):
            try:
                u = canonicalize_unit(row[i])
            except (ValueError, IndexError):
                continue
            if u in unit_family_canonicals:
                return u

    # 3. Check the value column's own header (e.g., "2024 (tCO2e)")
    if value_col < len(headers):
        for token in re.findall(r"[A-Za-z0-9µ³²\-]+", normalize_co2(headers[value_col] or "")):
            try:
                u = canonicalize_unit(token)
            except ValueError:
                continue
            if u in unit_family_canonicals:
                return u

    # 4. Check the KPI-row header (first cell) for a unit
    if row:
        for token in re.findall(r"[A-Za-z0-9µ³²\-]+", normalize_co2(row[0] or "")):
            try:
                u = canonicalize_unit(token)
            except ValueError:
                continue
            if u in unit_family_canonicals:
                return u

    # 5. Check the table-level headers for a unit annotation
    for h in headers:
        for token in re.findall(r"[A-Za-z0-9µ³²\-]+", normalize_co2(h or "")):
            try:
                u = canonicalize_unit(token)
            except ValueError:
                continue
            if u in unit_family_canonicals:
                return u

    return None


class BaselineExtractor(Extractor):
    name = "baseline"

    def extract(self, report: Any, kpi_key: str) -> KPIExtraction:
        kpi = KPIS[kpi_key]
        flags: list[str] = []
        unit_family_canonicals: set[str] = set()
        for u in kpi["unit_family"]:
            try:
                unit_family_canonicals.add(canonicalize_unit(u))
            except ValueError:
                continue

        query_emb = embed_texts([kpi["query"]])[0]
        query_tokens = set(kpi["query"].lower().split())

        # --- Table-first search ---
        # Collect all candidates above threshold, score each, keep the best
        # extraction across the entire candidate set (rather than returning on
        # the first successful extraction, which may pick a less specific row).
        table_candidates: list[tuple[float, dict, int, list[str]]] = []
        for th in report["table_headers"]:
            sim = cosine_sim(query_emb, th["embedding"])
            if sim < TAU_TABLE:
                continue
            table = report["tables"][th["table_idx"]]
            eff_headers, _ = _effective_headers_and_rows(table)
            score = sim * _structural_score(eff_headers, report["report_year"])
            table_candidates.append((score, table, th["table_idx"], eff_headers))

        table_candidates.sort(key=lambda x: x[0], reverse=True)

        best_table_result: tuple[float, KPIExtraction] | None = None

        for table_sim, table, _, _ in table_candidates:
            headers, rows = _effective_headers_and_rows(table)
            year_col = _find_year_col(headers, report["report_year"])
            if year_col is None:
                continue

            best_row_idx = None
            best_row_score = 0.0
            for ri, row in enumerate(rows):
                if not row:
                    continue
                rs = _row_score(kpi["query"], query_tokens, row[0] or "")
                if rs > best_row_score:
                    best_row_score = rs
                    best_row_idx = ri

            if best_row_idx is None or best_row_score == 0.0:
                continue

            row = rows[best_row_idx]
            if year_col >= len(row):
                continue
            cell = row[year_col]
            num_match = re.search(_NUMBER_RE, cell)
            if not num_match:
                continue
            try:
                raw_value = parse_number(num_match.group(0))
            except ValueError:
                continue

            unit = _infer_unit_from_row_or_header(
                headers, row, year_col, unit_family_canonicals
            )
            if unit is None or unit not in unit_family_canonicals:
                continue

            try:
                canonical_value = to_canonical_value(
                    raw_value, unit, kpi["canonical_unit"]
                )
            except ValueError:
                continue

            lo, hi = kpi["plausible_range"]
            if not (lo <= canonical_value <= hi):
                log.debug(
                    "baseline: table candidate rejected out_of_range (kpi=%s, value=%s)",
                    kpi_key, canonical_value,
                )
                continue

            candidate_flags = list(flags)
            combined_score = table_sim * best_row_score
            if _YEAR_RE.fullmatch(cell.strip()):
                candidate_flags.append("year_shaped_value")
                combined_score *= 0.5

            candidate = KPIExtraction(
                company=report["company"],
                report_year=report["report_year"],
                kpi=kpi_key,
                value=canonical_value,
                unit=kpi["canonical_unit"],
                reporting_year=report["report_year"],
                source_snippet=f"table@page {table['page_num']}: {row[0]} | {cell}",
                source_page=table["page_num"],
                confidence=combined_score,
                extractor=self.name,
                flags=candidate_flags,
            )
            if best_table_result is None or combined_score > best_table_result[0]:
                best_table_result = (combined_score, candidate)

        if best_table_result is not None:
            return best_table_result[1]

        # --- Sentence fallback ---
        if not report["sentences"]:
            return KPIExtraction(
                company=report["company"], report_year=report["report_year"],
                kpi=kpi_key, value=None, unit=None, reporting_year=None,
                source_snippet=None, source_page=None, confidence=None,
                extractor=self.name, flags=flags,
            )

        sent_embs = np.stack([s["embedding"] for s in report["sentences"]])
        top_idxs = top_k(query_emb, sent_embs, k=TOP_K_SENTENCES)

        best: tuple[float, float, str, str, int] | None = None
        for idx in top_idxs:
            s = report["sentences"][idx]
            sim = cosine_sim(query_emb, s["embedding"])
            pv = parse_value(s["text"], kpi_unit_family=kpi["unit_family"])
            if pv is None:
                continue
            raw_value, unit = pv
            if unit not in unit_family_canonicals:
                continue
            try:
                canonical_value = to_canonical_value(
                    raw_value, unit, kpi["canonical_unit"]
                )
            except ValueError:
                continue
            lo, hi = kpi["plausible_range"]
            if not (lo <= canonical_value <= hi):
                if "out_of_range" not in flags:
                    flags.append("out_of_range")
                continue
            year_bonus = 0.1 if str(report["report_year"]) in s["text"] else 0.0
            score = sim + year_bonus
            if best is None or score > best[0]:
                best = (score, canonical_value, s["text"], unit, s["page_num"])

        if best is None:
            return KPIExtraction(
                company=report["company"], report_year=report["report_year"],
                kpi=kpi_key, value=None, unit=None, reporting_year=None,
                source_snippet=None, source_page=None, confidence=None,
                extractor=self.name, flags=flags,
            )

        score, canonical_value, sentence, unit, page = best
        return KPIExtraction(
            company=report["company"],
            report_year=report["report_year"],
            kpi=kpi_key,
            value=canonical_value,
            unit=kpi["canonical_unit"],
            reporting_year=report["report_year"],
            source_snippet=f"sentence@page {page}: {sentence}",
            source_page=page,
            confidence=float(score),
            extractor=self.name,
            flags=flags,
        )
