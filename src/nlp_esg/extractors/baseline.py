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
from nlp_esg.retrieval import cosine_sim, embed_texts, rank_pages_hybrid, top_k
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

        # --- Page-level line-scanning fallback ---
        # The original sentence fallback failed because ESG data lines almost
        # never end in punctuation — the regex sentence splitter mangled them.
        # We instead rank pages with the hybrid BM25 + RRF ranker, then scan
        # each top page line by line for a (label-keyword, value, unit) hit.
        line_result = self._scan_lines_for_kpi(
            report, kpi, kpi_key, unit_family_canonicals, flags
        )
        if line_result is not None:
            return line_result

        return KPIExtraction(
            company=report["company"], report_year=report["report_year"],
            kpi=kpi_key, value=None, unit=None, reporting_year=None,
            source_snippet=None, source_page=None, confidence=None,
            extractor=self.name, flags=flags,
        )

    def _scan_lines_for_kpi(
        self,
        report: Any,
        kpi: dict,
        kpi_key: str,
        unit_family_canonicals: set[str],
        flags: list[str],
        top_n_pages: int = 8,
        min_kw_score: float = 0.15,
    ) -> KPIExtraction | None:
        """Page-level line scanner.

        Returns the highest-scoring (line, value, unit) match across the top-N
        ranked pages, or None if nothing passes the filters.
        """
        queries = list(kpi.get("queries") or [kpi["query"]])
        ranked = rank_pages_hybrid(report, queries)
        top_pages = [pn for pn, _ in ranked[:top_n_pages]]
        pages_by_num = {p["page_num"]: p for p in report["pages"]}

        # KPI relevance vocabulary: keywords from the query strings, length > 3.
        kpi_tokens: set[str] = set()
        for q in queries:
            for tok in re.findall(r"[A-Za-z]+", q.lower()):
                if len(tok) > 3:
                    kpi_tokens.add(tok)

        negative_tokens = [t.lower() for t in kpi.get("negative_tokens", [])]
        lo, hi = kpi["plausible_range"]

        best: tuple[float, float, str, str, int] | None = None
        for pn in top_pages:
            page = pages_by_num.get(pn)
            if page is None:
                continue
            text = normalize_co2(page["text"])
            for line in text.split("\n"):
                stripped = line.strip()
                if len(stripped) < 10:
                    continue
                line_lower = stripped.lower()
                if any(neg in line_lower for neg in negative_tokens):
                    continue
                kw_hits = sum(1 for t in kpi_tokens if t in line_lower)
                if kw_hits == 0:
                    continue
                kw_score = kw_hits / max(1, len(kpi_tokens))
                if kw_score < min_kw_score:
                    continue
                pv = parse_value(stripped, kpi_unit_family=kpi["unit_family"])
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
                if not (lo <= canonical_value <= hi):
                    if "out_of_range" not in flags:
                        flags.append("out_of_range")
                    continue
                if best is None or kw_score > best[0]:
                    best = (kw_score, canonical_value, stripped, unit, pn)

        if best is None:
            return None

        score, canonical_value, snippet, unit, page = best
        return KPIExtraction(
            company=report["company"],
            report_year=report["report_year"],
            kpi=kpi_key,
            value=canonical_value,
            unit=kpi["canonical_unit"],
            reporting_year=report["report_year"],
            source_snippet=f"line@page {page}: {snippet[:160]}",
            source_page=page,
            confidence=float(score),
            extractor=self.name,
            flags=flags,
        )
