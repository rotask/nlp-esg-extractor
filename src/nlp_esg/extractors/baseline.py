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
    best = None
    for i, h in enumerate(headers):
        m = _YEAR_RE.search(h or "")
        if m:
            year = int(m.group(0))
            if year == report_year:
                return i
            # fall back to most recent year found
            if best is None or year > best[1]:
                best = (i, year)
    return best[0] if best else None


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
        for token in re.findall(r"[A-Za-z0-9µ³²\-]+", headers[value_col] or ""):
            try:
                u = canonicalize_unit(token)
            except ValueError:
                continue
            if u in unit_family_canonicals:
                return u

    # 4. Check the KPI-row header (first cell) for a unit
    if row:
        for token in re.findall(r"[A-Za-z0-9µ³²\-]+", row[0] or ""):
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

        # --- Table-first search ---
        table_candidates: list[tuple[float, dict, int, list[str]]] = []
        for th in report["table_headers"]:
            sim = cosine_sim(query_emb, th["embedding"])
            if sim < TAU_TABLE:
                continue
            table = report["tables"][th["table_idx"]]
            score = sim * _structural_score(table["headers"], report["report_year"])
            table_candidates.append((score, table, th["table_idx"], table["headers"]))

        table_candidates.sort(key=lambda x: x[0], reverse=True)

        for score, table, _, headers in table_candidates:
            year_col = _find_year_col(headers, report["report_year"])
            if year_col is None:
                continue
            # Find the KPI row: row whose first cell semantically contains the KPI
            query_tokens = set(kpi["query"].lower().split())
            best_row_idx = None
            best_overlap = 0
            for ri, row in enumerate(table["rows"]):
                if not row:
                    continue
                row_tokens = set((row[0] or "").lower().split())
                overlap = len(query_tokens & row_tokens)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_row_idx = ri
            if best_row_idx is None:
                continue
            row = table["rows"][best_row_idx]
            if year_col >= len(row):
                continue
            cell = row[year_col]
            # Extract the number (strip any unit embedded in the cell)
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
            if unit is None:
                continue
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
                flags.append("out_of_range")
                log.debug("baseline: table candidate rejected out_of_range (kpi=%s, value=%s)", kpi_key, canonical_value)
                continue

            # Guard: a raw cell that is literally a 4-digit year is suspicious
            # even when it nominally lies in the plausible range. Flag it so
            # downstream review can distinguish "we extracted a value" from
            # "we extracted the year-column header copied into the cell".
            if _YEAR_RE.fullmatch(cell.strip()):
                flags.append("year_shaped_value")

            return KPIExtraction(
                company=report["company"],
                report_year=report["report_year"],
                kpi=kpi_key,
                value=canonical_value,
                unit=kpi["canonical_unit"],
                reporting_year=report["report_year"],
                source_snippet=f"table@page {table['page_num']}: {row[0]} | {cell}",
                source_page=table["page_num"],
                confidence=float(score),
                extractor=self.name,
                flags=flags,
            )

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
