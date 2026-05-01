"""Deterministic baseline extractor.

Two-stage pipeline. Table-first: rank table headers by cosine to the
KPI query (>= `TAU_TABLE`), select the year column with `_find_year_col`,
score candidate rows with a phrase + token-overlap metric, infer the
unit from the cell / 'Unit' column / header / row label, canonicalise
the value, and reject anything outside the KPI's plausible range or
matching a `negative_tokens` entry. On miss, falls back to a page-level
line scanner that uses the same `rank_pages_hybrid` retrieval and
preserves magnitude when picking the most-recent year column.
"""
from __future__ import annotations
import logging
import re
from typing import Any

from nlp_esg.config import KPIS, TAU_TABLE
from nlp_esg.extractors.base import Extractor
from nlp_esg.normalize import (
    _NUMBER_RE,
    canonicalize_unit,
    canonicalize_unit_robust,
    normalize_co2,
    parse_number,
    parse_value,
    to_canonical_value,
)
from nlp_esg.retrieval import cosine_sim, embed_texts, rank_pages_hybrid
from nlp_esg.types import KPIExtraction

log = logging.getLogger(__name__)

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _structural_score(headers: list[str], report_year: int) -> float:
    for h in headers:
        if _YEAR_RE.search(h or "") and str(report_year) in (h or ""):
            return 1.0
    return 0.5


def _find_year_col(headers: list[str], report_year: int) -> int | None:
    """Return the column index of the MOST RECENT year found in headers,
    capped at `report_year`.

    ESG reports typically show the current reporting year as the leftmost
    data column. The cap rejects milestone/target columns
    (Iberdrola tables include 2026/2040/2050 alongside actual 2024/2025) —
    without it, `_find_year_col` selected `'2050'` and every data cell in
    that column is `'N/AV.'` so the table-path silently dropped every
    candidate row.

    The convention is that the filename year matches the most recent data
    column: `bp_2025.pdf` is paired with the `2025` column in the table.
    """
    best: tuple[int, int] | None = None
    for i, h in enumerate(headers):
        m = _YEAR_RE.search(h or "")
        if m:
            year = int(m.group(0))
            if year > report_year:
                continue
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
) -> tuple[float, str] | None:
    """Look for a unit; return `(multiplier, canonical_unit)` or None.

    Multiplier accounts for magnitude prefixes ('million', 'thousand') so the
    caller multiplies the parsed cell value before canonical conversion.
    Multiplier is 1.0 for "plain" units like 'MWh' or 'tCO2e'.

    Search order (first match wins):
    1. The value cell itself (`"45,678 tCO2e"`)
    2. A `'Unit'`/`'Units'` column header
    3. row[1] as a fallback unit cell (Docling tables often place the unit in
       column 1 without labelling the header)
    4. The value column's own header (`"2024 (tCO2e)"` or `"million MWh.2025"`)
    5. The KPI-row label (first cell)
    6. Any other table header
    """
    def _match(s: str) -> tuple[float, str] | None:
        r = canonicalize_unit_robust(s)
        if r and r[1] in unit_family_canonicals:
            return r
        return None

    # 1. Check the value cell itself ("45,678 tCO2e")
    cell = row[value_col] if value_col < len(row) else ""
    pv = parse_value(cell, kpi_unit_family=list(unit_family_canonicals))
    if pv:
        return (1.0, pv[1])

    # 2. Look for a 'Unit' column
    for i, h in enumerate(headers):
        if (h or "").strip().lower() in ("unit", "units"):
            if i < len(row):
                m = _match(row[i] or "")
                if m:
                    return m

    # 3. Docling pattern: empty header but unit sits in row[1].
    # ESG tables routinely follow [Label, Unit, year_values...] without
    # actually labelling the second column 'Unit'.
    if len(row) > 1 and value_col != 1:
        m = _match(row[1] or "")
        if m:
            return m

    # 4. Check the value column's own header. Try the WHOLE header first
    # (so compound forms like 'million cubic metres.2025' or 'millionMWh.2025'
    # hit canonicalize_unit_robust's magnitude+year-suffix path), then fall
    # back to per-token scanning for embedded forms like '2024 (tCO2e)'.
    if value_col < len(headers):
        h = headers[value_col] or ""
        m = _match(h)
        if m:
            return m
        for token in re.findall(r"[A-Za-z0-9µ³²\-]+", normalize_co2(h)):
            m = _match(token)
            if m:
                return m

    # 5. Check the KPI-row header (first cell) for a unit
    if row:
        h0 = row[0] or ""
        m = _match(h0)
        if m:
            return m
        for token in re.findall(r"[A-Za-z0-9µ³²\-]+", normalize_co2(h0)):
            m = _match(token)
            if m:
                return m

    # 6. Check the table-level headers for a unit annotation
    for h in headers:
        ht = h or ""
        m = _match(ht)
        if m:
            return m
        for token in re.findall(r"[A-Za-z0-9µ³²\-]+", normalize_co2(ht)):
            m = _match(token)
            if m:
                return m

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

        # Generic column-header words that sometimes appear as row[0] (notably
        # in Iberdrola's [Metric, Description, Unit, year, year] schema where
        # every data row's first cell is the literal string 'Metric'). When
        # detected, score row[1] instead so the actual label is considered.
        _COLUMN_HEADER_ARTIFACTS = {"metric", "metrics", "indicator", "indicators",
                                    "description", "descriptions", "topic", "kpi"}

        # Collect ALL viable table candidates (one per table). Magnitude tiebreak
        # below picks among candidates whose combined_score is within 5% of the
        # best — preferring the LARGEST canonical value, which corresponds to
        # the operational-control / consolidated total in ESG datasheets.
        all_table_candidates: list[tuple[float, KPIExtraction]] = []

        # Build a page_num -> Markdown-heading-only string map for page-level
        # context checks. We restrict to lines starting with '#' so body-text
        # cross-references ('See page 422 for ... financial control
        # boundary') don't disqualify a page whose actual section heading is
        # innocent. Docling outputs page text with Markdown headings.
        page_heading_text: dict[int, str] = {}
        for p in report["pages"]:
            txt = p.get("text") or ""
            heads = [ln for ln in txt.split("\n") if ln.lstrip().startswith("#")]
            page_heading_text[p["page_num"]] = " ".join(heads).lower()
        page_negative_phrases = [p.lower() for p in kpi.get("page_negative_phrases", [])]

        for table_sim, table, _, _ in table_candidates:
            # Page-level rejection (Shell water 'financial control boundary'
            # heading lives in page text, not in any table row).
            heading_text_lc = page_heading_text.get(table["page_num"], "")
            if any(phrase in heading_text_lc for phrase in page_negative_phrases):
                continue

            headers, rows = _effective_headers_and_rows(table)
            year_col = _find_year_col(headers, report["report_year"])
            if year_col is None:
                continue

            negative_tokens = [t.lower() for t in kpi.get("negative_tokens", [])]
            current_section = ""  # propagated from preceding section-header rows
            scored_rows: list[tuple[float, int, str]] = []
            for ri, row in enumerate(rows):
                if not row:
                    continue
                if year_col >= len(row):
                    continue
                # Section-header rows have no number in the year column. Track
                # their row[0] as the current section so subsequent data rows
                # inherit it for negative-token filtering.
                if not re.search(_NUMBER_RE, row[year_col] or ""):
                    if (row[0] or "").strip():
                        current_section = row[0]
                    continue

                # Pick the label cell. Falls back to row[1] when row[0] is a
                # generic column-header artefact.
                label_cell = row[0] or ""
                if label_cell.strip().lower() in _COLUMN_HEADER_ARTIFACTS and len(row) > 1:
                    label_cell = row[1] or ""

                neg_haystack = (
                    current_section + " " + label_cell + " " +
                    (row[1] if len(row) > 1 else "")
                ).lower()
                if any(neg in neg_haystack for neg in negative_tokens):
                    continue
                rs = _row_score(kpi["query"], query_tokens, label_cell)
                if rs > 0:
                    scored_rows.append((rs, ri, label_cell))

            if not scored_rows:
                continue
            scored_rows.sort(key=lambda x: x[0], reverse=True)

            # Try each row in row-score order. The first one that passes unit
            # and plausible-range checks becomes this table's candidate.
            # Without this fall-through Eni page 166's gold row 'Direct GHG
            # emissions (Scope 1)' (rs=0.20, value 28.4 MtCO2e) was masked by
            # 'Percentage of Scope 1 ... emission trading system' (rs=0.30,
            # value 61 %, fails unit check), and the whole table got skipped.
            for best_row_score, best_row_idx, display_label in scored_rows:
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

                unit_match = _infer_unit_from_row_or_header(
                    headers, row, year_col, unit_family_canonicals
                )
                if unit_match is None:
                    continue
                multiplier, unit = unit_match
                if unit not in unit_family_canonicals:
                    continue

                try:
                    canonical_value = to_canonical_value(
                        raw_value * multiplier, unit, kpi["canonical_unit"]
                    )
                except ValueError:
                    continue

                lo, hi = kpi["plausible_range"]
                if not (lo <= canonical_value <= hi):
                    log.debug(
                        "baseline: table-row candidate rejected out_of_range (kpi=%s, value=%s)",
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
                    source_snippet=f"table@page {table['page_num']}: {display_label} | {cell}",
                    source_page=table["page_num"],
                    confidence=combined_score,
                    extractor=self.name,
                    flags=candidate_flags,
                )
                all_table_candidates.append((combined_score, candidate))
                break  # one candidate per table

        if all_table_candidates:
            # Magnitude tiebreak: among candidates within 5% of the best
            # combined_score, prefer the LARGEST canonical value. This breaks
            # ties between equity-share vs operational-control sub-tables
            # toward the operational-control figure (BP scope_1, Shell scope_1).
            best_score = max(c[0] for c in all_table_candidates)
            tied = [c for c in all_table_candidates
                    if best_score == 0 or (best_score - c[0]) / best_score <= 0.05]
            best = max(tied, key=lambda c: c[1].value or 0.0) if len(tied) > 1 else max(
                all_table_candidates, key=lambda c: c[0]
            )
            return best[1]

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
        top_n_pages: int = 25,
        min_kw_score: float = 0.15,
    ) -> KPIExtraction | None:
        """Page-level line scanner.

        Returns the highest-scoring (line, value, unit) match across the top-N
        ranked pages, or None if nothing passes the filters.
        """
        queries = list(kpi.get("queries") or [kpi["query"]])
        ranked = rank_pages_hybrid(report, queries)
        top_pages = [pn for pn, _ in ranked[:top_n_pages]]
        page_rank = {pn: i for i, pn in enumerate(top_pages)}
        pages_by_num = {p["page_num"]: p for p in report["pages"]}

        kpi_tokens: set[str] = set()
        for q in queries:
            for tok in re.findall(r"[A-Za-z]+", q.lower()):
                if len(tok) > 3:
                    kpi_tokens.add(tok)

        negative_tokens = [t.lower() for t in kpi.get("negative_tokens", [])]
        lo, hi = kpi["plausible_range"]

        # Collect ALL candidates that pass filters, then resolve ties on the
        # consolidated-vs-segment ambiguity by preferring the larger canonical
        # value among candidates with similar kw_score. The classic case is
        # Eni scope_1, where the consolidated 28.4 MtCO2e and a segment-level
        # 18.6 MtCO2e share the same template; the only distinguishing signal
        # is magnitude.
        candidates: list[dict] = []
        for pn in top_pages:
            page = pages_by_num.get(pn)
            if page is None:
                continue
            text = normalize_co2(page["text"])
            page_lines = text.split("\n")
            for li, line in enumerate(page_lines):
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

                adjusted = self._pick_year_column_value(
                    page_lines, li, stripped, raw_value
                )
                if adjusted is not None:
                    raw_value = adjusted
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

                rank_bonus = 0.5 * (1.0 - page_rank[pn] / max(1, len(top_pages)))
                total_bonus = 0.3 if line_lower.lstrip("|").lstrip().startswith("total ") else 0.0
                score = kw_score + rank_bonus + total_bonus
                candidates.append({
                    "score": score, "kw_score": kw_score,
                    "canonical_value": canonical_value,
                    "snippet": stripped, "unit": unit, "page": pn,
                })

        if not candidates:
            return None

        # Pick the highest-scoring candidate. If multiple candidates have
        # kw_scores within 0.05 of the best (= same content template but
        # different pages/segments), prefer the LARGEST canonical value.
        # This correctly resolves consolidated-vs-segment lines for emissions
        # without affecting water/energy where year-column has already picked
        # the right within-line value.
        candidates.sort(key=lambda c: c["score"], reverse=True)
        best_kw = candidates[0]["kw_score"]
        kw_tied = [c for c in candidates if abs(c["kw_score"] - best_kw) < 0.05]
        if len(kw_tied) > 1:
            best = max(kw_tied, key=lambda c: c["canonical_value"])
        else:
            best = candidates[0]

        score = best["score"]
        canonical_value = best["canonical_value"]
        snippet = best["snippet"]
        unit = best["unit"]
        page = best["page"]
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

    @staticmethod
    def _pick_year_column_value(
        page_lines: list[str],
        line_idx: int,
        data_line: str,
        raw_value: float,
    ) -> float | None:
        """Adjust raw_value to the most-recent-year column when a year-row is
        nearby. Returns the magnitude-preserved value at that column, or None
        if no usable year-row context is found.

        Magnitude preservation: the data line may contain a magnitude word
        like 'million' that parse_value already factored into raw_value. We
        scale by the ratio (target_col_num / first_data_num) so the magnitude
        carries through, regardless of whether the unit is '(MWh)' or
        'million MWh'.

        Robustness:
        - Search ±5 lines above and ±2 below (year-rows are usually headers).
        - Year row must contain >= 2 sequential years in 2010-2030 with no gaps.
        - Data line numbers are sliced from the END (last N), since the noise
          digits ('1' in 'Scope 1', '2' in 'MtCO2e') sit BEFORE the data values.
        """
        from nlp_esg.normalize import parse_number

        year_re = re.compile(r"\b(20[1-3]\d)\b")
        # Search above first (year-rows are header rows), then below.
        # Range goes deep — long tables can have a header 20+ lines above the
        # last data row (BP datasheet has up to ~14 rows per sub-table).
        # Iteration order matters: closest above wins (the most-recent header
        # in a multi-table page applies to the data rows below it).
        offsets = [-d for d in range(1, 26)] + [1, 2, 3]
        for offset in offsets:
            idx = line_idx + offset
            if not (0 <= idx < len(page_lines)):
                continue
            ymatches = list(year_re.finditer(page_lines[idx]))
            if len(ymatches) < 2:
                continue
            years = [int(m.group(0)) for m in ymatches]
            # Sequential check: max - min must equal len - 1 (no gaps, no dupes)
            if max(years) - min(years) != len(years) - 1:
                continue
            n_years = len(years)
            most_recent = max(years)
            col_idx = years.index(most_recent)

            # Parse all numbers on the data line and slice the LAST n_years
            # (the noise digits sit before the data values in a tabular row).
            num_re = re.compile(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d+(?:[.,]\d+)?")
            parsed: list[float] = []
            for s in num_re.findall(data_line):
                try:
                    parsed.append(parse_number(s))
                except ValueError:
                    continue
            if len(parsed) < n_years:
                continue
            data_nums = parsed[-n_years:]
            first_num = data_nums[0]
            target_num = data_nums[col_idx]
            if first_num == 0:
                continue
            return raw_value * (target_num / first_num)
        return None
