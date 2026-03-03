import re
import pandas as pd
from typing import List, Dict, Optional

from .helpers import (
    df_to_payload, is_number, is_date_value, format_date_header
)


def detect_wide_format(df: pd.DataFrame) -> dict:
    if df.empty or len(df.columns) < 4:
        return {"is_wide": False}
    date_cols = [col for col in df.columns if is_date_value(col)]
    id_cols = [col for col in df.columns if not is_date_value(col)]
    date_ratio = len(date_cols) / len(df.columns)
    if len(date_cols) >= 3 and date_ratio >= 0.25:
        return {"is_wide": True, "date_cols": date_cols, "id_cols": id_cols}
    return {"is_wide": False}


def classify_id_columns(df: pd.DataFrame, id_cols: List) -> dict:
    summary_patterns = re.compile(
        r'\b(total|sum|avg|average|capacity|target|quota|budget|max|min|kpi|'
        r'rate|ratio|link|url|http|template|note|remark|comment)\b',
        re.IGNORECASE
    )
    meta_patterns = re.compile(r'https?://', re.IGNORECASE)
    core, summary, meta = [], [], []

    for col in id_cols:
        col_str = str(col).strip()
        if re.match(r'^Column_\d+$', col_str) or col_str.lower() in ("nan", "none", ""):
            meta.append(col)
            continue
        if meta_patterns.search(col_str):
            meta.append(col)
            continue
        if summary_patterns.search(col_str):
            summary.append(col)
            continue
        non_null_ratio = df[col].notna().sum() / max(len(df), 1)
        if non_null_ratio < 0.1:
            meta.append(col)
            continue
        core.append(col)

    return {"core": core, "summary": summary, "meta": meta}


def detect_section_rows(df: pd.DataFrame, core_cols: List) -> Dict[int, str]:
    if len(core_cols) < 2:
        return {}
    primary = core_cols[0]
    secondary_cols = core_cols[1:]
    section_map = {}
    skip_keywords = re.compile(
        r'\b(total|grand|sum|overall|subtotal)\b', re.IGNORECASE
    )
    for i, row in df.iterrows():
        pval = row.get(primary)
        if pd.isna(pval) or str(pval).strip() == "":
            continue
        name = str(pval).strip()
        if skip_keywords.search(name):
            continue
        all_secondary_empty = all(
            pd.isna(row.get(sc)) or str(row.get(sc, "")).strip() == ""
            for sc in secondary_cols
        )
        if all_secondary_empty:
            section_map[i] = name
    return section_map


def find_value_columns(df: pd.DataFrame, date_cols: List, summary_cols: List) -> List:
    value_keywords = re.compile(
        r'\b(total|sum|amount|production|output|hours|units|qty|quantity|'
        r'revenue|sales|cost|pay|earned|score)\b',
        re.IGNORECASE
    )
    result = []
    for col in summary_cols:
        col_str = str(col).strip()
        if not value_keywords.search(col_str):
            continue
        non_null = df[col].dropna()
        if len(non_null) < 3:
            continue
        numeric_count = sum(1 for v in non_null if is_number(v))
        if numeric_count / len(non_null) >= 0.7:
            result.append(col)
    return result


def build_wide_table(df: pd.DataFrame, wide_info: dict):
    date_cols = wide_info["date_cols"]
    id_cols = wide_info["id_cols"]

    classified = classify_id_columns(df, id_cols)
    core_cols = classified["core"]
    summary_cols = classified["summary"]
    value_cols_to_keep = find_value_columns(df, date_cols, summary_cols)

    rename_map = {col: format_date_header(col) for col in date_cols}
    seen = {}
    for orig, formatted in rename_map.items():
        if formatted in seen:
            seen[formatted] += 1
            rename_map[orig] = f"{formatted} ({seen[formatted]})"
        else:
            seen[formatted] = 1

    df = df.rename(columns=rename_map)
    renamed_date_cols = [rename_map[c] for c in date_cols]
    section_map = detect_section_rows(df, core_cols)

    current_section = None
    section_values = []
    for i, row in df.iterrows():
        if i in section_map:
            current_section = section_map[i]
            section_values.append(None)
        else:
            section_values.append(current_section)

    df = df.copy()
    has_sections = bool(section_map)
    if has_sections:
        df["Section"] = section_values

    df = df[~df.index.isin(set(section_map.keys()))]

    if core_cols:
        primary = core_cols[0]
        df = df[df[primary].notna()]
        df = df[~df[primary].astype(str).str.strip().str.lower().isin(
            ["total", "grand total", "sub total", "subtotal", "overall", "summary", ""]
        )]

    id_part = (["Section"] if has_sections else []) + core_cols
    final_cols = id_part + value_cols_to_keep + renamed_date_cols
    final_cols = [c for c in final_cols if c in df.columns]
    df = df[final_cols].reset_index(drop=True)

    value_col_name = value_cols_to_keep[0] if value_cols_to_keep else "Value"
    return df, renamed_date_cols, core_cols, value_col_name, has_sections


def build_analytics(
    df: pd.DataFrame,
    date_cols: List,
    core_cols: List,
    value_col_name: str,
    has_sections: bool,
) -> dict:
    available_date_cols = [c for c in date_cols if c in df.columns]
    primary_col = core_cols[0] if core_cols else None

    id_vars = [
        c for c in (["Section"] if has_sections else []) + core_cols
        if c in df.columns
    ]
    melted = df[id_vars + available_date_cols].melt(
        id_vars=id_vars,
        value_vars=available_date_cols,
        var_name="Period",
        value_name=value_col_name,
    )
    melted = melted.dropna(subset=[value_col_name])
    melted[value_col_name] = pd.to_numeric(melted[value_col_name], errors="coerce")
    melted = melted.dropna(subset=[value_col_name])

    period_order = {d: i for i, d in enumerate(available_date_cols)}
    period_totals = (
        melted.groupby("Period")[value_col_name].sum().reset_index()
    )
    period_totals["_ord"] = period_totals["Period"].map(period_order)
    period_totals = period_totals.sort_values("_ord").drop("_ord", axis=1)

    primary_totals = None
    if primary_col:
        primary_totals = (
            melted.groupby(primary_col)[value_col_name]
            .sum().reset_index()
            .sort_values(value_col_name, ascending=False)
        )

    section_totals = None
    if has_sections and "Section" in melted.columns:
        section_totals = (
            melted.groupby("Section")[value_col_name]
            .sum().reset_index()
            .sort_values(value_col_name, ascending=False)
        )

    return {
        "periodTotals": df_to_payload(period_totals),
        "primaryTotals": df_to_payload(primary_totals),
        "sectionTotals": df_to_payload(section_totals),
        "primaryCol": primary_col,
        "valueCol": value_col_name,
        "periodCol": "Period",
    }