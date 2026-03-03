import re
import io
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

from .helpers import (
    clean_value, df_to_payload, is_number,
    is_date_value, format_date_header
)


def find_header_row(df: pd.DataFrame) -> int:
    first_row = df.iloc[0]
    non_empty = first_row.notna().sum()
    unique_ratio = first_row.nunique() / non_empty if non_empty > 0 else 0
    if non_empty >= 3 and unique_ratio > 0.8:
        return 0
    for i in range(1, min(len(df) - 5, 20)):
        current = df.iloc[i]
        next_rows = df.iloc[i + 1:i + 6]
        non_empty = current.notna().sum()
        if non_empty < 3:
            continue
        next_density = next_rows.notna().sum(axis=1).mean()
        if abs(next_density - non_empty) < 2:
            unique_ratio = current.nunique() / non_empty
            if unique_ratio > 0.8:
                return i
    return 0


def cut_after_empty_column(df: pd.DataFrame) -> pd.DataFrame:
    for i in range(len(df.columns)):
        if df.iloc[:, i].isna().all():
            return df.iloc[:, :i]
    return df


def cut_by_header_gap(df: pd.DataFrame, header_row_index: int) -> pd.DataFrame:
    header = df.iloc[header_row_index].tolist()
    for i, value in enumerate(header):
        if pd.isna(value):
            return df.iloc[:, :i]
    return df


def is_total_row(row: pd.Series, primary_col: str) -> bool:
    total_keywords = {
        "total", "grand total", "sub total", "subtotal",
        "overall", "summary", "net total"
    }
    val = str(row.get(primary_col, "") or "").lower().strip()
    return val in total_keywords or val.startswith("total ") or val.endswith(" total")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all standard cleaning steps to a raw DataFrame."""
    df = df.dropna(how="all")

    first_row = df.iloc[0]
    non_empty = first_row.notna().sum()
    unique_ratio = first_row.nunique() / non_empty if non_empty > 0 else 0
    if non_empty <= 2 or unique_ratio < 0.5:
        df = df.iloc[1:].reset_index(drop=True)

    header_row_index = find_header_row(df)
    df = cut_by_header_gap(df, header_row_index)
    df.columns = df.iloc[header_row_index]
    df = df.iloc[header_row_index + 1:].reset_index(drop=True)
    df.columns = [
        str(col).strip() if pd.notna(col) else f"Column_{i}"
        for i, col in enumerate(df.columns)
    ]
    df = df.dropna(axis=1, how="all")
    df = df[~df.astype(str).apply(
        lambda row: row.str.contains(
            r"Grand Total|Sum of", case=False, regex=True
        ).any(),
        axis=1,
    )]
    df = cut_after_empty_column(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def extract_from_bytes(contents: bytes, sheet_name: str = None) -> dict:
    """
    Core extraction function.
    Takes raw xlsx bytes + optional sheet name.
    Returns full table payload ready for blueprint generation.
    Also returns allSheets for sheet switching.
    """
    excel = pd.ExcelFile(io.BytesIO(contents))
    all_sheets = excel.sheet_names

    if not sheet_name or sheet_name not in all_sheets:
        sheet_name = all_sheets[0]

    df = pd.read_excel(excel, sheet_name=sheet_name, header=None)
    df = clean_dataframe(df)

    from .analytics import detect_wide_format, build_wide_table, build_analytics

    wide_info = detect_wide_format(df)

    if wide_info["is_wide"]:
        df_wide, renamed_date_cols, core_cols, value_col_name, has_sections = build_wide_table(df, wide_info)
        analytics = build_analytics(df_wide, renamed_date_cols, core_cols, value_col_name, has_sections)
        _wide_payload = df_to_payload(df_wide)
        table_data = {
            **_wide_payload,
            "rowCount": len(_wide_payload["rows"]),
            "columnCount": len(_wide_payload["headers"]),
            "dateCols": renamed_date_cols,
            "tableFormat": "wide",
            "wasTransformed": True,
            "transformNote": f"Wide format detected: {len(renamed_date_cols)} date columns across {len(df_wide)} rows.",
            "analytics": analytics,
        }
    else:
        records = df.to_dict(orient="records")
        rows_out = [[clean_value(row.get(col)) for col in df.columns] for row in records]
        table_data = {
            "headers": list(df.columns),
            "rows": rows_out,
            "rowCount": len(rows_out),
            "columnCount": len(df.columns),
            "dateCols": [],
            "tableFormat": "long",
            "wasTransformed": False,
            "transformNote": None,
            "analytics": None,
        }

    return table_data, all_sheets, sheet_name