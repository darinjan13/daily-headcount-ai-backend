import datetime
import numpy as np
import pandas as pd
from typing import List, Dict


def clean_value(value):
    if value is None:
        return None
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, (datetime.datetime, datetime.date, pd.Timestamp)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def df_to_payload(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return None
    headers = list(df.columns)
    rows = [[clean_value(v) for v in row] for row in df.values.tolist()]
    return {"headers": headers, "rows": rows}


def rows_to_objects(headers: List[str], rows: List[List]) -> List[Dict]:
    return [
        {h: (r[i] if i < len(r) else None) for i, h in enumerate(headers)}
        for r in rows
    ]


def is_number(x) -> bool:
    if x is None:
        return False
    if isinstance(x, bool):
        return False
    if isinstance(x, (int, float)) and not (
        isinstance(x, float) and (np.isnan(x) or np.isinf(x))
    ):
        return True
    if isinstance(x, str):
        s = x.strip().replace(",", "").replace("$", "").replace("%", "").replace(" ", "")
        if not s:
            return False
        try:
            float(s)
            return True
        except ValueError:
            return False
    return False


def is_date_value(val) -> bool:
    if isinstance(val, (datetime.datetime, datetime.date, pd.Timestamp)):
        return True
    if isinstance(val, str):
        s = val.strip()
        if len(s) < 4:
            return False
        try:
            pd.to_datetime(s, errors="raise")
            return True
        except Exception:
            return False
    return False


def format_date_header(val) -> str:
    try:
        ts = pd.Timestamp(val)
        day = ts.day
        return ts.strftime(f"%b {day}")
    except Exception:
        return str(val)