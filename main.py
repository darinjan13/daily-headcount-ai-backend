from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "AI Agent Running Locally"}


# -------- HEADER DETECTION --------
def find_header_row(df):
    first_row = df.iloc[0]
    non_empty = first_row.notna().sum()
    unique_ratio = first_row.nunique() / non_empty if non_empty > 0 else 0
    if non_empty >= 3 and unique_ratio > 0.8:
        return 0
    for i in range(1, len(df) - 5):
        current = df.iloc[i]
        next_rows = df.iloc[i+1:i+6]
        non_empty = current.notna().sum()
        if non_empty < 3:
            continue
        next_density = next_rows.notna().sum(axis=1).mean()
        if abs(next_density - non_empty) < 2:
            unique_ratio = current.nunique() / non_empty
            if unique_ratio > 0.8:
                return i
    return 0

def select_main_block(df):
    counts = df.notna().sum().tolist()
    active = [c >= 3 for c in counts]
    best_start = 0
    best_len = 0
    cur_start = None
    cur_len = 0
    for i, is_active in enumerate(active):
        if is_active:
            if cur_start is None:
                cur_start = i
                cur_len = 1
            else:
                cur_len += 1
        else:
            if cur_start is not None and cur_len > best_len:
                best_start, best_len = cur_start, cur_len
            cur_start = None
            cur_len = 0
    if cur_start is not None and cur_len > best_len:
        best_start, best_len = cur_start, cur_len
    return df.iloc[:, best_start:best_start + best_len]

def cut_after_empty_column(df):
    for i in range(len(df.columns)):
        if df.iloc[:, i].isna().all():
            return df.iloc[:, :i]
    return df

def cut_by_header_gap(df, header_row_index):
    header = df.iloc[header_row_index].tolist()
    for i, value in enumerate(header):
        if pd.isna(value):
            return df.iloc[:, :i]
    return df

import datetime

def is_date_value(val) -> bool:
    """Check if a value is a date/datetime object or a parseable date string."""
    if isinstance(val, (datetime.datetime, datetime.date, pd.Timestamp)):
        return True
    if isinstance(val, str):
        s = val.strip()
        if len(s) < 6: return False
        try:
            pd.to_datetime(s, errors="raise")
            return True
        except:
            return False
    return False

def detect_wide_format(df) -> dict:
    """
    Detect if the DataFrame has date columns as headers (wide format).
    Returns info about which columns are date-columns vs id-columns.
    """
    if df.empty or len(df.columns) < 4:
        return {"is_wide": False}

    date_cols = []
    id_cols = []

    for col in df.columns:
        if is_date_value(col):
            date_cols.append(col)
        else:
            id_cols.append(col)

    total_cols = len(df.columns)
    date_ratio = len(date_cols) / total_cols if total_cols > 0 else 0

    # Wide format: at least 3 date columns AND they make up >30% of columns
    if len(date_cols) >= 3 and date_ratio >= 0.3:
        return {
            "is_wide": True,
            "date_cols": date_cols,
            "id_cols": id_cols,
            "date_ratio": date_ratio,
        }

    return {"is_wide": False}

def melt_wide_format(df, date_cols, id_cols) -> pd.DataFrame:
    """
    Unpivot a wide-format date table into long format.
    e.g. NAME | Sep-16 | Sep-17  →  NAME | Date | Value
    Filters out non-data id columns (total, capacity, etc.) and keeps
    only the core id columns + melted date + value.
    """
    # Filter id_cols: drop ones that look like summary/total columns
    summary_keywords = ["total", "capacity", "link", "template", "ks/hr", "ks/day", "http"]
    clean_id_cols = []
    for col in id_cols:
        col_str = str(col).lower().strip()
        if any(kw in col_str for kw in summary_keywords):
            continue
        # Skip unnamed/auto-generated columns
        if col_str.startswith("column_") or col_str == "nan":
            continue
        # Skip columns that are mostly empty
        non_null = df[col].notna().sum()
        if non_null < len(df) * 0.1:
            continue
        clean_id_cols.append(col)

    # Format date column headers as clean date strings
    rename_map = {}
    for col in date_cols:
        if isinstance(col, (datetime.datetime, pd.Timestamp)):
            rename_map[col] = col.strftime("%Y-%m-%d")
        else:
            try:
                rename_map[col] = pd.to_datetime(str(col)).strftime("%Y-%m-%d")
            except:
                rename_map[col] = str(col)

    df = df.rename(columns=rename_map)
    renamed_date_cols = [rename_map.get(c, c) for c in date_cols]

    # Only keep rows that have at least one non-null date value
    df = df[df[renamed_date_cols].notna().any(axis=1)]

    # Melt
    melted = df[clean_id_cols + renamed_date_cols].melt(
        id_vars=clean_id_cols,
        value_vars=renamed_date_cols,
        var_name="Date",
        value_name="Value",
    )

    # Drop rows with null values (sparse wide tables have lots of NaN)
    melted = melted.dropna(subset=["Value"])
    melted["Value"] = pd.to_numeric(melted["Value"], errors="coerce")
    melted = melted.dropna(subset=["Value"])

    return melted.reset_index(drop=True)


# -------- EXTRACT RAW TABLE --------
@app.post("/extract-raw-table")
async def extract_raw_table(file: UploadFile = File(...), sheet_name: str = None):
    contents = await file.read()
    excel = pd.ExcelFile(io.BytesIO(contents))
    if sheet_name not in excel.sheet_names:
        sheet_name = excel.sheet_names[0]
    df = pd.read_excel(excel, sheet_name=sheet_name, header=None)
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
        lambda row: row.str.contains("Grand Total|Sum of", case=False).any(),
        axis=1
    )]
    df = cut_after_empty_column(df)
    df = df.replace([np.inf, -np.inf], np.nan)

    # -------- WIDE FORMAT DETECTION & AUTO-MELT --------
    wide_info = detect_wide_format(df)
    table_format = "long"
    if wide_info["is_wide"]:
        table_format = "wide"
        df = melt_wide_format(df, wide_info["date_cols"], wide_info["id_cols"])

    records = df.to_dict(orient="records")
    cleaned_rows = []
    for row in records:
        clean_row = {}
        for key, value in row.items():
            if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                clean_row[key] = None
            elif isinstance(value, (datetime.datetime, datetime.date, pd.Timestamp)):
                clean_row[key] = str(value)[:10]  # format as YYYY-MM-DD string
            else:
                clean_row[key] = value
        cleaned_rows.append(clean_row)

    return {
        "headers": list(df.columns),
        "rows": [[row.get(col) for col in df.columns] for row in cleaned_rows],
        "rowCount": len(cleaned_rows),
        "columnCount": len(df.columns),
        "tableFormat": table_format,  # "wide" or "long" — frontend can use this
        "wasTransformed": wide_info["is_wide"],
        "transformNote": "Wide format detected: date columns were unpivoted into rows." if wide_info["is_wide"] else None,
    }


# -------- GET SHEET NAMES --------
@app.post("/get-sheets")
async def get_sheets(file: UploadFile = File(...)):
    contents = await file.read()
    excel = pd.ExcelFile(io.BytesIO(contents))
    return {"sheets": excel.sheet_names}


from typing import Any, Dict, List, Optional
from fastapi import Body

def rows_to_objects(headers, rows):
    return [{h: (r[i] if i < len(r) else None) for i, h in enumerate(headers)} for r in rows]

def is_number(x):
    if x is None: return False
    if isinstance(x, (int, float)) and not (isinstance(x, float) and (np.isnan(x) or np.isinf(x))): return True
    if isinstance(x, str):
        s = x.strip().replace(",", "").replace("$", "").replace("%", "")
        if s == "": return False
        try:
            float(s)
            return True
        except:
            return False
    return False

def try_parse_date(x):
    if x is None: return False
    if isinstance(x, pd.Timestamp): return True
    if isinstance(x, str):
        s = x.strip()
        if s == "": return False
        try:
            pd.to_datetime(s, errors="raise")
            return True
        except:
            return False
    return False

def detect_format_hint(col: str, values: list) -> str:
    """Guess if a numeric column is currency, percentage, or plain number."""
    name = col.lower()
    if any(k in name for k in ["amount", "cost", "revenue", "sales", "price", "budget", "expense", "income", "payment"]):
        return "currency"
    if any(k in name for k in ["rate", "ratio", "pct", "percent", "%", "margin", "utilization"]):
        return "percent"
    # Check if values look like percentages (0-1 range)
    nums = [float(str(v).replace(",", "")) for v in values if is_number(v)]
    if nums and all(0 <= n <= 1 for n in nums[:20]):
        return "percent_decimal"
    return "number"

def detect_column_profile(data, sample_size=50):
    if not data: return {}
    sample = data[:min(sample_size, len(data))]
    cols = list(sample[0].keys())
    profile = {}
    for col in cols:
        values = [row.get(col) for row in sample]
        non_null = [v for v in values if v is not None and str(v).strip() != ""]
        n = len(non_null)
        if n == 0:
            profile[col] = {"type": "empty", "nonNull": 0, "unique": 0}
            continue
        num_count = sum(1 for v in non_null if is_number(v))
        date_count = sum(1 for v in non_null if try_parse_date(v))
        unique_count = len(set(str(v).strip().lower() for v in non_null))
        num_ratio = num_count / n
        date_ratio = date_count / n
        unique_ratio = unique_count / n
        if date_ratio >= 0.7:
            ctype = "date"
        elif num_ratio >= 0.7:
            ctype = "numeric"
        else:
            ctype = "identifier" if unique_ratio >= 0.9 else "category"

        entry = {
            "type": ctype,
            "nonNull": n,
            "unique": unique_count,
            "uniqueRatio": round(unique_ratio, 3),
            "numericRatio": round(num_ratio, 3),
            "dateRatio": round(date_ratio, 3),
        }
        if ctype == "numeric":
            entry["formatHint"] = detect_format_hint(col, non_null)
        profile[col] = entry
    return profile

INDEX_LIKE_COLS = {"no", "no.", "num", "number", "id", "seq", "sequence", "index", "row", "sr", "sr.", "s/n", "#"}

def is_index_column(col: str, profile: dict) -> bool:
    name = col.strip().lower()
    if name in INDEX_LIKE_COLS:
        return True
    # Sequential integers with near-100% uniqueness = row index
    p = profile.get(col, {})
    if p.get("uniqueRatio", 0) >= 0.95 and p.get("numericRatio", 0) >= 0.95:
        return True
    return False

def score_measure(col, profile):
    if is_index_column(col, profile):
        return -999  # Never use index columns as measures
    p = profile[col]
    name = col.lower()
    score = p["nonNull"] / 10
    for k in ["total", "amount", "asset", "revenue", "sales", "cost", "count", "hours", "production", "kwh", "budget"]:
        if k in name: score += 5
    return score

def score_dimension(col, profile):
    p = profile[col]
    uniq = p["unique"]
    if uniq < 2: return -1
    score = 0
    name = col.lower()
    for k in ["name", "site", "team", "category", "type", "region", "department", "language", "status", "group"]:
        if k in name: score += 4
    if 2 <= uniq <= 30: score += 5
    elif 31 <= uniq <= 100: score += 2
    else: score -= 2
    return score

def pick_measures(profile, max_measures=3):
    cols = [c for c, p in profile.items() if p["type"] == "numeric"]
    scored = sorted(cols, key=lambda c: score_measure(c, profile), reverse=True)
    return scored[:max_measures]

def pick_dimensions(profile, max_dims=3):
    cols = [c for c, p in profile.items() if p["type"] in ["category", "identifier"]]
    scored = [(score_dimension(c, profile), c) for c in cols]
    scored = [(s, c) for s, c in scored if s >= 0]
    scored.sort(reverse=True)
    return [c for _, c in scored[:max_dims]]

def pick_date_col(profile):
    dates = [c for c, p in profile.items() if p["type"] == "date"]
    if not dates: return None
    scored = sorted(dates, key=lambda c: (5 if any(k in c.lower() for k in ["date", "day", "month", "year"]) else 0), reverse=True)
    return scored[0]

def build_filters(profile, max_filters=6):
    filters = []
    for c, p in profile.items():
        if p["type"] == "category" and 2 <= p["unique"] <= 50:
            filters.append((p["unique"], c))
    filters.sort()
    return [{"column": c, "type": "dropdown"} for _, c in filters[:max_filters]]


@app.post("/generate-dashboard-blueprint")
async def generate_dashboard_blueprint(payload: Dict[str, Any] = Body(...)):
    headers = payload.get("headers", [])
    rows = payload.get("rows", [])
    data = rows_to_objects(headers, rows)
    profile = detect_column_profile(data)

    measures = pick_measures(profile)
    dimensions = pick_dimensions(profile)
    date_col = pick_date_col(profile)

    primary_measure = measures[0] if measures else None
    primary_dim = dimensions[0] if dimensions else None

    charts = []
    chart_id = 0

    # Bar charts: top dimensions × top measure
    for dim in dimensions:
        for measure in measures[:2]:
            charts.append({
                "id": f"chart_bar_{chart_id}",
                "type": "bar",
                "title": f"{measure} by {dim}",
                "x": dim,
                "y": measure,
                "aggregation": "sum",
                "topN": 15,
            })
            chart_id += 1

    # Line chart if date column exists
    if date_col and primary_measure:
        charts.append({
            "id": f"chart_line_{chart_id}",
            "type": "line",
            "title": f"{primary_measure} over time",
            "x": date_col,
            "y": primary_measure,
            "aggregation": "sum",
        })
        chart_id += 1

    # Donut chart for low-cardinality dimensions
    for dim in dimensions:
        uniq = profile[dim]["unique"]
        if 2 <= uniq <= 8 and primary_measure:
            charts.append({
                "id": f"chart_donut_{chart_id}",
                "type": "donut",
                "title": f"{primary_measure} share by {dim}",
                "x": dim,
                "y": primary_measure,
                "aggregation": "sum",
            })
            chart_id += 1

    # KPI cards
    cards = []
    for i, measure in enumerate(measures[:4]):
        hint = profile[measure].get("formatHint", "number")
        cards.append({"id": f"card_sum_{i}", "label": f"Total {measure}", "column": measure, "aggregation": "sum", "formatHint": hint})
        if i == 0:
            cards.append({"id": f"card_avg_{i}", "label": f"Avg {measure}", "column": measure, "aggregation": "avg", "formatHint": hint})

    # Suggested pivots: dim × second dim if available
    suggested_pivots = []
    if primary_dim and primary_measure:
        suggested_pivots.append({"rowDim": primary_dim, "colDim": dimensions[1] if len(dimensions) > 1 else None, "measure": primary_measure})
    if len(dimensions) > 1 and len(measures) > 1:
        suggested_pivots.append({"rowDim": dimensions[1], "colDim": None, "measure": measures[1]})

    return {
        "profile": profile,
        "suggested": {"measure": primary_measure, "dimension": primary_dim, "date": date_col},
        "suggestedPivots": suggested_pivots,
        "cards": cards,
        "filters": build_filters(profile),
        "charts": charts,
        "table": {"enabled": True, "defaultColumns": headers[:12], "pageSize": 25},
        "formatHints": {c: p.get("formatHint", "number") for c, p in profile.items() if p["type"] == "numeric"},
    }