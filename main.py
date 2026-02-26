from fastapi import FastAPI, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
import datetime
import io
import re

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
    return {"status": "Dashboard Agent Running"}


# ══════════════════════════════════════════════════════
# GENERIC HELPERS
# ══════════════════════════════════════════════════════

def clean_value(value):
    """Safely serialize any cell value to JSON-compatible type."""
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
    """Convert a DataFrame to {headers, rows} payload."""
    if df is None or df.empty:
        return None
    headers = list(df.columns)
    rows = [[clean_value(v) for v in row] for row in df.values.tolist()]
    return {"headers": headers, "rows": rows}

def rows_to_objects(headers: List[str], rows: List[List]) -> List[Dict]:
    return [{h: (r[i] if i < len(r) else None) for i, h in enumerate(headers)} for r in rows]

def is_number(x) -> bool:
    if x is None:
        return False
    if isinstance(x, bool):
        return False
    if isinstance(x, (int, float)) and not (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
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
    """Check if a value is or looks like a date."""
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
    """Format a date value as a clean short string for display."""
    try:
        ts = pd.Timestamp(val)
        # Use cross-platform date formatting
        day = ts.day
        return ts.strftime(f"%b {day}")  # "Sep 16" without zero-padding
    except Exception:
        return str(val)


# ══════════════════════════════════════════════════════
# TABLE EXTRACTION
# ══════════════════════════════════════════════════════

def find_header_row(df: pd.DataFrame) -> int:
    """Find the most likely header row index."""
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
    """Check if a row is a summary/total row that should be excluded."""
    total_keywords = {"total", "grand total", "sub total", "subtotal", "overall", "summary", "net total"}
    val = str(row.get(primary_col, "") or "").lower().strip()
    return val in total_keywords or val.startswith("total ") or val.endswith(" total")


# ══════════════════════════════════════════════════════
# WIDE FORMAT DETECTION & HANDLING
# ══════════════════════════════════════════════════════

def detect_wide_format(df: pd.DataFrame) -> dict:
    """
    Detect if the table uses dates as column headers (wide/pivot format).
    Works for any domain — payroll, attendance, sales by day, etc.
    """
    if df.empty or len(df.columns) < 4:
        return {"is_wide": False}

    date_cols = [col for col in df.columns if is_date_value(col)]
    id_cols = [col for col in df.columns if not is_date_value(col)]
    date_ratio = len(date_cols) / len(df.columns)

    if len(date_cols) >= 3 and date_ratio >= 0.25:
        return {"is_wide": True, "date_cols": date_cols, "id_cols": id_cols}
    return {"is_wide": False}

def classify_id_columns(df: pd.DataFrame, id_cols: List) -> dict:
    """
    Classify non-date columns into:
    - core: actual identity/dimension columns (name, id, category...)
    - summary: aggregated/computed columns (totals, averages, capacities)
    - meta: metadata/links/notes not useful for analysis
    """
    # Generic patterns for summary columns
    summary_patterns = re.compile(
        r'\b(total|sum|avg|average|capacity|target|quota|budget|max|min|kpi|rate|ratio|link|url|http|template|note|remark|comment)\b',
        re.IGNORECASE
    )
    meta_patterns = re.compile(r'https?://', re.IGNORECASE)

    core, summary, meta = [], [], []

    for col in id_cols:
        col_str = str(col).strip()

        # Auto-generated unnamed columns
        if re.match(r'^Column_\d+$', col_str) or col_str.lower() in ("nan", "none", ""):
            meta.append(col)
            continue

        # URL/link content in column name
        if meta_patterns.search(col_str):
            meta.append(col)
            continue

        # Summary keyword in name
        if summary_patterns.search(col_str):
            summary.append(col)
            continue

        # Mostly empty column
        non_null_ratio = df[col].notna().sum() / max(len(df), 1)
        if non_null_ratio < 0.1:
            meta.append(col)
            continue

        core.append(col)

    return {"core": core, "summary": summary, "meta": meta}

def detect_section_rows(df: pd.DataFrame, core_cols: List) -> Dict[int, str]:
    """
    Detect rows that act as group/section headers rather than data rows.
    These are rows where the primary ID column has a value but all secondary
    core columns are empty — generic for any wide table structure.
    """
    if len(core_cols) < 2:
        return {}

    primary = core_cols[0]
    # Use ALL other core cols to confirm emptiness (not just col 1)
    secondary_cols = core_cols[1:]

    section_map = {}
    skip_keywords = re.compile(r'\b(total|grand|sum|overall|subtotal)\b', re.IGNORECASE)

    for i, row in df.iterrows():
        pval = row.get(primary)
        if pd.isna(pval) or str(pval).strip() == "":
            continue
        name = str(pval).strip()
        if skip_keywords.search(name):
            continue
        # Check if ALL secondary core cols are empty
        all_secondary_empty = all(
            pd.isna(row.get(sc)) or str(row.get(sc, "")).strip() == ""
            for sc in secondary_cols
        )
        if all_secondary_empty:
            section_map[i] = name

    return section_map

def find_value_columns(df: pd.DataFrame, date_cols: List, summary_cols: List) -> List:
    """
    Among summary/non-date cols, find which ones look like aggregated values
    (e.g. Total Production, Total Hours, Amount Due) that are worth keeping in the table.
    These are numeric columns with decent fill rate.
    """
    value_keywords = re.compile(
        r'\b(total|sum|amount|production|output|hours|units|qty|quantity|revenue|sales|cost|pay|earned|score)\b',
        re.IGNORECASE
    )
    result = []
    for col in summary_cols:
        col_str = str(col).strip()
        if not value_keywords.search(col_str):
            continue
        # Must be mostly numeric
        non_null = df[col].dropna()
        if len(non_null) < 3:
            continue
        numeric_count = sum(1 for v in non_null if is_number(v))
        if numeric_count / len(non_null) >= 0.7:
            result.append(col)
    return result

def build_wide_table(df: pd.DataFrame, wide_info: dict):
    """
    Prepare the wide-format table for display:
    - Format date headers to readable short strings
    - Add a 'Section' column for group/department rows (if detected)
    - Remove section header rows and total rows
    - Keep only core ID cols + key value cols + date cols
    Returns: (clean_df, renamed_date_cols, core_id_cols, value_col_name)
    """
    date_cols = wide_info["date_cols"]
    id_cols = wide_info["id_cols"]

    classified = classify_id_columns(df, id_cols)
    core_cols = classified["core"]
    summary_cols = classified["summary"]

    # Find any aggregated value columns worth keeping
    value_cols_to_keep = find_value_columns(df, date_cols, summary_cols)

    # Format date column headers
    rename_map = {col: format_date_header(col) for col in date_cols}
    # Handle duplicate formatted names (e.g. two dates formatting to same string)
    seen = {}
    for orig, formatted in rename_map.items():
        if formatted in seen:
            seen[formatted] += 1
            rename_map[orig] = f"{formatted} ({seen[formatted]})"
        else:
            seen[formatted] = 1

    df = df.rename(columns=rename_map)
    renamed_date_cols = [rename_map[c] for c in date_cols]

    # Detect section/group header rows
    section_map = detect_section_rows(df, core_cols)

    # Assign section to each data row
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

    # Remove section header rows
    df = df[~df.index.isin(set(section_map.keys()))]

    # Remove total/summary rows
    if core_cols:
        primary = core_cols[0]
        df = df[df[primary].notna()]
        df = df[~df[primary].astype(str).str.strip().str.lower().isin(
            ["total", "grand total", "sub total", "subtotal", "overall", "summary", ""]
        )]

    # Build final column order
    id_part = (["Section"] if has_sections else []) + core_cols
    final_cols = id_part + value_cols_to_keep + renamed_date_cols
    final_cols = [c for c in final_cols if c in df.columns]
    df = df[final_cols].reset_index(drop=True)

    # Determine the best value column name for analytics labels
    value_col_name = value_cols_to_keep[0] if value_cols_to_keep else "Value"

    return df, renamed_date_cols, core_cols, value_col_name, has_sections

def build_analytics(df: pd.DataFrame, date_cols: List, core_cols: List, value_col_name: str, has_sections: bool) -> dict:
    """
    From the clean wide table, compute analytics tables:
    - periodTotals: sum per date column (for line chart)
    - primaryTotals: sum per primary identity (e.g. per person, per product)
    - sectionTotals: sum per section/group (if sections detected)
    All computed generically — no domain assumptions.
    """
    available_date_cols = [c for c in date_cols if c in df.columns]
    primary_col = core_cols[0] if core_cols else None

    # Melt to long format for aggregation
    id_vars = [c for c in (["Section"] if has_sections else []) + core_cols if c in df.columns]
    melted = df[id_vars + available_date_cols].melt(
        id_vars=id_vars,
        value_vars=available_date_cols,
        var_name="Period",
        value_name=value_col_name,
    )
    melted = melted.dropna(subset=[value_col_name])
    melted[value_col_name] = pd.to_numeric(melted[value_col_name], errors="coerce")
    melted = melted.dropna(subset=[value_col_name])

    # Period totals (preserve original date order)
    period_order = {d: i for i, d in enumerate(available_date_cols)}
    period_totals = (
        melted.groupby("Period")[value_col_name]
        .sum().reset_index()
    )
    period_totals["_ord"] = period_totals["Period"].map(period_order)
    period_totals = period_totals.sort_values("_ord").drop("_ord", axis=1)

    # Primary entity totals (per person / product / location etc.)
    primary_totals = None
    if primary_col:
        primary_totals = (
            melted.groupby(primary_col)[value_col_name]
            .sum().reset_index()
            .sort_values(value_col_name, ascending=False)
        )

    # Section totals
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


# ══════════════════════════════════════════════════════
# COLUMN PROFILING
# ══════════════════════════════════════════════════════

def try_parse_date(x) -> bool:
    if x is None: return False
    if isinstance(x, (pd.Timestamp, datetime.datetime, datetime.date)): return True
    if isinstance(x, str):
        s = x.strip()
        if not s or len(s) < 4: return False
        try:
            pd.to_datetime(s, errors="raise")
            return True
        except Exception:
            return False
    return False

def detect_format_hint(col: str, values: list) -> str:
    name = col.lower()
    if any(k in name for k in ["amount", "cost", "revenue", "sales", "price", "budget", "expense", "income", "payment", "wage", "salary", "pay"]):
        return "currency"
    if any(k in name for k in ["rate", "ratio", "pct", "percent", "%", "margin", "utilization", "efficiency"]):
        return "percent"
    nums = []
    for v in values[:30]:
        try:
            nums.append(float(str(v).replace(",", "")))
        except Exception:
            pass
    if nums and all(0 <= n <= 1 for n in nums):
        return "percent_decimal"
    return "number"

def detect_column_profile(data: List[Dict], sample_size: int = 100) -> dict:
    if not data: return {}
    sample = data[:min(sample_size, len(data))]
    profile = {}
    for col in sample[0].keys():
        values = [row.get(col) for row in sample]
        non_null = [v for v in values if v is not None and str(v).strip() not in ("", "None", "nan")]
        n = len(non_null)
        if n == 0:
            profile[col] = {"type": "empty", "nonNull": 0, "unique": 0}
            continue

        num_count = sum(1 for v in non_null if is_number(v))
        date_count = sum(1 for v in non_null if try_parse_date(v))
        unique_vals = set(str(v).strip().lower() for v in non_null)
        unique_count = len(unique_vals)
        num_ratio = num_count / n
        date_ratio = date_count / n
        unique_ratio = unique_count / n

        if date_ratio >= 0.7:
            ctype = "date"
        elif num_ratio >= 0.7:
            ctype = "numeric"
        elif unique_ratio >= 0.95:
            ctype = "identifier"
        else:
            ctype = "category"

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

# Columns that are almost certainly row indices, not real measures
INDEX_COL_NAMES = {"no", "no.", "num", "#", "sr", "sr.", "s/n", "seq", "sequence", "index", "row", "row no", "row no.", "sl", "sl."}

def is_index_column(col: str, profile: dict) -> bool:
    if col.strip().lower() in INDEX_COL_NAMES:
        return True
    p = profile.get(col, {})
    # High uniqueness + fully numeric + small values = likely sequential index
    if p.get("uniqueRatio", 0) >= 0.95 and p.get("numericRatio", 0) >= 0.95:
        return True
    return False

def score_measure(col: str, profile: dict) -> float:
    if is_index_column(col, profile):
        return -999
    p = profile[col]
    name = col.lower()
    score = p.get("nonNull", 0) / 10
    # Generic high-value measure keywords (no domain-specific terms)
    high_value = ["total", "amount", "value", "revenue", "sales", "cost", "count",
                  "hours", "units", "qty", "quantity", "budget", "output", "score", "sum"]
    for k in high_value:
        if k in name:
            score += 5
    return score

def score_dimension(col: str, profile: dict) -> float:
    p = profile[col]
    uniq = p.get("unique", 0)
    if uniq < 2:
        return -1
    score = 0
    name = col.lower()
    # Generic dimension keywords
    dim_keywords = ["name", "category", "type", "region", "site", "location", "team",
                    "department", "status", "group", "class", "division", "branch",
                    "product", "project", "client", "customer", "country", "city"]
    for k in dim_keywords:
        if k in name:
            score += 4
    # Sweet spot: enough variety to be interesting, not too many to be an identifier
    if 2 <= uniq <= 30:
        score += 5
    elif 31 <= uniq <= 100:
        score += 2
    elif uniq > 200:
        score -= 3
    return score

def pick_measures(profile: dict, max_n: int = 2) -> List[str]:
    cols = [c for c, p in profile.items() if p["type"] == "numeric"]
    return sorted(cols, key=lambda c: score_measure(c, profile), reverse=True)[:max_n]

def pick_dimensions(profile: dict, max_n: int = 2) -> List[str]:
    cols = [c for c, p in profile.items() if p["type"] in ("category", "identifier")]
    scored = [(score_dimension(c, profile), c) for c in cols]
    scored = [(s, c) for s, c in scored if s >= 0]
    scored.sort(reverse=True)
    return [c for _, c in scored[:max_n]]

def pick_date_col(profile: dict) -> Optional[str]:
    dates = [c for c, p in profile.items() if p["type"] == "date"]
    if not dates:
        return None
    priority = ["date", "day", "week", "month", "year", "period", "time"]
    return sorted(dates, key=lambda c: sum(3 for k in priority if k in c.lower()), reverse=True)[0]


# ══════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════

@app.post("/get-sheets")
async def get_sheets(file: UploadFile = File(...)):
    contents = await file.read()
    excel = pd.ExcelFile(io.BytesIO(contents))
    return {"sheets": excel.sheet_names}


@app.post("/extract-raw-table")
async def extract_raw_table(file: UploadFile = File(...), sheet_name: str = None):
    contents = await file.read()
    excel = pd.ExcelFile(io.BytesIO(contents))
    if sheet_name not in excel.sheet_names:
        sheet_name = excel.sheet_names[0]

    df = pd.read_excel(excel, sheet_name=sheet_name, header=None)
    df = df.dropna(how="all")

    # Skip single-cell title rows at the top
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
        lambda row: row.str.contains(r"Grand Total|Sum of", case=False, regex=True).any(), axis=1
    )]
    df = cut_after_empty_column(df)
    df = df.replace([np.inf, -np.inf], np.nan)

    # ── WIDE FORMAT ──────────────────────────────────
    wide_info = detect_wide_format(df)

    if wide_info["is_wide"]:
        df_wide, renamed_date_cols, core_cols, value_col_name, has_sections = build_wide_table(df, wide_info)
        analytics = build_analytics(df_wide, renamed_date_cols, core_cols, value_col_name, has_sections)

        return {
            **df_to_payload(df_wide),
            "tableFormat": "wide",
            "wasTransformed": True,
            "transformNote": f"Wide format detected: {len(renamed_date_cols)} date columns across {len(df_wide)} rows.",
            "analytics": analytics,
        }

    # ── LONG FORMAT ──────────────────────────────────
    records = df.to_dict(orient="records")
    rows_out = [[clean_value(row.get(col)) for col in df.columns] for row in records]

    return {
        "headers": list(df.columns),
        "rows": rows_out,
        "rowCount": len(rows_out),
        "columnCount": len(df.columns),
        "tableFormat": "long",
        "wasTransformed": False,
        "transformNote": None,
        "analytics": None,
    }


@app.post("/generate-dashboard-blueprint")
async def generate_dashboard_blueprint(payload: Dict[str, Any] = Body(...)):
    headers = payload.get("headers", [])
    rows = payload.get("rows", [])
    analytics = payload.get("analytics", None)
    table_format = payload.get("tableFormat", "long")

    data = rows_to_objects(headers, rows)
    profile = detect_column_profile(data)

    cards, charts, pivots = [], [], []

    # ── WIDE FORMAT: analytics already computed by backend ────────────
    if table_format == "wide" and analytics:
        primary_col = analytics.get("primaryCol")
        value_col = analytics.get("valueCol", "Value")
        period_col = analytics.get("periodCol", "Period")
        period_data = analytics.get("periodTotals")
        primary_data = analytics.get("primaryTotals")
        section_data = analytics.get("sectionTotals")

        # KPI cards — derived generically from primary totals
        if primary_data:
            objs = rows_to_objects(primary_data["headers"], primary_data["rows"])
            values = [float(r[value_col]) for r in objs if is_number(r.get(value_col))]
            if values:
                cards = [
                    {"id": "card_total", "label": f"Total {value_col}", "value": sum(values), "formatHint": detect_format_hint(value_col, values)},
                    {"id": "card_avg", "label": f"Avg {value_col} per {primary_col or 'Entity'}", "value": sum(values) / len(values), "formatHint": detect_format_hint(value_col, values)},
                    {"id": "card_count", "label": f"Total {primary_col or 'Entities'}", "value": len(values), "formatHint": "number"},
                ]

        # Line chart over periods
        if period_data and len(period_data["rows"]) > 1:
            charts.append({
                "id": "chart_line_period",
                "type": "line",
                "title": f"{value_col} over Time",
                "dataSource": "periodTotals",
                "x": period_col,
                "y": value_col,
            })

        # Pivot: by primary entity
        if primary_data:
            pivots.append({
                "id": "pivot_primary",
                "title": f"{value_col} by {primary_col or 'Entity'}",
                "dataSource": "primaryTotals",
            })

        # Pivot: by section (only if multiple sections exist)
        if section_data and len(section_data["rows"]) > 1:
            pivots.append({
                "id": "pivot_section",
                "title": f"{value_col} by Section",
                "dataSource": "sectionTotals",
            })

        return {
            "profile": profile,
            "cards": cards,
            "charts": charts,
            "pivots": pivots,
            "tableFormat": "wide",
            "analytics": analytics,
        }

    # ── LONG FORMAT ───────────────────────────────────────────────────
    measures = pick_measures(profile)
    dimensions = pick_dimensions(profile)
    date_col = pick_date_col(profile)

    primary_measure = measures[0] if measures else None
    primary_dim = dimensions[0] if dimensions else None

    # KPI cards
    for i, m in enumerate(measures):
        hint = profile[m].get("formatHint", "number")
        cards.append({"id": f"card_sum_{i}", "label": f"Total {m}", "column": m, "aggregation": "sum", "formatHint": hint})
    if primary_dim:
        cards.append({"id": "card_unique", "label": f"Unique {primary_dim}s", "column": primary_dim, "aggregation": "count", "formatHint": "number"})

    # Line chart if date + measure exist
    if date_col and primary_measure:
        charts.append({
            "id": "chart_line_0",
            "type": "line",
            "title": f"{primary_measure} over time",
            "x": date_col,
            "y": primary_measure,
        })

    # Up to 2 pivots
    for i, dim in enumerate(dimensions[:2]):
        if primary_measure:
            pivots.append({
                "id": f"pivot_{i}",
                "title": f"{primary_measure} by {dim}",
                "rowDim": dim,
                "colDim": None,
                "measure": primary_measure,
                "aggregation": "sum",
            })

    return {
        "profile": profile,
        "cards": cards,
        "charts": charts,
        "pivots": pivots,
        "tableFormat": "long",
        "analytics": None,
    }