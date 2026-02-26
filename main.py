from fastapi import FastAPI, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
import datetime
import io
import re
import json
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

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
        day = ts.day
        return ts.strftime(f"%b {day}")
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
        r'\b(total|sum|avg|average|capacity|target|quota|budget|max|min|kpi|rate|ratio|link|url|http|template|note|remark|comment)\b',
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
    skip_keywords = re.compile(r'\b(total|grand|sum|overall|subtotal)\b', re.IGNORECASE)

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
        r'\b(total|sum|amount|production|output|hours|units|qty|quantity|revenue|sales|cost|pay|earned|score)\b',
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

def build_analytics(df: pd.DataFrame, date_cols: List, core_cols: List, value_col_name: str, has_sections: bool) -> dict:
    available_date_cols = [c for c in date_cols if c in df.columns]
    primary_col = core_cols[0] if core_cols else None

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

    period_order = {d: i for i, d in enumerate(available_date_cols)}
    period_totals = (
        melted.groupby("Period")[value_col_name]
        .sum().reset_index()
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


# ══════════════════════════════════════════════════════
# COLUMN PROFILING  (kept for fallback + AI context)
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

        # For numeric columns, add distribution stats to help AI understand the data
        if ctype == "numeric":
            nums = []
            for v in non_null:
                try:
                    nums.append(float(str(v).replace(",", "")))
                except Exception:
                    pass
            if nums:
                entry["min"] = round(min(nums), 2)
                entry["max"] = round(max(nums), 2)
                entry["mean"] = round(sum(nums) / len(nums), 2)
                entry["formatHint"] = detect_format_hint(col, non_null)
                # Detect if this looks like a transaction quantity (mostly 1s)
                ones_ratio = sum(1 for v in nums if v == 1) / len(nums)
                if ones_ratio >= 0.7:
                    entry["grain"] = "transaction_flag"  # each row = 1 item, should SUM across rows
                elif ones_ratio <= 0.1 and entry["max"] > entry["mean"] * 5:
                    entry["grain"] = "quantity"  # variable quantities per row

        profile[col] = entry
    return profile

INDEX_COL_NAMES = {"no", "no.", "num", "#", "sr", "sr.", "s/n", "seq", "sequence", "index", "row", "row no", "row no.", "sl", "sl."}

def is_index_column(col: str, profile: dict) -> bool:
    if col.strip().lower() in INDEX_COL_NAMES:
        return True
    p = profile.get(col, {})
    if p.get("uniqueRatio", 0) >= 0.95 and p.get("numericRatio", 0) >= 0.95:
        return True
    return False

def score_measure(col: str, profile: dict) -> float:
    if is_index_column(col, profile):
        return -999
    p = profile[col]
    name = col.lower()
    score = p.get("nonNull", 0) / 10
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
    dim_keywords = ["name", "category", "type", "region", "site", "location", "team",
                    "department", "status", "group", "class", "division", "branch",
                    "product", "project", "client", "customer", "country", "city"]
    for k in dim_keywords:
        if k in name:
            score += 4
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
# AI AGENT: BLUEPRINT GENERATION
# ══════════════════════════════════════════════════════

def build_ai_prompt(headers: List[str], sample_rows: List[Dict], profile: dict, table_format: str) -> str:
    """
    Build a flexible prompt for Gemini to analyze any dataset and return
    a rich dashboard blueprint.
    """
    # Format sample rows as a readable mini-table
    sample_str = ""
    for i, row in enumerate(sample_rows[:15]):
        row_parts = [f"{k}: {repr(v)}" for k, v in row.items()]
        sample_str += f"  Row {i+1}: {{ {', '.join(row_parts)} }}\n"

    # Rich profile summary with stats and sample values
    profile_summary = []
    for col, p in profile.items():
        sample_vals = list({str(r.get(col)) for r in sample_rows[:30] if r.get(col) is not None})[:5]
        desc = (
            f"  - {col!r}: type={p['type']}, unique={p.get('unique', '?')}, "
            f"fill={p.get('nonNull', '?')} rows"
        )
        if p.get("formatHint"):
            desc += f", format={p['formatHint']}"
        if p.get("min") is not None:
            desc += f", min={p['min']}, max={p['max']}, mean={p['mean']}"
        if p.get("grain"):
            desc += f", grain={p['grain']}"
            if p["grain"] == "transaction_flag":
                desc += " ⚠️ mostly-1s column: SUM across rows to get meaningful totals"
        if sample_vals:
            desc += f", samples={sample_vals}"
        profile_summary.append(desc)

    # Separate columns by detected type for easier AI reasoning
    numeric_cols = [c for c, p in profile.items() if p["type"] == "numeric"]
    date_cols = [c for c, p in profile.items() if p["type"] == "date"]
    category_cols = [c for c, p in profile.items() if p["type"] == "category"]
    identifier_cols = [c for c, p in profile.items() if p["type"] == "identifier"]

    return f"""You are an expert data analyst. Your job is to analyze a dataset and produce a dashboard blueprint.

════════════════════════════════════════
DATASET OVERVIEW
════════════════════════════════════════
- Table format: {table_format}
- Total columns: {len(headers)}
- All column names: {headers}

PRE-CLASSIFIED COLUMNS (based on data profiling):
  Numeric columns  : {numeric_cols}
  Date columns     : {date_cols}
  Category columns : {category_cols}
  Identifier cols  : {identifier_cols}

DETAILED COLUMN PROFILES:
{chr(10).join(profile_summary)}

SAMPLE DATA (up to 15 rows):
{sample_str}

════════════════════════════════════════
YOUR ANALYSIS TASK
════════════════════════════════════════
Step 1 — Understand the dataset:
  - What domain is this? (HR, sales, production, finance, logistics, etc.)
  - What is the grain of each row? (one transaction? one employee? one day?)
  - What is the user most likely trying to track or measure?

Step 2 — Decide on KPI cards (3–5 cards):
  - Pick the most important numeric columns to SUM (totals, quantities, amounts)
  - Pick key category columns to COUNT DISTINCT (how many sites, people, products, etc.)
  - Pick averages only when a rate/score makes sense
  - NEVER use row number / serial / index columns

Step 3 — Decide on charts (1–3 charts):
  - If a DATE column exists → always include a LINE chart showing a numeric measure over time
  - If a good CATEGORY column exists (2–20 unique values) → include a BAR chart of top values
  - If share/composition matters → include a DONUT chart
  - Chart types available: "line", "bar", "donut"
  - For line charts: x must be a DATE column, y must be a NUMERIC column
  - For bar/donut charts: x is a CATEGORY column, y is a NUMERIC column

Step 4 — Decide on pivot tables (1–3 pivots):
  - Pick the most insightful dimension × measure combinations
  - If two good category columns exist, use colDim for a cross-tab (e.g. Site × Language)
  - Use aggregation="count" when the measure column is categorical or when counting rows makes sense
  - Use aggregation="sum" for numeric totals
  - Use aggregation="avg" for rates and scores

════════════════════════════════════════
OUTPUT FORMAT
════════════════════════════════════════
Return ONLY a valid JSON object. No markdown, no explanation, no code fences.

{{
  "datasetSummary": "One clear sentence describing what this dataset tracks",
  "cards": [
    {{
      "id": "card_1",
      "label": "Friendly KPI label shown to user",
      "column": "EXACT column name from headers",
      "aggregation": "sum",
      "formatHint": "number"
    }}
  ],
  "charts": [
    {{
      "id": "chart_1",
      "type": "line",
      "title": "Descriptive chart title",
      "x": "EXACT column name (date for line, category for bar/donut)",
      "y": "EXACT numeric column name"
    }}
  ],
  "pivots": [
    {{
      "id": "pivot_1",
      "title": "Descriptive pivot title",
      "rowDim": "EXACT dimension column name",
      "colDim": "EXACT second dimension or null",
      "measure": "EXACT column name to aggregate",
      "aggregation": "sum"
    }}
  ]
}}

STRICT RULES:
1. Every column name must be an EXACT match from: {headers}
2. Never use index/serial/row-number columns (like 'No', '#', 'Sr')
3. cards.aggregation must be one of: "sum", "avg", "count"
4. charts.type must be one of: "line", "bar", "donut"
5. Line chart x MUST be a date column, bar/donut x MUST be a category column
6. pivots.aggregation must be one of: "sum", "avg", "count"
7. If no date column exists, do not generate a line chart
8. Aim for variety — don't repeat the same column across all charts/pivots
9. colDim in pivots should be null if no good second dimension exists
10. TRANSACTION LOG PATTERN: If rows are individual transactions (each row = 1 event/item),
    always SUM the quantity column grouped by date/category — never show raw per-row values.
    A column with grain=transaction_flag means it's mostly 1s and must be SUMmed to be useful.
11. For line charts on transaction data, the y-axis should show TOTAL count per day/period, not individual row values."""


def build_ai_blueprint_wide(analytics: dict, primary_col: str, value_col: str, headers: List[str]) -> dict:
    """
    For wide-format tables, AI still adds value by improving KPI card labels
    and understanding the domain — but analytics are pre-computed.
    """
    prompt = f"""You are a data analyst AI. This is a WIDE FORMAT spreadsheet (dates as columns).
The backend has already computed analytics. Your job is to generate smart KPI card definitions.

Columns in the final table: {headers}
Primary entity column: {primary_col!r}
Value column: {value_col!r}

Return ONLY valid JSON (no markdown):
{{
  "datasetSummary": "One sentence about what this dataset tracks",
  "cards": [
    {{
      "id": "card_1",
      "label": "KPI label",
      "value": 0,
      "formatHint": "currency" | "percent" | "number"
    }}
  ]
}}

For wide format, cards have pre-computed values so set value to 0 — the backend will fill them in.
Generate 3 meaningful KPI cards based on the column names and domain context."""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        raw = response.text.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        return json.loads(raw)
    except Exception as e:
        print(f"[AI Agent Wide] Error: {e}")
        return None


def generate_blueprint_with_ai(
    headers: List[str],
    data: List[Dict],
    profile: dict,
    table_format: str,
    analytics: Optional[dict] = None,
) -> dict:
    """
    Call Gemini to generate a dashboard blueprint, with fallback to
    rule-based logic if the AI call fails.
    """
    prompt = build_ai_prompt(headers, data, profile, table_format)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        raw = response.text.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        ai_blueprint = json.loads(raw)

        # Validate all column references exist
        valid_headers = set(headers)
        def validate_col(col):
            return col if col in valid_headers else None

        # Clean up cards
        cards = []
        for card in ai_blueprint.get("cards", []):
            col = validate_col(card.get("column", ""))
            if col:
                cards.append({**card, "column": col})

        # Clean up charts — validate both x and y columns exist
        charts = []
        for chart in ai_blueprint.get("charts", []):
            x = validate_col(chart.get("x", ""))
            y = validate_col(chart.get("y", ""))
            if x and y:
                charts.append({**chart, "x": x, "y": y})
            else:
                print(f"[AI Agent] Dropping chart '{chart.get('id')}' — invalid columns x={chart.get('x')!r} y={chart.get('y')!r}")

        # Clean up pivots
        pivots = []
        for pivot in ai_blueprint.get("pivots", []):
            row_dim = validate_col(pivot.get("rowDim", ""))
            col_dim = validate_col(pivot.get("colDim")) if pivot.get("colDim") else None
            measure = validate_col(pivot.get("measure", ""))
            if row_dim and measure:
                pivots.append({**pivot, "rowDim": row_dim, "colDim": col_dim, "measure": measure})

        return {
            "datasetSummary": ai_blueprint.get("datasetSummary", ""),
            "cards": cards,
            "charts": charts,
            "pivots": pivots,
            "aiGenerated": True,
        }

    except Exception as e:
        print(f"[AI Agent] Error: {e}. Falling back to rule-based blueprint.")
        return None  # Signal to use fallback


def generate_blueprint_fallback(profile: dict, analytics: Optional[dict], table_format: str) -> dict:
    """
    Original rule-based blueprint generation — used as fallback if AI fails.
    """
    cards, charts, pivots = [], [], []

    measures = pick_measures(profile)
    dimensions = pick_dimensions(profile)
    date_col = pick_date_col(profile)

    primary_measure = measures[0] if measures else None
    primary_dim = dimensions[0] if dimensions else None

    for i, m in enumerate(measures):
        hint = profile[m].get("formatHint", "number")
        cards.append({"id": f"card_sum_{i}", "label": f"Total {m}", "column": m, "aggregation": "sum", "formatHint": hint})
    if primary_dim:
        cards.append({"id": "card_unique", "label": f"Unique {primary_dim}s", "column": primary_dim, "aggregation": "count", "formatHint": "number"})

    if date_col and primary_measure:
        charts.append({
            "id": "chart_line_0",
            "type": "line",
            "title": f"{primary_measure} over time",
            "x": date_col,
            "y": primary_measure,
        })

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

    return {"cards": cards, "charts": charts, "pivots": pivots, "aiGenerated": False}


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

    # ── WIDE FORMAT: analytics already computed, AI enhances card labels ──
    if table_format == "wide" and analytics:
        primary_col = analytics.get("primaryCol")
        value_col = analytics.get("valueCol", "Value")
        period_col = analytics.get("periodCol", "Period")
        period_data = analytics.get("periodTotals")
        primary_data = analytics.get("primaryTotals")
        section_data = analytics.get("sectionTotals")

        # Compute KPI values
        cards = []
        if primary_data:
            objs = rows_to_objects(primary_data["headers"], primary_data["rows"])
            values = [float(r[value_col]) for r in objs if is_number(r.get(value_col))]
            if values:
                hint = detect_format_hint(value_col, values)
                cards = [
                    {"id": "card_total", "label": f"Total {value_col}", "value": sum(values), "formatHint": hint},
                    {"id": "card_avg", "label": f"Avg {value_col} per {primary_col or 'Entity'}", "value": sum(values) / len(values), "formatHint": hint},
                    {"id": "card_count", "label": f"Total {primary_col or 'Entities'}", "value": len(values), "formatHint": "number"},
                ]

        charts = []
        if period_data and len(period_data["rows"]) > 1:
            charts.append({
                "id": "chart_line_period",
                "type": "line",
                "title": f"{value_col} over Time",
                "dataSource": "periodTotals",
                "x": period_col,
                "y": value_col,
            })

        pivots = []
        if primary_data:
            pivots.append({"id": "pivot_primary", "title": f"{value_col} by {primary_col or 'Entity'}", "dataSource": "primaryTotals"})
        if section_data and len(section_data["rows"]) > 1:
            pivots.append({"id": "pivot_section", "title": f"{value_col} by Section", "dataSource": "sectionTotals"})

        return {
            "profile": profile,
            "cards": cards,
            "charts": charts,
            "pivots": pivots,
            "tableFormat": "wide",
            "analytics": analytics,
            "aiGenerated": False,
        }

    # ── LONG FORMAT: AI agent generates blueprint ─────────────────────
    ai_result = generate_blueprint_with_ai(headers, data, profile, table_format, analytics)

    if ai_result:
        return {
            "profile": profile,
            "cards": ai_result["cards"],
            "charts": ai_result["charts"],
            "pivots": ai_result["pivots"],
            "tableFormat": "long",
            "analytics": None,
            "aiGenerated": True,
            "datasetSummary": ai_result.get("datasetSummary", ""),
        }

    # ── FALLBACK: rule-based ──────────────────────────────────────────
    fallback = generate_blueprint_fallback(profile, analytics, table_format)
    return {
        "profile": profile,
        "cards": fallback["cards"],
        "charts": fallback["charts"],
        "pivots": fallback["pivots"],
        "tableFormat": "long",
        "analytics": None,
        "aiGenerated": False,
    }

# ══════════════════════════════════════════════════════
# CHAT ENDPOINT
# ══════════════════════════════════════════════════════

from pydantic import BaseModel

class ChatMessage(BaseModel):
    role: str        # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    headers: List[str]
    rows: List[List]
    datasetSummary: Optional[str] = None

@app.post("/chat")
async def chat(payload: ChatRequest):
    headers = payload.headers
    rows = payload.rows
    messages = payload.messages

    if not messages or not headers:
        return {"reply": "No data or messages provided."}

    # Build compact column profiles from the full dataset server-side
    col_profiles = []
    for col_idx, col in enumerate(headers):
        vals = [
            row[col_idx] for row in rows
            if col_idx < len(row)
            and row[col_idx] is not None
            and str(row[col_idx]).strip() != ""
        ]
        if not vals:
            col_profiles.append(f"  • {col}: empty")
            continue

        num_vals = []
        for v in vals:
            try:
                num_vals.append(float(str(v).replace(",", "")))
            except Exception:
                pass

        if len(num_vals) / len(vals) >= 0.7 and num_vals:
            total = sum(num_vals)
            avg = total / len(num_vals)
            col_profiles.append(
                f"  • {col}: numeric | sum={total:,.2f}, avg={avg:,.2f}, "
                f"min={min(num_vals):,.2f}, max={max(num_vals):,.2f}, count={len(num_vals)}"
            )
        else:
            unique = list({str(v).strip() for v in vals})
            samples = ", ".join(unique[:8])
            col_profiles.append(
                f"  • {col}: category | {len(unique)} unique values: "
                f"[{samples}{'...' if len(unique) > 8 else ''}]"
            )

    # Sample rows as readable text
    sample_lines = []
    for i, row in enumerate(rows[:15]):
        parts = [f"{headers[j]}: {row[j]}" for j in range(min(len(headers), len(row)))]
        sample_lines.append(f"  Row {i+1}: {' | '.join(parts)}")

    # Build full prompt with context + conversation history
    system_context = f"""You are a data analyst assistant. The user has uploaded a spreadsheet and wants to ask questions about it.

DATASET OVERVIEW:
- Total rows: {len(rows):,}
- Columns ({len(headers)}): {headers}
- AI Summary: {payload.datasetSummary or "N/A"}

COLUMN PROFILES (computed from full dataset):
{chr(10).join(col_profiles)}

SAMPLE DATA (first 15 rows):
{chr(10).join(sample_lines)}

INSTRUCTIONS:
- Answer questions about this dataset concisely and accurately
- Use column profiles for aggregate stats (sums, averages, counts, unique values)
- Format numbers with commas for readability
- Keep answers brief unless the user asks for detail
- If a question requires data beyond what is profiled, acknowledge the limitation"""

    # Format conversation history
    history = "\n\n".join(
        f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"
        for m in messages
    )

    full_prompt = f"{system_context}\n\n--- CONVERSATION ---\n{history}\n\nAssistant:"

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
        )
        reply = response.text.strip().removeprefix("Assistant:").strip()
        return {"reply": reply}
    except Exception as e:
        print(f"[Chat] Error: {e}")
        return {"reply": "Sorry, I couldn't process that right now. Please try again."}