import re
import json
import os
from typing import List, Dict, Optional

from .helpers import is_number, rows_to_objects
from google import genai

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


# ── Column profiling ───────────────────────────────────────────────────────────

def try_parse_date(x) -> bool:
    import datetime
    import pandas as pd
    if x is None:
        return False
    if isinstance(x, (pd.Timestamp, datetime.datetime, datetime.date)):
        return True
    if isinstance(x, str):
        s = x.strip()
        if not s or len(s) < 4:
            return False
        try:
            pd.to_datetime(s, errors="raise")
            return True
        except Exception:
            return False
    return False


def detect_format_hint(col: str, values: list) -> str:
    name = col.lower()
    if any(k in name for k in [
        "amount", "cost", "revenue", "sales", "price",
        "budget", "expense", "income", "payment", "wage", "salary", "pay"
    ]):
        return "currency"
    if any(k in name for k in [
        "rate", "ratio", "pct", "percent", "%", "margin",
        "utilization", "efficiency"
    ]):
        return "percent"
    nums = []
    for v in values[:30]:
        try:
            nums.append(float(str(v).replace(",", "")))
        except Exception:
            pass
    if nums and all(0 <= n <= 1 for n in nums):
        # only treat as percent_decimal if values are genuinely fractional
        # (not just integers that happen to be 0 or 1, like count columns)
        has_fractions = any(n != int(n) for n in nums)
        has_range = any(0 < n < 1 for n in nums)
        if has_fractions or has_range:
            return "percent_decimal"
    return "number"


def detect_column_profile(data: List[Dict], sample_size: int = 100) -> dict:
    if not data:
        return {}
    sample = data[:min(sample_size, len(data))]
    profile = {}
    for col in sample[0].keys():
        values = [row.get(col) for row in sample]
        non_null = [
            v for v in values
            if v is not None and str(v).strip() not in ("", "None", "nan")
        ]
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
                mostly_ones = sum(1 for x in nums if x == 1) / len(nums) > 0.8
                entry["grain"] = "transaction_flag" if mostly_ones else "measure"
                entry["formatHint"] = detect_format_hint(col, nums)

        profile[col] = entry
    return profile


def score_measure(col: str, profile: dict) -> int:
    p = profile[col]
    if p["type"] != "numeric":
        return -1
    score = 0
    name = col.lower()
    score += sum(3 for k in [
        "total", "amount", "revenue", "sales", "cost",
        "count", "qty", "hours", "units", "output", "score",
    ] if k in name)
    score += sum(-3 for k in [
        "id", "no", "num", "index", "sr", "row", "rank", "year",
    ] if k in name)
    if p.get("grain") == "transaction_flag":
        score -= 2
    if p.get("unique", 0) == p.get("nonNull", 0):
        score -= 5
    return score


def pick_measures(profile: dict, max_n: int = 3) -> List[str]:
    cols = [c for c, p in profile.items() if p["type"] == "numeric"]
    scored = [(score_measure(c, profile), c) for c in cols]
    scored = [(s, c) for s, c in scored if s >= 0]
    scored.sort(reverse=True)
    return [c for _, c in scored[:max_n]]


def score_dimension(col: str, profile: dict) -> int:
    p = profile[col]
    if p["type"] not in ("category", "identifier"):
        return -1
    score = 0
    name = col.lower()
    u = p.get("unique", 0)
    n = p.get("nonNull", 1)
    if u < 2 or u > 100:
        return -1
    score += sum(3 for k in [
        "name", "category", "type", "status", "region",
        "site", "department", "group", "team", "product",
    ] if k in name)
    if 2 <= u <= 20:
        score += 3
    elif 21 <= u <= 50:
        score += 1
    fill = n / max(len(profile), 1)
    if fill > 0.8:
        score += 2
    return score


def pick_dimensions(profile: dict, max_n: int = 3) -> List[str]:
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
    return sorted(
        dates,
        key=lambda c: sum(3 for k in priority if k in c.lower()),
        reverse=True,
    )[0]


# ── Fallback rule-based blueprint ─────────────────────────────────────────────

def generate_blueprint_fallback(
    profile: dict,
    analytics: Optional[dict],
    table_format: str,
) -> dict:
    cards, charts, pivots = [], [], []
    measures = pick_measures(profile)
    dimensions = pick_dimensions(profile)
    date_col = pick_date_col(profile)
    primary_measure = measures[0] if measures else None
    primary_dim = dimensions[0] if dimensions else None

    for i, m in enumerate(measures):
        hint = profile[m].get("formatHint", "number")
        cards.append({
            "id": f"card_sum_{i}",
            "label": f"Total {m}",
            "column": m,
            "aggregation": "sum",
            "formatHint": hint,
        })
    if primary_dim:
        cards.append({
            "id": "card_unique",
            "label": f"Unique {primary_dim}s",
            "column": primary_dim,
            "aggregation": "count",
            "formatHint": "number",
        })

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


# ── Wide format blueprint ──────────────────────────────────────────────────────

def generate_blueprint_wide(
    profile: dict,
    analytics: dict,
) -> dict:
    primary_col = analytics.get("primaryCol")
    value_col = analytics.get("valueCol", "Value")
    period_col = analytics.get("periodCol", "Period")
    period_data = analytics.get("periodTotals")
    primary_data = analytics.get("primaryTotals")
    section_data = analytics.get("sectionTotals")

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
        pivots.append({
            "id": "pivot_primary",
            "title": f"{value_col} by {primary_col or 'Entity'}",
            "dataSource": "primaryTotals",
        })
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
        "aiGenerated": False,
    }


# ── Main blueprint builder ─────────────────────────────────────────────────────

def build_blueprint(table_data: dict) -> dict:
    """
    Takes table_data payload (headers, rows, tableFormat, analytics)
    Returns full blueprint dict.
    """
    headers = table_data.get("headers", [])
    rows = table_data.get("rows", [])
    analytics = table_data.get("analytics", None)
    table_format = table_data.get("tableFormat", "long")

    data_objects = rows_to_objects(headers, rows)
    profile = detect_column_profile(data_objects)

    if table_format == "wide" and analytics:
        return generate_blueprint_wide(profile, analytics)

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