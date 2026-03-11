import re
import json
import os
from typing import List, Optional
from google import genai

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def build_chat_response(
    messages: list,
    headers: List[str],
    rows: List[list],
    dataset_summary: Optional[str] = None,
    current_chart_state: Optional[dict] = None,
) -> dict:
    if not messages or not headers:
        return {"reply": "No data or messages provided.", "chartSpec": None, "filterSpec": None}

    # Full dataset as CSV
    csv_header = ",".join(f'"{h}"' for h in headers)
    csv_rows = []
    for row in rows:
        cells = []
        for cell in row:
            if cell is None:
                cells.append("")
            else:
                cells.append(f'"{str(cell)}"')
        csv_rows.append(",".join(cells))
    full_csv = csv_header + "\n" + "\n".join(csv_rows)

    # Column profiles
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

    # Current chart context block
    chart_context_block = ""
    if current_chart_state:
        chart_context_block = f"""
CURRENT CHART STATE (the chart the user is most recently looking at):
{json.dumps(current_chart_state, indent=2)}

When the user sends a follow-up that MODIFIES this chart (e.g. "show top 10", "sort descending",
"change to bar chart", "filter above 100", "limit to 5"), you MUST:
1. Set "action": "modify" in CHART_SPEC
2. Copy and update the current chart state fields accordingly
3. Do NOT create a new chart

When the user asks for a completely NEW analysis (different columns, different question), set "action": "new".
If there is no current chart, always use "action": "new".
"""

    system_context = f"""You are a data analyst assistant. The user has uploaded a spreadsheet.
You have access to the FULL dataset below — use it to answer questions precisely.

DATASET OVERVIEW:
- Total rows: {len(rows):,}
- Columns ({len(headers)}): {headers}
- AI Summary: {dataset_summary or "N/A"}

COLUMN PROFILES (pre-computed aggregates):
{chr(10).join(col_profiles)}
{chart_context_block}
FULL DATASET (CSV):
{full_csv}

INSTRUCTIONS:
- Use the full CSV data to answer questions exactly — count rows, group by columns, sum values, find top N, etc.
- Format numbers with commas for readability
- Keep answers concise unless the user asks for detail
- NEVER render markdown tables in your reply text. If the user asks to see/show/filter rows or columns, use FILTER_SPEC instead.
- After answering, append specs on NEW LINES as described below when relevant.

FILTER_SPEC — use when user wants to SEE rows or specific columns ("show me", "filter", "list", "give me rows where", etc.):
FILTER_SPEC: {{"title": "...", "columns": ["EXACT col1", "EXACT col2"], "filters": [{{
  "column": "EXACT column name", "operator": "eq|neq|contains|gt|gte|lt|lte", "value": "..."
}}]}}
- "columns" = which columns to display (empty array = show all)
- "filters" = conditions to filter rows (empty array = no row filter, just column selection)
- Use FILTER_SPEC whenever showing tabular data — never put tables in reply text

CHART_SPEC — use when a chart would add genuine value (rankings, trends, breakdowns):
CHART_SPEC: {{"action": "new"|"modify", "type": "bar|donut|line|pivot", "title": "...", "x": "EXACT column", "y": "EXACT column", "limit": <number or null>, "sort": "asc|desc|null", "filters": [{{"column": "...", "operator": "...", "value": "..."}}]}}
For pivot: CHART_SPEC: {{"action": "new"|"modify", "type": "pivot", "title": "...", "rowDim": "EXACT column", "colDim": "EXACT column or null", "measure": "EXACT column", "aggregation": "sum|avg|count", "limit": <number or null>, "sort": "asc|desc|null"}}

CHART_SPEC fields:
- "action": REQUIRED. "new" = create new chart, "modify" = update the existing current chart
- "limit": top N rows to show (e.g. 10 for "top 10"). null = no limit
- "sort": "desc" = highest first (default), "asc" = lowest first, null = natural order
- "filters": row-level filters to apply before aggregation (same format as FILTER_SPEC filters)

Column names in ALL specs MUST exactly match one of: {headers}
Never output CHART_SPEC for simple factual questions. Never output both FILTER_SPEC and CHART_SPEC together."""

    history = "\n\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in messages
    )
    full_prompt = f"{system_context}\n\n--- CONVERSATION ---\n{history}\n\nAssistant:"

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
        )
        raw = response.text.strip().removeprefix("Assistant:").strip()

        chart_spec = None
        filter_spec = None
        reply_text = raw

        # Parse FILTER_SPEC first
        if "FILTER_SPEC:" in reply_text:
            parts = reply_text.split("FILTER_SPEC:", 1)
            reply_text = parts[0].strip()
            try:
                spec_raw = parts[1].strip()
                spec_raw = re.sub(r'^```(?:json)?\s*', '', spec_raw)
                spec_raw = re.sub(r'\s*```$', '', spec_raw)
                brace_count = 0
                end_idx = 0
                for i, ch in enumerate(spec_raw):
                    if ch == '{': brace_count += 1
                    elif ch == '}': brace_count -= 1
                    if brace_count == 0 and i > 0:
                        end_idx = i + 1
                        break
                filter_spec = json.loads(spec_raw[:end_idx])
                valid = set(headers)
                filter_spec["columns"] = [c for c in filter_spec.get("columns", []) if c in valid]
                filter_spec["filters"] = [
                    f for f in filter_spec.get("filters", []) if f.get("column") in valid
                ]
                if not filter_spec.get("title"):
                    filter_spec["title"] = "Filtered Data"
                filter_spec["id"] = str(id(filter_spec))
            except Exception as parse_err:
                print(f"[Chat] Filter spec parse error: {parse_err}")
                filter_spec = None

        # Parse CHART_SPEC
        if "CHART_SPEC:" in reply_text:
            parts = reply_text.split("CHART_SPEC:", 1)
            reply_text = parts[0].strip()
            try:
                spec_raw = parts[1].strip()
                spec_raw = re.sub(r'^```(?:json)?\s*', '', spec_raw)
                spec_raw = re.sub(r'\s*```$', '', spec_raw)
                chart_spec = json.loads(spec_raw)
                valid = set(headers)

                # Validate column refs
                if chart_spec.get("x") and chart_spec["x"] not in valid:
                    chart_spec = None
                if chart_spec and chart_spec.get("y") and chart_spec["y"] not in valid:
                    chart_spec = None
                if chart_spec and chart_spec.get("rowDim") and chart_spec["rowDim"] not in valid:
                    chart_spec = None

                # Default action to "new" if missing
                if chart_spec and "action" not in chart_spec:
                    chart_spec["action"] = "new"

                # Validate filter columns in chart spec
                if chart_spec and chart_spec.get("filters"):
                    chart_spec["filters"] = [
                        f for f in chart_spec["filters"] if f.get("column") in valid
                    ]

            except Exception as parse_err:
                print(f"[Chat] Chart spec parse error: {parse_err}")
                chart_spec = None

        return {"reply": reply_text, "chartSpec": chart_spec, "filterSpec": filter_spec}

    except Exception as e:
        print(f"[Chat] Error: {e}")
        return {
            "reply": "Sorry, I couldn't process that right now. Please try again.",
            "chartSpec": None,
            "filterSpec": None,
        }