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

    system_context = f"""You are a data analyst assistant. The user has uploaded a spreadsheet.
You have access to the FULL dataset below — use it to answer questions precisely.

DATASET OVERVIEW:
- Total rows: {len(rows):,}
- Columns ({len(headers)}): {headers}
- AI Summary: {dataset_summary or "N/A"}

COLUMN PROFILES (pre-computed aggregates):
{chr(10).join(col_profiles)}

FULL DATASET (CSV):
{full_csv}

INSTRUCTIONS:
- Use the full CSV data to answer questions exactly — count rows, group by columns, sum values, find top N, etc.
- Format numbers with commas for readability
- Keep answers concise unless the user asks for detail
- NEVER output raw CSV, raw data rows, or data as plain text in your reply. Data is always shown via FILTER_SPEC or CHART_SPEC.

---

CRITICAL RULE — FILTER_SPEC:
When the user asks to show, list, display, filter, or see specific rows or columns
(e.g. "show me section and name", "show me only sep 16-19", "list everyone from LG Homebased",
"give me name and userid", "show columns X Y Z", "filter by section"),
you MUST respond with a short one-sentence reply + a FILTER_SPEC on the next line.
NEVER dump the data as CSV or plain text. ALWAYS use FILTER_SPEC instead.
FILTER_SPEC results appear as an interactive table in the HOME TAB of the dashboard.

When the user specifies which columns to show (e.g. "show me name, userid, sep 16-19"),
put exactly those columns in the "columns" array of the FILTER_SPEC.
Use an empty "filters" array [] if there are no row conditions — this shows ALL rows with just those columns.

FILTER_SPEC format:
FILTER_SPEC: {{"title": "...", "columns": ["col1", "col2", ...], "filters": [{{"column": "EXACT col", "operator": "eq|contains|gt|gte|lt|lte|neq", "value": "..."}}]}}

Rules for FILTER_SPEC:
- "title": short descriptive label shown above the table (e.g. "LG Homebased — Above 200,000")
- "columns": list of column names to SHOW in the table — use exactly the columns the user asked for.
  Column names MUST exactly match one of: {headers}
- "filters": conditions to filter ROWS. Use [] if no row filter is needed (user only asked for columns).
  - "column": MUST exactly match one of {headers}
  - "operator": "eq" exact match, "contains" partial text, "gt"/"gte"/"lt"/"lte" numeric, "neq" not equal
  - "value": always a string
- Do NOT output FILTER_SPEC for aggregation questions (totals, averages, rankings) — use CHART_SPEC for those.

Examples:
User: "show me section, name, userid, sep 16-19"
→ FILTER_SPEC: {{"title": "Section, Name, UserID — Sep 16 to 19", "columns": ["Section", "NAME", "UserID", "Sep 16", "Sep 17", "Sep 18", "Sep 19"], "filters": []}}

User: "show employees from LG Homebased"
→ FILTER_SPEC: {{"title": "LG Homebased Employees", "columns": ["Section", "NAME", "Total Production"], "filters": [{{"column": "Section", "operator": "eq", "value": "LG Homebased"}}]}}

User: "list everyone above 200,000"
→ FILTER_SPEC: {{"title": "High Performers — Above 200,000", "columns": ["Section", "NAME", "Total Production"], "filters": [{{"column": "Total Production", "operator": "gt", "value": "200000"}}]}}

---

CHART_SPEC — use this for rankings, trends, and breakdowns (NOT for filter/show/list requests).
CHART_SPEC results appear as a chart in the CHARTS TAB of the dashboard.
CHART_SPEC: {{"type": "bar", "title": "...", "x": "EXACT column name", "y": "EXACT column name"}}
CHART_SPEC: {{"type": "donut", "title": "...", "x": "EXACT column name", "y": "EXACT column name"}}
CHART_SPEC: {{"type": "line", "title": "...", "x": "EXACT date column", "y": "EXACT numeric column"}}
CHART_SPEC: {{"type": "pivot", "title": "...", "rowDim": "EXACT column", "colDim": "EXACT column or null", "measure": "EXACT column", "aggregation": "sum"}}

Column names MUST exactly match one of: {headers}
Only output one spec per response. FILTER_SPEC and CHART_SPEC cannot both appear in the same reply."""

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

        # ── Parse FILTER_SPEC ──────────────────────────────────────────────
        if "FILTER_SPEC:" in raw:
            parts = raw.split("FILTER_SPEC:", 1)
            reply_text = parts[0].strip()
            try:
                spec_raw = parts[1].strip()
                spec_raw = re.sub(r'^```(?:json)?\s*', '', spec_raw)
                spec_raw = re.sub(r'\s*```$', '', spec_raw)
                parsed = json.loads(spec_raw)
                valid = set(headers)
                # validate all filter columns exist
                bad = any(
                    f.get("column") not in valid
                    for f in parsed.get("filters", [])
                )
                # validate all display columns exist
                bad = bad or any(c not in valid for c in parsed.get("columns", []))
                if not bad:
                    filter_spec = parsed
                else:
                    print(f"[Chat] FILTER_SPEC column validation failed: {parsed}")
            except Exception as parse_err:
                print(f"[Chat] Filter spec parse error: {parse_err}")

        # ── Parse CHART_SPEC ───────────────────────────────────────────────
        elif "CHART_SPEC:" in raw:
            parts = raw.split("CHART_SPEC:", 1)
            reply_text = parts[0].strip()
            try:
                spec_raw = parts[1].strip()
                spec_raw = re.sub(r'^```(?:json)?\s*', '', spec_raw)
                spec_raw = re.sub(r'\s*```$', '', spec_raw)
                chart_spec = json.loads(spec_raw)
                valid = set(headers)
                if chart_spec.get("x") and chart_spec["x"] not in valid:
                    chart_spec = None
                if chart_spec and chart_spec.get("y") and chart_spec["y"] not in valid:
                    chart_spec = None
                if chart_spec and chart_spec.get("rowDim") and chart_spec["rowDim"] not in valid:
                    chart_spec = None
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