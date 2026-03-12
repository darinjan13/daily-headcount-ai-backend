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
    existing_charts: Optional[list] = None,
    existing_tables: Optional[list] = None,
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

    # Existing charts context
    charts_context_block = ""
    if existing_charts:
        chart_list = "\n".join(
            f"  - id:{c.get('id')} title:\"{c.get('title')}\" type:{c.get('type')} pinned:{c.get('pinned', False)}"
            for c in existing_charts
        )
        charts_context_block = f"\nEXISTING CHARTS (currently in the dashboard):\n{chart_list}\n"

    if existing_tables:
        table_list = "\n".join(
            f"  - id:{t.get('id')} title:\"{t.get('title')}\""
            for t in existing_tables
        )
        charts_context_block += f"\nEXISTING FILTER TABLES (currently in the dashboard):\n{table_list}\n"

    # Existing tables context
    existing_tables_block = "(none)" if not existing_tables else "\n".join(
        f"  - id:{t.get('id')} title:\"{t.get('title')}\" pinned:{t.get('pinned', False)} filters:{len(t.get('filters', []))} active"
        for t in existing_tables
    )

    # Current chart context block
    chart_context_block = ""
    if current_chart_state:
        chart_context_block = f"""
CURRENT CHART STATE (the chart the user is currently editing):
{json.dumps(current_chart_state, indent=2)}

When the user modifies this chart (e.g. "show top 10", "sort descending", "change to bar", "filter above 100"):
- Output a chart step with "action": "modify" and update the relevant fields

When the user says "delete this" or "delete it":
- Output a delete step: {{"type": "delete", "targetTitle": "{current_chart_state.get('title', '')}"}}

When the user asks for a completely NEW analysis, use "action": "new".
"""

    system_context = f"""You are a data analyst assistant. The user has uploaded a spreadsheet.
You have access to the FULL dataset below — use it to answer questions precisely.

DATASET OVERVIEW:
- Total rows: {len(rows):,}
- Columns ({len(headers)}): {headers}
- AI Summary: {dataset_summary or "N/A"}

COLUMN PROFILES (pre-computed aggregates):
{chr(10).join(col_profiles)}
{charts_context_block}{chart_context_block}
FULL DATASET (CSV):
{full_csv}

INSTRUCTIONS:
- Use the full CSV data to answer questions exactly — count rows, group by columns, sum values, find top N, etc.
- Format numbers with commas for readability
- Keep answers concise unless the user asks for detail
- NEVER render markdown tables in your reply text.
- After your reply, output a STEPS block if any actions are needed.

STEPS SYSTEM — for ANY action (charts, filters, deletes, pins), output a STEPS block:
STEPS: [
  {{"type": "...", ...spec}},
  {{"type": "...", ...spec}}
]

Each step has a "type" field. Steps execute IN ORDER. You can combine multiple steps in one STEPS block.

Step types:

1. Chart (type: "chart") — create or modify a chart:
{{"type": "chart", "action": "new"|"modify", "chartType": "bar|hbar|donut|line|pivot", "title": "...", "x": "EXACT col", "y": "EXACT col", "limit": <n|null>, "sort": "asc|desc|null", "filters": []}}
For pivot: {{"type": "chart", "action": "new"|"modify", "chartType": "pivot", "title": "...", "rowDim": "EXACT col", "colDim": "EXACT col|null", "measure": "EXACT col", "aggregation": "sum|avg|count", "limit": <n|null>}}

2. Filter table (type: "filter") — show rows in a table:
{{"type": "filter", "title": "...", "columns": ["EXACT col", ...], "filters": [{{"column": "EXACT col", "operator": "eq|neq|contains|gt|gte|lt|lte", "value": "..."}}]}}

3. Delete chart (type: "delete") — remove chart(s):
{{"type": "delete", "deleteAll": true}} — removes ALL custom charts
{{"type": "delete", "targetTitle": "<exact chart title>"}} — removes ONE specific chart

4. Delete filter table (type: "delete_table") — remove filter table(s):
{{"type": "delete_table", "deleteAll": true}} — removes ALL filter tables
{{"type": "delete_table", "targetTitle": "<exact table title>"}} — removes ONE specific table

4. Pin/unpin chart (type: "pin") — pin or unpin chart(s):
{{"type": "pin", "pinAll": true}} — pins ALL existing charts
{{"type": "pin", "unpinAll": true}} — unpins ALL charts
{{"type": "pin", "targetTitle": "<exact chart title>", "pinned": true|false}} — pin or unpin ONE chart by title

5. Rename chart (type: "rename") — rename an existing chart:
{{"type": "rename", "targetTitle": "<current title>", "newTitle": "<new title>"}}

6. Navigate (type: "navigate") — switch the active dashboard tab:
{{"type": "navigate", "tab": "home"|"charts"}}

7. Sort/limit existing chart (type: "modify_chart") — update display settings on an existing chart without regenerating:
{{"type": "modify_chart", "targetTitle": "<exact title>", "limit": <n|null>, "sort": "asc|desc|null"}}

8. Filter table control (type: "table_action") — manage existing filtered tables:
{{"type": "table_action", "action": "delete", "targetTitle": "<exact title>"}} — delete a specific table
{{"type": "table_action", "action": "deleteAll"}} — delete ALL filtered tables
{{"type": "table_action", "action": "pin", "targetTitle": "<exact title>"}} — pin a table so it persists
{{"type": "table_action", "action": "pinAll"}} — pin ALL filtered tables
{{"type": "table_action", "action": "rename", "targetTitle": "<current title>", "newTitle": "<new title>"}} — rename a table

EXISTING FILTERED TABLES (currently visible on Home tab):
{existing_tables_block}
RULES:
- Output STEPS only when action is needed. For pure questions, no STEPS needed.
- Steps run in the order listed — e.g. delete first, then create: [{{"type":"delete","deleteAll":true}}, {{"type":"chart",...}}]
- When user says "pin all", "save all", "keep all" → use pin step with pinAll: true
- When user says "pin X chart" → use pin step with targetTitle matching from EXISTING CHARTS
- When creating charts and user also says "pin them" → add chart steps then a pin step with pinAll: true
- When user says "rename X to Y" → use rename step
- When user says "delete all tables", "clear tables", "remove filtered tables" → table_action deleteAll
- When user says "pin the X table", "save X table" → table_action pin with targetTitle
- When user says "pin all tables" → table_action pinAll
- When user says "rename X table to Y" → table_action rename
- Column names in ALL steps MUST exactly match one of: {headers}
- Never put tabular data in reply text — always use a filter step instead
- Use "chart" steps for rankings, trends, breakdowns; use "filter" steps for showing raw rows
- For modify: only output a chart step with action "modify" when a chart is currently selected"""

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

        reply_text = raw
        steps = []

        # Parse STEPS block
        if "STEPS:" in reply_text:
            parts = reply_text.split("STEPS:", 1)
            reply_text = parts[0].strip()
            try:
                steps_raw = parts[1].strip()
                steps_raw = re.sub(r'^```(?:json)?\s*', '', steps_raw)
                steps_raw = re.sub(r'\s*```$', '', steps_raw)
                bracket_count = 0
                end_idx = 0
                for i, ch in enumerate(steps_raw):
                    if ch == '[': bracket_count += 1
                    elif ch == ']': bracket_count -= 1
                    if bracket_count == 0 and i > 0:
                        end_idx = i + 1
                        break
                parsed_steps = json.loads(steps_raw[:end_idx])
                valid = set(headers)

                for step in parsed_steps:
                    stype = step.get("type")

                    if stype == "chart":
                        chart_type = step.get("chartType", "bar")
                        step["chartType"] = chart_type
                        if step.get("x") and step["x"] not in valid: continue
                        if step.get("y") and step["y"] not in valid: continue
                        if step.get("rowDim") and step["rowDim"] not in valid: continue
                        if "action" not in step: step["action"] = "new"
                        if step.get("filters"):
                            step["filters"] = [f for f in step["filters"] if f.get("column") in valid]
                        steps.append(step)

                    elif stype == "filter":
                        step["columns"] = [c for c in step.get("columns", []) if c in valid]
                        step["filters"] = [f for f in step.get("filters", []) if f.get("column") in valid]
                        if not step.get("title"): step["title"] = "Filtered Data"
                        step["id"] = str(id(step))
                        steps.append(step)

                    elif stype == "delete":
                        steps.append(step)

                    elif stype == "delete_table":
                        steps.append(step)

                    elif stype == "pin":
                        steps.append(step)

                    elif stype == "rename":
                        if step.get("targetTitle") and step.get("newTitle"):
                            steps.append(step)

                    elif stype == "navigate":
                        if step.get("tab") in ("home", "charts"):
                            steps.append(step)

                    elif stype == "modify_chart":
                        if step.get("targetTitle"):
                            steps.append(step)

                    elif stype == "table_action":
                        if step.get("action"):
                            steps.append(step)

            except Exception as parse_err:
                print(f"[Chat] Steps parse error: {parse_err}")
                steps = []

        # Legacy fallback for old spec formats
        if not steps:
            if "DELETE_SPEC:" in reply_text:
                parts = reply_text.split("DELETE_SPEC:", 1)
                reply_text = parts[0].strip()
                try:
                    spec_raw = parts[1].strip()
                    spec_raw = re.sub(r'^```(?:json)?\s*', '', spec_raw)
                    brace_count = 0; end_idx = 0
                    for i, ch in enumerate(spec_raw):
                        if ch == '{': brace_count += 1
                        elif ch == '}': brace_count -= 1
                        if brace_count == 0 and i > 0: end_idx = i + 1; break
                    steps.append({"type": "delete", **json.loads(spec_raw[:end_idx])})
                except Exception as e: print(f"[Chat] Legacy delete parse: {e}")

            if "FILTER_SPEC:" in reply_text:
                parts = reply_text.split("FILTER_SPEC:", 1)
                reply_text = parts[0].strip()
                try:
                    spec_raw = parts[1].strip()
                    spec_raw = re.sub(r'^```(?:json)?\s*', '', spec_raw)
                    brace_count = 0; end_idx = 0
                    for i, ch in enumerate(spec_raw):
                        if ch == '{': brace_count += 1
                        elif ch == '}': brace_count -= 1
                        if brace_count == 0 and i > 0: end_idx = i + 1; break
                    fs = json.loads(spec_raw[:end_idx])
                    valid = set(headers)
                    fs["columns"] = [c for c in fs.get("columns", []) if c in valid]
                    fs["filters"] = [f for f in fs.get("filters", []) if f.get("column") in valid]
                    if not fs.get("title"): fs["title"] = "Filtered Data"
                    fs["id"] = str(id(fs))
                    steps.append({"type": "filter", **fs})
                except Exception as e: print(f"[Chat] Legacy filter parse: {e}")

            if "CHART_SPEC:" in reply_text:
                parts = reply_text.split("CHART_SPEC:", 1)
                reply_text = parts[0].strip()
                try:
                    spec_raw = parts[1].strip()
                    spec_raw = re.sub(r'^```(?:json)?\s*', '', spec_raw)
                    cs = json.loads(spec_raw)
                    valid = set(headers)
                    if cs.get("x") and cs["x"] not in valid: cs = None
                    if cs and cs.get("y") and cs["y"] not in valid: cs = None
                    if cs and cs.get("rowDim") and cs["rowDim"] not in valid: cs = None
                    if cs:
                        if "action" not in cs: cs["action"] = "new"
                        cs["chartType"] = cs.pop("type", "bar")
                        cs["type"] = "chart"
                        steps.append(cs)
                except Exception as e: print(f"[Chat] Legacy chart parse: {e}")

        return {"reply": reply_text, "steps": steps}

    except Exception as e:
        print(f"[Chat] Error: {e}")
        return {"reply": "Sorry, I couldn't process that right now. Please try again.", "steps": []}