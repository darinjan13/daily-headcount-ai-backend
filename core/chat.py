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
        return {"reply": "No data or messages provided.", "steps": []}

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

    col_profiles = []
    for col_idx, col in enumerate(headers):
        vals = [
            row[col_idx] for row in rows
            if col_idx < len(row)
            and row[col_idx] is not None
            and str(row[col_idx]).strip() != ""
        ]
        if not vals:
            col_profiles.append(f"  * {col}: empty")
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
                f"  * {col}: numeric | sum={total:,.2f}, avg={avg:,.2f}, "
                f"min={min(num_vals):,.2f}, max={max(num_vals):,.2f}, count={len(num_vals)}"
            )
        else:
            unique = list({str(v).strip() for v in vals})
            samples = ", ".join(unique[:8])
            col_profiles.append(
                f"  * {col}: category | {len(unique)} unique values: "
                f"[{samples}{'...' if len(unique) > 8 else ''}]"
            )

    charts_context_block = ""
    if existing_charts:
        chart_list = "\n".join(
            f"  - id:{c.get('id')} title:\"{c.get('title')}\" type:{c.get('type')} pinned:{c.get('pinned', False)}"
            for c in existing_charts
        )
        charts_context_block += f"\nEXISTING CHARTS (currently in the dashboard):\n{chart_list}\n"

    existing_tables_block = "(none)"
    if existing_tables:
        rows_list = "\n".join(
            f"  - id:{t.get('id')} title:\"{t.get('title')}\" pinned:{t.get('pinned', False)} filters:{len(t.get('filters', []))} active"
            for t in existing_tables
        )
        existing_tables_block = rows_list
        charts_context_block += f"\nEXISTING FILTER TABLES (currently on Home tab):\n{rows_list}\n"

    chart_context_block = ""
    if current_chart_state:
        chart_context_block = f"""
CURRENT CHART STATE (chart the user is editing):
{json.dumps(current_chart_state, indent=2)}

When the user modifies this chart (e.g. "show top 10", "sort descending", "change to bar", "filter above 100"):
- Output a chart step with "action": "modify" and update the relevant fields
When the user says "delete this" or "delete it":
- Output: {{"type": "delete", "targetTitle": "{current_chart_state.get('title', '')}"}}
When the user asks for a completely NEW analysis, use "action": "new".
"""

    system_context = f"""You are a powerful data analyst AI with full control over the dashboard.
The user has uploaded a spreadsheet. You have access to the FULL dataset below.

DATASET OVERVIEW:
- Total rows: {len(rows):,}
- Columns ({len(headers)}): {headers}
- AI Summary: {dataset_summary or "N/A"}

COLUMN PROFILES:
{chr(10).join(col_profiles)}
{charts_context_block}{chart_context_block}
FULL DATASET (CSV):
{full_csv}

INSTRUCTIONS:
- Use full CSV to answer questions precisely — count, sum, rank, group, compare
- Format numbers with commas
- NEVER render markdown tables in reply — use filter step instead
- Output a STEPS block whenever any dashboard action is needed

STEPS SYSTEM:
Output after your reply:
STEPS: [
  {{"type": "...", ...spec}},
  ...
]
Steps execute IN ORDER. Combine any types in one block.

=== CHART STEPS (type: "chart") ===
Bar/Hbar/Donut/Line:
{{"type":"chart","action":"new","chartType":"bar|hbar|donut|line","title":"...","x":"EXACT col","y":"EXACT col","limit":<n|null>,"sort":"asc|desc|null","filters":[]}}
Pivot:
{{"type":"chart","action":"new","chartType":"pivot","title":"...","rowDim":"EXACT col","colDim":"EXACT col|null","measure":"EXACT col","aggregation":"sum|avg|count","limit":<n|null>}}
Modify selected chart:
{{"type":"chart","action":"modify","chartType":"...","title":"...","x":"...","y":"...","limit":<n>,"sort":"asc|desc"}}

=== FILTER TABLE STEPS (type: "filter") ===
Create a filtered data table on Home tab:
{{"type":"filter","title":"...","columns":["EXACT col",...],"filters":[{{"column":"EXACT col","operator":"eq|neq|contains|gt|gte|lt|lte","value":"..."}}],"sort_col":"EXACT col|null","sort_dir":"asc|desc|null","limit":<n|null>}}

=== DELETE CHART STEPS (type: "delete") ===
{{"type":"delete","deleteAll":true}}
{{"type":"delete","targetTitle":"<exact chart title>"}}

=== PIN CHART STEPS (type: "pin") ===
{{"type":"pin","pinAll":true}}
{{"type":"pin","unpinAll":true}}
{{"type":"pin","targetTitle":"<exact chart title>","pinned":true|false}}

=== RENAME CHART (type: "rename") ===
{{"type":"rename","targetTitle":"<current title>","newTitle":"<new title>"}}

=== MODIFY CHART DISPLAY (type: "modify_chart") ===
Only changes limit/sort, no data refetch:
{{"type":"modify_chart","targetTitle":"<exact title>","limit":<n|null>,"sort":"asc|desc|null"}}

=== NAVIGATE (type: "navigate") ===
{{"type":"navigate","tab":"home|charts"}}

=== TABLE MANAGEMENT (type: "table_action") ===
ALL filter table operations. Use ONLY table_action (never delete_table):

Delete:       {{"type":"table_action","action":"delete","targetTitle":"<exact title>"}}
Delete all:   {{"type":"table_action","action":"deleteAll"}}
Pin:          {{"type":"table_action","action":"pin","targetTitle":"<exact title>"}}
Pin all:      {{"type":"table_action","action":"pinAll"}}
Unpin all:    {{"type":"table_action","action":"unpinAll"}}
Rename:       {{"type":"table_action","action":"rename","targetTitle":"<current title>","newTitle":"<new title>"}}
Sort:         {{"type":"table_action","action":"sort","targetTitle":"<exact title>","sort_col":"EXACT col","sort_dir":"asc|desc"}}
Limit rows:   {{"type":"table_action","action":"limit","targetTitle":"<exact title>","limit":<n>}}
Add filter:   {{"type":"table_action","action":"add_filter","targetTitle":"<exact title>","filter":{{"column":"EXACT col","operator":"eq|neq|contains|gt|gte|lt|lte","value":"..."}}}}
Remove filter:{{"type":"table_action","action":"remove_filter","targetTitle":"<exact title>","filter_column":"EXACT col"}}

EXISTING FILTER TABLES:
{existing_tables_block}

RULES:
- Column names MUST exactly match one of: {headers}
- NEVER use delete_table type — always use table_action
- "delete [name] table" / "remove [name] table" → table_action delete
- "delete all tables" / "clear all tables" → table_action deleteAll
- "sort [table] by [col] ascending/descending" → table_action sort
- "only show top N rows in [table]" → table_action limit
- "filter [table] where X > Y" → table_action add_filter
- "remove [col] filter from [table]" → table_action remove_filter
- "pin [table]" → table_action pin
- "pin all tables" → table_action pinAll
- "rename [table] to X" → table_action rename
- "pin chart X" → pin step with targetTitle from EXISTING CHARTS
- "go to home/charts tab" → navigate step
- Deletes before creates in multi-step sequences"""

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
                        if step.get("x") and step["x"] not in valid: continue
                        if step.get("y") and step["y"] not in valid: continue
                        if step.get("rowDim") and step["rowDim"] not in valid: continue
                        if "chartType" not in step: step["chartType"] = "bar"
                        if "action" not in step: step["action"] = "new"
                        if step.get("filters"):
                            step["filters"] = [f for f in step["filters"] if f.get("column") in valid]
                        steps.append(step)

                    elif stype == "filter":
                        step["columns"] = [c for c in step.get("columns", []) if c in valid]
                        step["filters"] = [f for f in step.get("filters", []) if f.get("column") in valid]
                        if step.get("sort_col") and step["sort_col"] not in valid:
                            step["sort_col"] = None
                        if not step.get("title"): step["title"] = "Filtered Data"
                        step["id"] = str(id(step))
                        steps.append(step)

                    elif stype == "delete":
                        steps.append(step)

                    # Auto-translate legacy delete_table to table_action
                    elif stype == "delete_table":
                        if step.get("deleteAll"):
                            steps.append({"type": "table_action", "action": "deleteAll"})
                        elif step.get("targetTitle"):
                            steps.append({"type": "table_action", "action": "delete", "targetTitle": step["targetTitle"]})

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
                        act = step.get("action")
                        if not act:
                            continue
                        if step.get("sort_col") and step["sort_col"] not in valid:
                            step["sort_col"] = None
                        if act == "add_filter":
                            if not step.get("filter", {}).get("column") in valid:
                                continue
                        steps.append(step)

            except Exception as parse_err:
                print(f"[Chat] Steps parse error: {parse_err}")
                steps = []

        # Legacy fallback
        if not steps:
            if "DELETE_SPEC:" in reply_text:
                parts = reply_text.split("DELETE_SPEC:", 1)
                reply_text = parts[0].strip()
                try:
                    spec_raw = re.sub(r'^```(?:json)?\s*', '', parts[1].strip())
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
                    spec_raw = re.sub(r'^```(?:json)?\s*', '', parts[1].strip())
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
                    spec_raw = re.sub(r'^```(?:json)?\s*', '', parts[1].strip())
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