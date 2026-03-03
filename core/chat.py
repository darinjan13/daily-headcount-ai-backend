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
        return {"reply": "No data or messages provided.", "chartSpec": None}

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
- After answering, if the question implies a useful chart or pivot table could visualize the answer,
  append a JSON spec on a NEW LINE starting with exactly "CHART_SPEC:" followed by valid JSON.
  Only suggest a spec when it adds genuine value (rankings, trends, breakdowns).

CHART_SPEC format (pick the most appropriate type):
CHART_SPEC: {{"type": "bar", "title": "...", "x": "EXACT column name", "y": "EXACT column name"}}
CHART_SPEC: {{"type": "donut", "title": "...", "x": "EXACT column name", "y": "EXACT column name"}}
CHART_SPEC: {{"type": "line", "title": "...", "x": "EXACT date column", "y": "EXACT numeric column"}}
CHART_SPEC: {{"type": "pivot", "title": "...", "rowDim": "EXACT column", "colDim": "EXACT column or null", "measure": "EXACT column", "aggregation": "sum"}}

Column names in spec MUST exactly match one of: {headers}
Only output CHART_SPEC if confident it adds value. Never output it for simple factual questions."""

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
        reply_text = raw
        if "CHART_SPEC:" in raw:
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

        return {"reply": reply_text, "chartSpec": chart_spec}

    except Exception as e:
        print(f"[Chat] Error: {e}")
        return {
            "reply": "Sorry, I couldn't process that right now. Please try again.",
            "chartSpec": None,
        }