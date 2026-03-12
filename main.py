from fastapi import FastAPI, UploadFile, File, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import io
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from core.extractor import extract_from_bytes
from core.blueprint import build_blueprint, detect_column_profile
from core.helpers import rows_to_objects
from core.chat import build_chat_response

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://headcount-ai.netlify.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "Dashboard Agent Running"}


# ── /get-sheets ────────────────────────────────────────────────────────────────
@app.post("/get-sheets")
async def get_sheets(file: UploadFile = File(...)):
    contents = await file.read()
    excel = pd.ExcelFile(io.BytesIO(contents))
    return {"sheets": excel.sheet_names}


# ── /extract-raw-table ─────────────────────────────────────────────────────────
@app.post("/extract-raw-table")
async def extract_raw_table(
    file: UploadFile = File(...),
    sheet_name: Optional[str] = Query(None),
):
    contents = await file.read()
    table_data, _, _ = extract_from_bytes(contents, sheet_name)
    return table_data


# ── /generate-dashboard-blueprint ─────────────────────────────────────────────
@app.post("/generate-dashboard-blueprint")
async def generate_dashboard_blueprint(payload: Dict[str, Any] = Body(...)):
    return build_blueprint(payload)


# ── /analyze-bytes (NEW — replaces 3 separate calls) ──────────────────────────
# Accepts xlsx bytes + optional sheet name
# Returns tableData + blueprint + allSheets in one shot
# Used by HomePage (Drive download) and DashboardPage (sheet switching)
@app.post("/analyze-bytes")
async def analyze_bytes(
    file: UploadFile = File(...),
    sheet_name: Optional[str] = Query(None),
):
    contents = await file.read()
    table_data, all_sheets, current_sheet = extract_from_bytes(contents, sheet_name)
    blueprint = build_blueprint(table_data)

    return {
        "tableData": table_data,
        "blueprint": blueprint,
        "allSheets": all_sheets,
        "currentSheet": current_sheet,
    }


# ── /chat ──────────────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    headers: List[str]
    rows: List[List]
    datasetSummary: Optional[str] = None
    currentChartState: Optional[dict] = None
    existingCharts: Optional[list] = None
    existingTables: Optional[list] = None

@app.post("/chat")
async def chat(payload: ChatRequest):
    return build_chat_response(
        messages=[m.dict() for m in payload.messages],
        headers=payload.headers,
        rows=payload.rows,
        dataset_summary=payload.datasetSummary,
        current_chart_state=payload.currentChartState,
        existing_charts=payload.existingCharts,
        existing_tables=payload.existingTables,
    )