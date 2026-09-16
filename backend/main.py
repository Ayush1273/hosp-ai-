"""
HOSP-AI COMMAND — Production Backend
FastAPI service executing the real XGBoost 24-hour occupancy forecast,
deterministic operational risk assessment, RAG policy retrieval, and
Google Gemini grounded decision support.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from xgboost import XGBRegressor
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

# ================================================================
# 1. PATHS & ENVIRONMENT CONFIGURATION
# ================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")

MODEL_PATH = BASE_DIR / "models" / "xgb_model.json"
DATA_PATH = BASE_DIR / "Operational data.xlsx"
RAG_DIR = BASE_DIR / "hosp_ai_embeddings"

# ================================================================
# 2. APPLICATION LIFECYCLE & CACHED RESOURCES
# ================================================================

app = FastAPI(
    title="HOSP-AI COMMAND API",
    description="Hospital Operational Decision Support System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Resources:
    model: Optional[XGBRegressor] = None
    feature_names: List[str] = []
    tail_history: Optional[pd.DataFrame] = None
    capacity_reference: float = 0.0
    latest_timestamp: pd.Timestamp = pd.Timestamp.now()
    rag_embeddings: Optional[np.ndarray] = None
    policy_df: Optional[pd.DataFrame] = None
    rag_text_column: str = "text"
    embedding_model: Optional[SentenceTransformer] = None
    gemini_client: Optional[genai.Client] = None


resources = Resources()


def load_all_resources():
    """Load model, historical baseline data, RAG embeddings, and clients once."""
    print("[HOSP-AI] Loading XGBoost model from:", MODEL_PATH)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"XGBoost model not found at {MODEL_PATH}")
    xgb = XGBRegressor()
    xgb.load_model(str(MODEL_PATH))
    resources.model = xgb
    resources.feature_names = list(xgb.feature_names_in_)

    print("[HOSP-AI] Loading historical operational data from:", DATA_PATH)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Operational data file not found at {DATA_PATH}")
    history_df = pd.read_excel(DATA_PATH, sheet_name="Hospital_Operational_Data")
    history_df["Timestamp"] = pd.to_datetime(history_df["Timestamp"])
    history_df = history_df.sort_values("Timestamp").reset_index(drop=True)

    # Cache maximum hospital capacity and tail window (100 rows covers all 72h lags)
    resources.capacity_reference = float(history_df["Total_Occupied_Beds"].max())
    resources.latest_timestamp = pd.to_datetime(history_df["Timestamp"]).max()
    resources.tail_history = history_df.tail(100).copy()

    print("[HOSP-AI] Loading RAG resources from:", RAG_DIR)
    if not RAG_DIR.exists():
        raise FileNotFoundError(f"RAG directory not found at {RAG_DIR}")

    emb_path = RAG_DIR / "document_embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings file not found at {emb_path}")
    resources.rag_embeddings = np.load(str(emb_path))

    csv_candidates = [
        RAG_DIR / "embedded_knowledge_base.csv",
        RAG_DIR / "knowledge_base.csv",
        RAG_DIR / "policy_documents.csv",
        RAG_DIR / "policies.csv",
    ]
    csv_path = next((p for p in csv_candidates if p.exists()), None)
    if not csv_path:
        raise FileNotFoundError("Policy CSV not found in RAG directory")
    policy_df = pd.read_csv(csv_path)
    resources.policy_df = policy_df

    text_candidates = [
        "text",
        "Text",
        "Policy_Content",
        "policy_content",
        "Content",
        "content",
        "document",
        "Document",
    ]
    text_col = next((c for c in text_candidates if c in policy_df.columns), None)
    if not text_col:
        raise ValueError(f"Could not identify text column in {csv_path}")
    resources.rag_text_column = text_col

    print("[HOSP-AI] Initializing SentenceTransformer (all-MiniLM-L6-v2)...")
    resources.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    if GEMINI_API_KEY:
        print("[HOSP-AI] Initializing Google GenAI Client with model:", GEMINI_MODEL)
        try:
            resources.gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        except Exception as e:
            print("[HOSP-AI] Warning: Failed to initialize Gemini Client:", e)
    else:
        print(
            "[HOSP-AI] Notice: GEMINI_API_KEY not set. Operating in fallback explanation mode."
        )

    print("[HOSP-AI] All resources successfully loaded and cached.")


@app.on_event("startup")
def startup_event():
    load_all_resources()


# ================================================================
# 3. PYDANTIC DATA CONTRACTS
# ================================================================


class OperationalInput(BaseModel):
    # Column 1
    er_arrivals: int = Field(default=42, ge=0, description="Emergency Room Arrivals")
    ambulance_arrivals: int = Field(
        default=11, ge=0, description="Ambulance Deliveries"
    )
    general_admissions: int = Field(
        default=30, ge=0, description="General Inpatient Admissions"
    )
    icu_admissions: int = Field(default=8, ge=0, description="Direct ICU Admissions")

    # Column 2
    icu_discharges: int = Field(default=5, ge=0, description="ICU Discharges")
    icu_transfers_in: int = Field(default=5, ge=0, description="ICU Transfers In")
    icu_transfers_out: int = Field(default=2, ge=0, description="ICU Transfers Out")
    icu_occupied_beds: int = Field(default=43, ge=0, description="ICU Occupied Beds")

    # Column 3
    icu_available_beds: int = Field(default=5, ge=0, description="ICU Available Beds")
    hdu_occupied_beds: int = Field(
        default=4, ge=0, description="High Dependency Unit Occupied Beds"
    )
    general_occupied_beds: int = Field(
        default=3, ge=0, description="General Ward Occupied Beds"
    )
    private_occupied_beds: int = Field(
        default=6, ge=0, description="Private Ward Occupied Beds"
    )

    # Column 4
    ventilators_in_use: int = Field(
        default=2, ge=0, description="Ventilators currently deployed"
    )
    ventilators_available: int = Field(
        default=1, ge=0, description="Ventilators in reserve/available"
    )
    icu_staff_available: int = Field(default=12, ge=0, description="ICU Staff on duty")
    staff_shortage: str = Field(
        default="YES", description="Staff shortage flag: YES or NO"
    )
    elective_surgeries: int = Field(
        default=3, ge=0, description="Scheduled elective surgeries"
    )


class PolicyResult(BaseModel):
    document: str
    category: str
    content: str
    similarity: float


class HospitalState(BaseModel):
    icu_beds_available: int
    ventilators_available: int
    icu_staff_available: int
    staff_shortage: str


class AnalysisResponse(BaseModel):
    current_occupancy: float
    predicted_occupancy_24h: float
    delta: float
    risk_score: float
    risk_level: str
    risk_components: Dict[str, int]
    risk_factors: List[str]
    positive_trends: List[str]
    forecast_change: float
    hospital_state: HospitalState
    retrieved_policies: List[PolicyResult]
    ai_response: str
    timestamp: str


# ================================================================
# 4. FEATURE ENGINEERING & ML FORECASTING (EXACT REPLICATION)
# ================================================================


def build_inference_dataframe(inp: OperationalInput) -> pd.DataFrame:
    """Combines historical context with current input to compute exact 39 features."""
    staff_shortage_flag = 1 if inp.staff_shortage.upper() == "YES" else 0
    total_icu_beds = inp.icu_occupied_beds + inp.icu_available_beds
    icu_occupancy_pct = (
        (inp.icu_occupied_beds / total_icu_beds * 100) if total_icu_beds > 0 else 0.0
    )

    total_occupied_beds = (
        inp.icu_occupied_beds
        + inp.hdu_occupied_beds
        + inp.general_occupied_beds
        + inp.private_occupied_beds
    )
    cap_ref = resources.capacity_reference if resources.capacity_reference > 0 else 1.0
    hospital_occupancy_pct = (total_occupied_beds / cap_ref) * 100

    new_timestamp = resources.latest_timestamp + pd.Timedelta(hours=1)

    current_row = pd.DataFrame(
        [
            {
                "Timestamp": new_timestamp,
                "ER_Arrivals": inp.er_arrivals,
                "Ambulance_Arrivals": inp.ambulance_arrivals,
                "General_Admissions": inp.general_admissions,
                "ICU_Admissions": inp.icu_admissions,
                "ICU_Discharges": inp.icu_discharges,
                "ICU_Transfers_In": inp.icu_transfers_in,
                "ICU_Transfers_Out": inp.icu_transfers_out,
                "ICU_Occupied_Beds": inp.icu_occupied_beds,
                "ICU_Available_Beds": inp.icu_available_beds,
                "ICU_Occupancy_Pct": icu_occupancy_pct,
                "HDU_Occupied_Beds": inp.hdu_occupied_beds,
                "General_Occupied_Beds": inp.general_occupied_beds,
                "Private_Occupied_Beds": inp.private_occupied_beds,
                "Total_Occupied_Beds": total_occupied_beds,
                "Hospital_Occupancy_Pct": hospital_occupancy_pct,
                "Ventilators_In_Use": inp.ventilators_in_use,
                "Ventilators_Available": inp.ventilators_available,
                "ICU_Staff_Available": inp.icu_staff_available,
                "Staff_Shortage_Flag": staff_shortage_flag,
                "Elective_Surgeries": inp.elective_surgeries,
            }
        ]
    )

    # Append to tail history to compute rolling & lag calculations
    combined = pd.concat([resources.tail_history, current_row], ignore_index=True)

    # Time features
    combined["Hour"] = combined["Timestamp"].dt.hour
    combined["DayOfWeek"] = combined["Timestamp"].dt.dayofweek
    combined["Month"] = combined["Timestamp"].dt.month
    combined["IsWeekend"] = (combined["DayOfWeek"] >= 5).astype(int)

    # ICU occupancy lag features
    for lag in [1, 3, 6, 12, 24, 48, 72]:
        combined[f"ICU_Occupancy_Lag_{lag}h"] = combined["ICU_Occupancy_Pct"].shift(lag)

    # ICU admission lag features
    for lag in [1, 6, 24]:
        combined[f"ICU_Admissions_Lag_{lag}h"] = combined["ICU_Admissions"].shift(lag)

    # Rolling features
    combined["ICU_Occupancy_Rolling_6h"] = (
        combined["ICU_Occupancy_Pct"].rolling(6).mean()
    )
    combined["ICU_Occupancy_Rolling_12h"] = (
        combined["ICU_Occupancy_Pct"].rolling(12).mean()
    )
    combined["ICU_Occupancy_Rolling_24h"] = (
        combined["ICU_Occupancy_Pct"].rolling(24).mean()
    )
    combined["ER_Arrivals_Rolling_6h"] = combined["ER_Arrivals"].rolling(6).mean()
    combined["ICU_Admissions_Rolling_24h"] = (
        combined["ICU_Admissions"].rolling(24).mean()
    )

    feature_matrix = combined.tail(1)[resources.feature_names].copy()
    return feature_matrix, icu_occupancy_pct


# ================================================================
# 5. DETERMINISTIC RISK ENGINE (EXACT REPLICATION)
# ================================================================


def calculate_operational_risk(
    current_icu_occupancy: float,
    predicted_icu_occupancy: float,
    icu_available_beds: int,
    staff_shortage_flag: int,
    ventilators_available: int,
    icu_admissions: int,
    icu_discharges: int,
    er_arrivals: int,
    ambulance_arrivals: int,
) -> Dict[str, Any]:
    components = {}
    risk_factors = []
    positive_trends = []

    # Current ICU occupancy
    if current_icu_occupancy >= 90:
        occupancy_score = 3
        risk_factors.append("Critical ICU occupancy")
    elif current_icu_occupancy >= 80:
        occupancy_score = 2
        risk_factors.append("High ICU occupancy")
    elif current_icu_occupancy >= 70:
        occupancy_score = 1
    else:
        occupancy_score = 0
    components["Current ICU Occupancy"] = occupancy_score

    # Forecast trend
    forecast_change = predicted_icu_occupancy - current_icu_occupancy
    if forecast_change >= 10:
        forecast_score = 2
    elif forecast_change >= 3:
        forecast_score = 1
    elif forecast_change <= -20:
        forecast_score = -1
        positive_trends.append(
            "Forecast indicates a substantial decrease in ICU occupancy"
        )
    else:
        forecast_score = 0
    components["24h Forecast Trend"] = forecast_score

    # ICU beds
    if icu_available_beds <= 2:
        bed_score = 2
        risk_factors.append("Very limited ICU bed availability")
    elif icu_available_beds <= 5:
        bed_score = 1
        risk_factors.append("Limited ICU bed availability")
    else:
        bed_score = 0
    components["ICU Bed Availability"] = bed_score

    # Staffing
    if staff_shortage_flag == 1:
        staff_score = 1
        risk_factors.append("ICU staffing shortage is active")
    else:
        staff_score = 0
    components["Staff Shortage"] = staff_score

    # Ventilators
    if ventilators_available <= 1:
        ventilator_score = 2
        risk_factors.append("Critical ventilator availability")
    elif ventilators_available <= 4:
        ventilator_score = 1
        risk_factors.append("Limited ventilator availability")
    else:
        ventilator_score = 0
    components["Ventilator Availability"] = ventilator_score

    # Patient flow
    net_flow = icu_admissions - icu_discharges
    if net_flow >= 3:
        flow_score = 2
        risk_factors.append("Strong positive ICU patient-flow pressure")
    elif net_flow > 0:
        flow_score = 1
        risk_factors.append("Positive ICU patient-flow pressure")
    else:
        flow_score = 0
    components["ICU Patient Flow"] = flow_score

    # Emergency inflow
    if er_arrivals >= 50 or ambulance_arrivals >= 15:
        emergency_score = 2
        risk_factors.append("Very high emergency patient inflow")
    elif er_arrivals >= 40 or ambulance_arrivals >= 10:
        emergency_score = 1
        risk_factors.append("High emergency patient inflow")
    else:
        emergency_score = 0
    components["Emergency Inflow"] = emergency_score

    raw_score = sum(components.values())
    risk_score = round(min((raw_score / 14) * 10, 10), 1)

    if risk_score >= 8:
        risk_level = "CRITICAL"
    elif risk_score >= 6:
        risk_level = "HIGH"
    elif risk_score >= 3:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "raw_score": raw_score,
        "risk_components": components,
        "risk_factors": risk_factors,
        "positive_trends": positive_trends,
        "forecast_change": round(forecast_change, 2),
    }


# ================================================================
# 6. RAG RETRIEVAL (POLICY DOCUMENTS)
# ================================================================


def retrieve_policies(query: str, top_k: int = 5) -> List[PolicyResult]:
    """Retrieve top-k relevant hospital policies using sentence-transformers cosine similarity."""
    if resources.embedding_model is None or resources.rag_embeddings is None:
        return []

    query_embedding = resources.embedding_model.encode([query], convert_to_numpy=True)
    similarity_scores = cosine_similarity(query_embedding, resources.rag_embeddings)[0]
    top_indices = np.argsort(similarity_scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        row = resources.policy_df.iloc[idx]
        title = row.get(
            "document_title", row.get("Document", row.get("title", "Hospital Policy"))
        )
        category = row.get("category", row.get("Category", "OPERATIONAL"))
        content = str(row[resources.rag_text_column])
        sim = float(similarity_scores[idx])
        results.append(
            PolicyResult(
                document=str(title),
                category=str(category),
                content=content,
                similarity=round(sim, 4),
            )
        )
    return results


# ================================================================
# 7. AI DECISION SUPPORT (GEMINI SYNTHESIS)
# ================================================================


def build_grounded_prompt(
    risk_context: str,
    current_icu_occupancy: float,
    predicted_icu_occupancy: float,
    retrieved_policies: List[PolicyResult],
) -> str:
    policy_blocks = []
    for i, p in enumerate(retrieved_policies, start=1):
        policy_blocks.append(
            f"--- POLICY {i} ---\n"
            f"Document: {p.document}\n"
            f"Category: {p.category}\n"
            f"Similarity Score: {p.similarity}\n"
            f"Policy Content: {p.content}\n"
        )
    policy_context = "\n".join(policy_blocks)

    return f"""You are the explanation and recommendation layer of HOSP-AI COMMAND, an administrative hospital operational decision-support prototype.

IMPORTANT ARCHITECTURE RULES:
1. The XGBoost model produces the numerical 24-hour ICU occupancy forecast.
2. The deterministic Operational Risk Engine produces the numerical risk score and risk level.
3. Retrieved hospital policies provide the operational knowledge context for recommendations.
4. You MUST NOT invent, change, or recalculate the numerical risk score.
5. You MUST NOT present this system as a clinical decision-making system.
6. Recommendations must be operational and grounded in the retrieved policies.
7. Do not invent hospital policies that are not present in the retrieved policy context.
8. Clearly distinguish current operational pressure from the 24-hour forecast.
9. Human review is required before any real operational action.

------------------------------------------------------------
CURRENT HOSPITAL OBSERVATION
------------------------------------------------------------
Current ICU Occupancy: {current_icu_occupancy:.2f}%
Predicted 24-Hour ICU Occupancy: {predicted_icu_occupancy:.2f}%

{risk_context}

------------------------------------------------------------
RETRIEVED HOSPITAL POLICY CONTEXT
------------------------------------------------------------
{policy_context}

------------------------------------------------------------
TASK
------------------------------------------------------------
Generate a concise administrative operational decision-support summary using ONLY this structure:

### 1. Operational Situation
Briefly explain the current operational situation using the observed values.

### 2. Predicted Risk
Report the exact ML forecast ({predicted_icu_occupancy:.2f}%), exact deterministic risk score, and exact deterministic risk level.

### 3. Key Risk Factors
List the important operational risk factors.

### 4. Recommended Actions
Provide practical administrative recommendations grounded in the retrieved hospital policies.

### 5. Forecast Trend
Briefly explain whether the 24-hour forecast indicates increasing, stable, or decreasing ICU occupancy.

Do NOT include:
- Supporting Policies section
- Technical model math or tensor details
- Clinical treatment recommendations
- Long limitation sections

Keep the response concise and formatted cleanly in markdown suitable for a hospital operations dashboard."""


def generate_ai_decision_support(
    risk_result: Dict[str, Any],
    current_icu_occupancy: float,
    predicted_icu_occupancy: float,
    retrieved_policies: List[PolicyResult],
    inp: OperationalInput,
) -> str:
    risk_factors_text = (
        "\n".join([f"- {f}" for f in risk_result["risk_factors"]])
        or "- No major operational risk factors detected."
    )
    risk_context = f"""OPERATIONAL RISK ASSESSMENT
Risk Score: {risk_result["risk_score"]} / 10
Risk Level: {risk_result["risk_level"]}
Current ICU Occupancy: {current_icu_occupancy:.2f}%
Predicted 24-Hour ICU Occupancy: {predicted_icu_occupancy:.2f}%
ICU Beds Available: {inp.icu_available_beds}
Ventilators Available: {inp.ventilators_available}
ICU Staff Available: {inp.icu_staff_available}
ICU Staff Shortage: {inp.staff_shortage}
ICU Admissions: {inp.icu_admissions}
ICU Discharges: {inp.icu_discharges}
Emergency Department Arrivals: {inp.er_arrivals}
Ambulance Arrivals: {inp.ambulance_arrivals}

Key Risk Factors:
{risk_factors_text}"""

    prompt = build_grounded_prompt(
        risk_context, current_icu_occupancy, predicted_icu_occupancy, retrieved_policies
    )

    if resources.gemini_client:
        try:
            response = resources.gemini_client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt
            )
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            print("[HOSP-AI] Gemini API generation error:", e)

    # Deterministic fallback explanation if Gemini is unavailable
    forecast_dir = (
        "increasing"
        if risk_result["forecast_change"] > 0
        else "decreasing"
        if risk_result["forecast_change"] < 0
        else "stable"
    )
    return f"""### 1. Operational Situation
The ICU is operating at {current_icu_occupancy:.2f}% capacity with {inp.icu_available_beds} bed(s) available. Emergency intake is {inp.er_arrivals} walk-ins and {inp.ambulance_arrivals} ambulance arrivals.

### 2. Predicted Risk
The XGBoost model forecasts 24-hour ICU occupancy at {predicted_icu_occupancy:.2f}%. The deterministic operational risk score is **{risk_result["risk_score"]} / 10** ({risk_result["risk_level"]}).

### 3. Key Risk Factors
{risk_factors_text}

### 4. Recommended Actions
- **Bed Management**: Review patient transfer readiness and coordinate with step-down units.
- **Resource Allocation**: Protect minimum ventilator reserve ({inp.ventilators_available} currently available).
- **Staffing Protocol**: {"Activate nursing escalation protocol due to active staffing shortage." if inp.staff_shortage.upper() == "YES" else "Maintain scheduled shift allocations."}
- Grounded in retrieved hospital operational SOPs.

### 5. Forecast Trend
The 24-hour predictive trajectory indicates a **{forecast_dir}** trend ({risk_result["forecast_change"]:+.2f} percentage points)."""


# ================================================================
# 8. API ENDPOINTS
# ================================================================


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "HOSP-AI COMMAND",
        "model_loaded": resources.model is not None,
        "rag_loaded": resources.rag_embeddings is not None,
        "gemini_connected": resources.gemini_client is not None,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/default-inputs")
def get_default_inputs():
    return OperationalInput().dict()


@app.post("/api/analyze", response_model=AnalysisResponse)
def analyze_hospital_conditions(inp: OperationalInput):
    if resources.model is None or resources.tail_history is None:
        raise HTTPException(status_code=503, detail="Operational models are not ready.")

    # 1. Feature engineering
    feature_matrix, current_icu_occupancy = build_inference_dataframe(inp)

    # 2. XGBoost 24-hour forecast
    raw_pred = float(resources.model.predict(feature_matrix)[0])
    predicted_icu_occupancy = float(np.clip(raw_pred, 0, 100))

    # 3. Deterministic risk engine
    staff_shortage_flag = 1 if inp.staff_shortage.upper() == "YES" else 0
    risk_result = calculate_operational_risk(
        current_icu_occupancy=current_icu_occupancy,
        predicted_icu_occupancy=predicted_icu_occupancy,
        icu_available_beds=inp.icu_available_beds,
        staff_shortage_flag=staff_shortage_flag,
        ventilators_available=inp.ventilators_available,
        icu_admissions=inp.icu_admissions,
        icu_discharges=inp.icu_discharges,
        er_arrivals=inp.er_arrivals,
        ambulance_arrivals=inp.ambulance_arrivals,
    )

    # 4. RAG policy retrieval
    rag_query = f"""Hospital operational decision support.
Current ICU occupancy: {current_icu_occupancy:.2f}%
Predicted 24-hour ICU occupancy: {predicted_icu_occupancy:.2f}%
Risk level: {risk_result["risk_level"]}
Risk score: {risk_result["risk_score"]}/10
ICU beds available: {inp.icu_available_beds}
Ventilators available: {inp.ventilators_available}
Staff shortage: {inp.staff_shortage}
ICU admissions: {inp.icu_admissions}
ICU discharges: {inp.icu_discharges}
Emergency arrivals: {inp.er_arrivals}
Ambulance arrivals: {inp.ambulance_arrivals}
Find hospital policies relevant to ICU capacity, staffing, bed management, emergency inflow, patient flow and operational escalation."""

    retrieved_policies = retrieve_policies(rag_query, top_k=5)

    # 5. Gemini grounded decision support
    ai_response = generate_ai_decision_support(
        risk_result,
        current_icu_occupancy,
        predicted_icu_occupancy,
        retrieved_policies,
        inp,
    )

    delta = round(predicted_icu_occupancy - current_icu_occupancy, 2)

    return AnalysisResponse(
        current_occupancy=round(current_icu_occupancy, 2),
        predicted_occupancy_24h=round(predicted_icu_occupancy, 2),
        delta=delta,
        risk_score=risk_result["risk_score"],
        risk_level=risk_result["risk_level"],
        risk_components=risk_result["risk_components"],
        risk_factors=risk_result["risk_factors"],
        positive_trends=risk_result["positive_trends"],
        forecast_change=risk_result["forecast_change"],
        hospital_state=HospitalState(
            icu_beds_available=inp.icu_available_beds,
            ventilators_available=inp.ventilators_available,
            icu_staff_available=inp.icu_staff_available,
            staff_shortage=inp.staff_shortage,
        ),
        retrieved_policies=retrieved_policies,
        ai_response=ai_response,
        timestamp=datetime.utcnow().isoformat(),
    )


# ================================================================
# 9. STATIC ASSETS & SPA ROUTING (FOR RENDER DEPLOYMENT)
# ================================================================

# Serve Vite build in production (dist directory)
dist_path = BASE_DIR / "dist"
if dist_path.exists():
    if (dist_path / "assets").exists():
        app.mount(
            "/assets",
            StaticFiles(directory=str(dist_path / "assets")),
            name="static-assets",
        )

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Allow /api routes to be handled by FastAPI
        if full_path.startswith("api"):
            raise HTTPException(status_code=404, detail="API endpoint not found")
        file_path = dist_path / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(dist_path / "index.html")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=True)
