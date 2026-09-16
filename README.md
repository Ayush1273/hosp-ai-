# HOSP-AI COMMAND

**AI-Powered Hospital Operational Decision Support Dashboard**

HOSP-AI COMMAND is a real-time hospital operational decision-support system designed to help hospital administrators, medical directors, and bed management coordinators monitor patient flow, forecast ICU occupancy 24 hours in advance, evaluate composite operational risk, and receive policy-grounded recommendations.

---

## System Architecture

```text
18 Hospital Operational Inputs
         ↓
Feature Engineering (39 Features: Lags 1-72h, Rolling 6-24h, Inflow Dynamics)
         ↓
XGBoost ML Model (models/xgb_model.json)
         ↓
24-Hour ICU Occupancy Forecast
         ↓
Deterministic Operational Risk Engine (0–10 Score + Severity Classification)
         ↓
RAG Policy Retrieval (all-MiniLM-L6-v2 + hosp_ai_embeddings/)
         ↓
Google Gemini AI (gemini-3.6-flash)
         ↓
Policy-Grounded Administrative Recommendations
         ↓
Modern Web Command Dashboard (React 19, Tailwind CSS v4)
```

> **Clinical Scope Disclaimer:** HOSP-AI COMMAND is an administrative resource-management tool (beds, ventilators, staffing, and inflow). It does not provide clinical diagnosis, triage, or individual patient treatment guidance. AI recommendations are advisory; a qualified human administrator reviews every signal.

---

## Key Features

- **24-Hour Predictive Census**: High-accuracy XGBoost regression forecasting ICU bed pressure with 39 engineered features.
- **Deterministic Operational Risk Engine**: Transparent arithmetic risk scoring (0–10) and classification (`LOW`, `MODERATE`, `HIGH`, `CRITICAL`) that cannot be altered or hallucinated by the LLM.
- **Retrieval-Augmented Generation (RAG)**: Cosine similarity semantic search over synthetic hospital SOPs using SentenceTransformers.
- **AI-Synthesized Decision Support**: Google Gemini generates concise operational summaries, key risk factors, recommended actions, and forecast trends grounded in retrieved policies.
- **Modern Clinical Web Interface**: Real-time interactive control panel with exact 18 operational fields, ECG animation, capacity outlook metrics, and policy accordion.

---

## Project Structure

```text
hosp-ai-command/
├── backend/
│   └── main.py              # FastAPI server (ML inference, RAG, Gemini, static serving)
├── models/
│   └── xgb_model.json       # Trained XGBoost regression model (3.24 MB)
├── hosp_ai_embeddings/
│   ├── document_embeddings.npy       # Precomputed 384-d policy embeddings
│   └── embedded_knowledge_base.csv   # Hospital SOP knowledge base
├── Operational data.xlsx    # Historical hospital operational baseline dataset
├── client/                  # React 19 + Tailwind CSS frontend source
│   ├── src/
│   │   ├── pages/Home.tsx   # Command dashboard interface
│   │   └── index.css        # Clinical glassmorphism theme & ECG animation
│   └── index.html
├── dist/                    # Compiled production frontend (built via Vite)
├── render.yaml              # Render Blueprint deployment configuration
├── build.sh                 # Render build script
├── requirements.txt         # Root Python requirements
├── package.json             # Frontend dependencies & scripts
└── .env                     # Environment variables (GEMINI_API_KEY, GEMINI_MODEL)
```

---

## Local Development

### 1. Prerequisites
- Python 3.10+ (Python 3.11 / 3.12 / 3.13 supported)
- Node.js 18+ and npm

### 2. Environment Configuration
Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.6-flash
PORT=8000
```

### 3. Setup Python Backend
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python requirements
pip install -r requirements.txt

# Run FastAPI backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Setup Frontend (Dev Mode)
In a separate terminal:
```bash
# Install frontend dependencies
npm install

# Start Vite dev server with proxy to backend
npm run dev
```
Open **http://localhost:5173** in your browser.

---

## Deploying on Render

This repository is pre-configured for deployment as a single **Python Web Service** on [Render](https://render.com).

### Option A: Using Render Blueprints (`render.yaml`)
1. Push your repository to GitHub or GitLab.
2. In Render Dashboard, click **New +** → **Blueprint**.
3. Connect your repository. Render will automatically detect `render.yaml`.
4. Add your `GEMINI_API_KEY` under Environment Variables.
5. Click **Apply**.

### Option B: Manual Web Service Setup
1. In Render Dashboard, click **New +** → **Web Service**.
2. Connect your repository.
3. Configure settings:
   - **Name:** `hosp-ai-command`
   - **Environment:** `Python`
   - **Build Command:** `./build.sh` (or `npm install --legacy-peer-deps && npm run build && pip install -r requirements.txt`)
   - **Start Command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
4. Under **Environment Variables**, add:
   - `GEMINI_API_KEY`: `<Your Google Gemini API Key>`
   - `GEMINI_MODEL`: `gemini-3.6-flash`
   - `PYTHON_VERSION`: `3.11.9` (or `3.12`)
5. Click **Deploy Web Service**.

Once deployed, FastAPI serves both the REST API endpoints (`/api/analyze`, `/api/health`) and the bundled React SPA from `/`.
