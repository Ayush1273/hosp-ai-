# Decisions & Migration Record

## Migration from Streamlit Prototype

The project has been migrated from the prototype in `~/Downloads/HOSPO AI COMMAND/` to a production-ready, unified web dashboard designed for deployment on **Render**.

### 1. Removal of Unwanted AI Template Bloat
- **Authentication & Database**: Completely removed Manus OAuth, session cookie verification, Drizzle ORM, MySQL dependencies, and AWS S3 SDK. The tool is an administrative operations huddle dashboard that requires no login gate.
- **Extraneous Tabs**: Removed the fake "History" view (which rendered a simulated SVG sine-wave) and separate "Safety & scope" tab. A prominent operational disclaimer banner is positioned at the top of the dashboard, and clinical notices are in the footer.
- **Hallucinated Input Fields**: Removed `hdu_available`, `general_available`, and `private_available` which were never part of the clinical model. Restored the exact 18 operational fields in a 4-column layout matching the original Streamlit app.

### 2. Native ML & RAG Pipeline (FastAPI)
- Loaded the real trained XGBoost model (`models/xgb_model.json`, 3.24 MB) utilizing the exact 39 engineered features (7 ICU occupancy lags: 1h, 3h, 6h, 12h, 24h, 48h, 72h; 3 admission lags; rolling windows: 6h, 12h, 24h; time features).
- Connected the deterministic operational risk engine (`calculate_operational_risk`) producing 0–10 score, risk classification (LOW / MODERATE / HIGH / CRITICAL), and triggered bottleneck alerts.
- Integrated SentenceTransformers (`all-MiniLM-L6-v2`) cosine similarity retrieval against `document_embeddings.npy` and `embedded_knowledge_base.csv` for top-5 policy retrieval.
- Grounded Google Gemini AI (`gemini-3.6-flash`) with the exact structured prompt to synthesize concise administrative briefings.

### 3. Unified Production Deployment for Render
- FastAPI serves both the `/api/*` routes and the static compiled React 19 / Tailwind CSS frontend (`dist/`).
- Added `render.yaml` for 1-click Blueprints deployment.
- Added `build.sh` automated build script for Render.
