# TerraMind AI — AI-Powered Agriculture Crop Advisory Assistant

*Smarter crop decisions. Evidence-driven farming.*

## Project Overview
TerraMind AI answers natural-language questions about crop choice by retrieving **similar historical farming cases** (hybrid BM25 + semantic search with metadata filters), turning them into **evidence-backed crop recommendations**, and adding a trained **yield estimate**, a rule-based **suitability risk**, **soil/weather/input impact analysis** and an **LLM-as-judge** validation. It is decision support: it never promises yields, and it keeps *observed* history, *retrieved* evidence, *model predictions*, *rule-based assessments*, *LLM explanations* and *judge verdicts* visibly separate.

## Problem Statement
Agricultural teams accumulate thousands of records (crop, season, state, soil, weather, inputs, yield) but cannot quickly find comparable cases or see the evidence behind a crop choice.

## Business Objective
Let advisors and farmers ask in plain language and get evidence-backed, auditable crop recommendations. TerraMind lets them ask, e.g. *"Which crops perform well in Punjab during Rabi with pH 7.8 and ~600 mm rainfall?"* and see the recommended crops, the exact historical cases behind each, risks, missing information and measured evaluation results.

## Honest status (read this first)
| Item | Status |
|---|---|
| Dataset | **55,000 records = 19,689 real + 35,311 labelled synthetic.** The Kaggle file only has 19,689 real rows, so at the owner's request the rest are generated variants of real records (`is_synthetic=true`, ids `TS-…`, "synthetic" badge in the UI, filter/toggle to exclude). Models, statistics, suitability envelopes and the yield-model metrics use **real rows only**. `python scripts/prepare_dataset.py --no-augment` gives a real-only dataset. |
| Soil / weather | Soil N, P, K, pH are **one value per state** (30 states); weather is a **state-year** aggregate. Yield units are crop-specific. |
| Backend, retrieval, yield model, agents, API, React UI, feedback, evaluation pipeline | Implemented and verified (56 automated tests pass; UI walked through in a headless browser). |
| Claude features (classification enrichment, explanations, judge, DeepEval) | Implemented and verified working with `claude-sonnet-5-5` in manual runs. **The recorded evaluation runs have DeepEval and judge metrics as “Not evaluated”** because the Anthropic account ran out of credit when the evaluation executed. Add credit and re-run `python scripts/run_evaluation.py` to populate them. Nothing was estimated or filled in. |
| Suitability risk | Rule-based heuristic (no suitability labels exist). Not a trained classifier, not a probability. |
| Feedback loop | Storage, summary and review-candidate workflow. **No automatic retraining.** |

## Solution Summary
Dataset -> validation/merge -> one case per row -> MiniLM embeddings -> FAISS + BM25 -> metadata filters -> RRF fusion -> prioritisation -> 7-agent pipeline (classifier, retrieval, recommendation, suitability, yield, impact, validation) -> FastAPI -> React UI. Details below.

## Architecture
See `architecture/Architecture_Diagram.pdf` (and `.png`). Flow: Dataset → validation/merge/derive → case records + text → MiniLM embeddings → FAISS + BM25 → metadata filtering → RRF fusion → prioritisation → classification → recommendation → suitability / yield / impact (parallel) → explanation → judge → FastAPI → React; plus an evaluation pipeline and a feedback loop.

**Retrieval.** One CSV row = one case = one chunk. Filters (crop, season, state, year, inclusive numeric ranges) are applied first; BM25 and semantic rankings are fused with Reciprocal Rank Fusion (`Σ 1/(60+rank)`, a ranking score — *not* a probability). One backend failing returns the other's results with a fallback notice; both failing is a 503. Embeddings are weak on numbers, so the retrieval agent also adds the cases whose soil/weather values are numerically closest. **Prioritisation** weights: hybrid relevance 0.30, crop 0.15, season 0.10, region 0.10, soil similarity 0.15, weather similarity 0.10, completeness 0.10; unassessed components are excluded, never treated as matches; historical yield is *not* a component.

**Agents** (orchestrator in `src/agents/orchestrator.py`): 1 Condition Classifier (regex + Claude, per-field provenance) · 2 Case Retrieval · 3 Crop Recommendation · 4 Suitability & Risk · 5 Yield Prediction · 6 Impact Analysis · 7 Recommendation Validation (deterministic checks + Claude judge). Stages 4–6 run in parallel with timeouts; a failing optional agent is reported and the rest are kept.

## Technology Stack
| Item | Choice |
|---|---|
| Programming Language | Python 3.11 (backend), JavaScript/React 18 (frontend) |
| LLM Used | Anthropic Claude (`claude-sonnet-5-5`, configurable via `LLM_MODEL`) for query enrichment, explanations and LLM-as-judge |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Database | FAISS (`artifacts/indexes/cases.faiss`) + BM25 (`rank-bm25`) for hybrid search |
| Frameworks Used | FastAPI, Uvicorn, Pydantic, scikit-learn, pandas/NumPy, DeepEval, pytest, React, Vite, Tailwind CSS, Recharts |

Full list:
Python 3.11 · FastAPI · Pydantic · pandas/NumPy · scikit-learn (HistGradientBoosting, RandomForest) · sentence-transformers (all-MiniLM-L6-v2) · FAISS · rank-bm25 · Anthropic SDK · DeepEval · pytest · React 18 · Vite · Tailwind CSS 4 · React Router · Recharts · Lucide. Versions are in `requirements.txt` / `frontend/package.json`.

## Installation Instructions
### Environment Setup, Dependency Installation and API Key Configuration (Windows, PowerShell, VS Code)
Steps 1-2 create the environment and install dependencies; step 2 also sets `ANTHROPIC_API_KEY` in `.env` (never commit it). Steps 5-6 are the local deployment.
```powershell
# 1. backend environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. configuration (backend only; never commit .env)
Copy-Item .env.example .env      # then edit .env: ANTHROPIC_API_KEY and LLM_MODEL
```
`.env` variables: `ANTHROPIC_API_KEY`, `LLM_PROVIDER=anthropic`, `LLM_MODEL` (default `claude-sonnet-5-5`, any model your key can use), `DATASET_PATH`, `DATABASE_URL`, `API_HOST`, `API_PORT`. The key is read only by the backend and never returned by the API or exposed to the frontend. Without a key the dataset, retrieval, filters, prioritisation, yield prediction, suitability, impact analysis, feedback and retrieval metrics all still work; query enrichment, explanations, the judge and DeepEval need it.

```powershell
# 3. dataset (public CC0 file; no Kaggle login needed)
New-Item -ItemType Directory -Force data\raw | Out-Null
Invoke-WebRequest "https://www.kaggle.com/api/v1/datasets/download/anshumish/crop-yield-data-with-soil-and-weather-dataset" -OutFile data\raw\kaggle.zip
Expand-Archive data\raw\kaggle.zip data\raw -Force
#    (alternative: download manually from the Kaggle page and put crop_yield.csv, state_soil_data.csv,
#     state_weather_data_1997_2020.csv in data\raw\)

# 4. prepare working dataset + data-quality report, build index (~6 min on CPU), train model
python scripts/prepare_dataset.py
python scripts/build_index.py
python scripts/train_yield_model.py

# 5. run the API  (docs at http://127.0.0.1:8000/docs)
uvicorn src.main:app --reload

# 6. run the frontend (new terminal)
cd frontend
npm install
npm run dev                       # http://localhost:5173  (VITE_API_BASE_URL in frontend/.env)

# 7. tests / evaluation / documents
python -m pytest -q
python scripts/run_evaluation.py              # add --no-llm to skip Claude-dependent metrics
python scripts/build_docs.py                  # PDFs from real artifacts
python scripts/build_presentation.py
```

## API (all under `/api`; full reference in `docs/API_Documentation.pdf`, live at `/docs`)
Health/stats: `GET /health`, `/api/health`, `/api/system/status`, `/api/stats` · Advisory: `POST /api/advisory/query`, `/classify`, `/recommend`, `POST /advisory/jobs` + `GET /advisory/jobs/{id}` (real stage progress), `GET /advisory/recent`, `/advisory/queries/{id}` · Cases: `GET /api/cases`, `/cases/{id}`, `POST /cases/search`, `GET /metadata/options` · `POST /api/predictions/yield`, `/predictions/suitability`, `/analysis/impact` · Evaluation: `GET /api/evaluations`, `/evaluations/runs`, `/runs/{id}`, `/test-cases`, `POST /evaluations/run` · Feedback: `POST/GET /api/feedback`, `GET /feedback/summary`.

## Local Deployment Instructions
Run `uvicorn src.main:app --reload` (API, http://127.0.0.1:8000/docs) and, in `frontend/`, `npm run dev` (UI, http://localhost:5173) as in steps 5-6 above.

## Sample Queries
Real outputs (run without the LLM; full list in `docs/Sample_Test_Cases.md`).

| # | Query | Expected output |
|---|---|---|
| 1 | *Which crops perform well in Punjab during the Rabi season with soil pH 7.8, rainfall around 600 mm and nitrogen 150?* | #1 Wheat (score 0.936, 23 supporting cases, historical median yield 4.50, predicted 4.69, risk low); #2 Gram; #3 Barley |
| 2 | *Suggest alternatives to rice for Kharif in West Bengal with pH 6.2.* | #1 Other Kharif pulses (0.972, median 0.55, risk low); #2 Jute; #3 Sunflower |
| 3 | *I want to grow Coconut in Delhi. Is it a good choice?* | Warning that no historical Coconut records match Delhi, so it is not recommended on evidence; data-backed alternatives: Rice, Bajra, Maize |
| 4 | *What should I plant?* | Status `needs_clarification`: asks for at least a state or season, or soil/weather values (e.g. 'Kharif in Assam') |

### Sample query (API call)
```powershell
curl -X POST http://127.0.0.1:8000/api/advisory/query -H "Content-Type: application/json" `
  -d '{"query":"Which crops perform well in Punjab during the Rabi season with soil pH 7.8, rainfall around 600 mm and nitrogen 150?"}'
```
Returns (abridged; real outputs in `docs/Sample_Test_Cases.md`): `conditions` (each field `{value, provenance}`), `retrieval` (filters, filtered-out counts, backends), `cases[]` (ranks, scores, priority breakdown), `recommendations[]` — e.g. **#1 Wheat**: heuristic score 0.936, 23 supporting cases (e.g. TM-4F967CC8E7), *historical* median yield 4.50 (actual), *predicted* 4.69 (model estimate, 10–90% range 2.15–5.23), risk low (rule-based), judge verdict — plus `impact_analysis`, `agents[]`/`stages[]` status, `judge`, `warnings`, `metadata`.

## Models and evaluation
* **Yield model** (`python scripts/train_yield_model.py`): `log1p(yield)`, no `production` feature (leakage), temporal split (train ≤2014 / validation 2015–17 / test 2018–20), baseline vs HGB vs RF, selection on validation log-RMSE, refit on train+validation. Held-out test (full features): MAE 10.05, RMSE 151.2, R² 0.968 (crop-median baseline R² 0.836); log-scale R² 0.956. Original-scale numbers are dominated by a few high-magnitude crops. 10–90% interval covers 70% on test (nominal 80%): slightly under-covering.
* **Evaluation** (`python scripts/run_evaluation.py`): 31 verified cases (16 constraint-retrieval, 8 recommendation, 4 abstention, 3 clarification). Relevance labels are deterministic dataset filters, re-verified every run. Latest recorded run: Precision@10 0.806 (capped by only 3–4 relevant cases in some groups), Recall@10 0.750, MRR 1.000, Hit@10 1.000, 31/31 passed, constraint satisfaction 1.00; DeepEval / judge: **Not evaluated** (see status). Report: `docs/Evaluation_Report.pdf`. Note: retrieval pass/fail uses R-precision; the first run exposed that a Precision@10 ≥ 0.5 rule is impossible when <5 relevant cases exist (disclosed in the report).
* **Feedback → improvement**: submit feedback in the UI → `GET /api/feedback/summary` lists review candidates → verify by hand → add to `data/evaluation/test_cases_extra.json` → re-run evaluation and compare runs.

## Deliverables
`architecture/Architecture_Diagram.pdf` · `design/Design_Document.pdf` · `docs/API_Documentation.pdf` · `docs/Dataset_Details.pdf` · `docs/Evaluation_Report.pdf` · `presentation/Project_Presentation.pptx` · `docs/Sample_Test_Cases.md` · `docs/screenshots/`.

## Project structure
`src/` (`main.py`, `config.py`, `api/`, `agents/`, `rag/`, `models/`, `services/`, `schemas/`, `evaluation/`, `utils/`) · `frontend/` · `data/{raw,processed,sample_data,evaluation}` · `artifacts/{indexes,models,evaluation}` · `scripts/` · `tests/`.

## Limitations
Only 19,689 real records (the other 35,311 are synthetic variants that add retrieval volume, not new evidence); state-level soil and state-year weather limit how well conditions describe a single farm; yield units are crop-specific (compare within a crop); recommendations show where crops were historically recorded under similar conditions, not agronomic prescriptions; correlations and feature importances are associations, not causal effects; interval under-coverage; LLM features depend on API availability and credit.

## Troubleshooting
* **503 “index not built / model not found / dataset not found”** — run steps 3–4 above; `/api/system/status` shows which artifact is missing.
* **“credit balance is too low” / LLM warnings** — add Anthropic credit; the app keeps working without the LLM (template explanations, deterministic validation).
* **Model not available (404)** — set `LLM_MODEL` in `.env` to a model your key can use.
* **Frontend “Backend unreachable”** — start uvicorn; check `VITE_API_BASE_URL` in `frontend/.env`; CORS allows `localhost:5173`.
* **First request slow** — the embedding model loads once (~15 s).
* **Hugging Face symlink warning on Windows** — harmless; set `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.

## Audit, experiments and honest results (added)
Full methodology, tables and limitations: `docs/experiments/Experiments_Report.md`. Reproduce:
```powershell
python scripts/audit_data.py                 # dataset facts -> docs/experiments/data_audit.json
python scripts/run_experiments.py            # regression, classification, retrieval, recommendation (~25 min CPU)
python scripts/build_experiment_report.py    # -> docs/experiments/Experiments_Report.md
python scripts/check_llm.py                  # confirms Claude + DeepEval work once API credit exists
python -m pytest -q
```
Key points: experiments use Kaggle-sourced rows with a temporal split (train <=2014, validation 2015-17, test >=2018); synthetic rows used in experiments come from generator v2 built from training rows only; every record has `data_source` (`kaggle`/`synthetic`). Persistence baselines are strong, synthetic augmentation does not improve real-test metrics, and Claude/DeepEval metrics are reported as not evaluated until the API has credit.
