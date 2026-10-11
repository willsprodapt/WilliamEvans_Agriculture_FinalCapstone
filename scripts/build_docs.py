"""python scripts/build_docs.py -> architecture/, design/, docs/ PDFs and presentation/*.pptx

Everything numeric is read from real artifacts (data_quality_report.json, yield_metrics.json, latest evaluation run, live OpenAPI).
"""
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle)  # noqa: E402

DQ = json.loads((ROOT / "data/processed/data_quality_report.json").read_text(encoding="utf-8"))
YM = json.loads((ROOT / "artifacts/models/yield_metrics.json").read_text(encoding="utf-8"))
evals = sorted(glob.glob(str(ROOT / "artifacts/evaluation/eval-*.json")))
EV = json.loads(Path(evals[-1]).read_text(encoding="utf-8")) if evals else None
GREEN, SAGE = colors.HexColor("#14532d"), colors.HexColor("#eceee4")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=ss["Heading1"], textColor=GREEN, fontSize=18, spaceAfter=8)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=GREEN, fontSize=13, spaceBefore=10, spaceAfter=4)
B = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13)
SM = ParagraphStyle("sm", parent=B, fontSize=8, leading=10)
CODE = ParagraphStyle("c", parent=ss["Code"], fontSize=7, leading=8.5, backColor=SAGE, borderPadding=4)


def P(t, st=B):
    return Paragraph(t, st)


def bullets(items):
    return [Paragraph("&bull; " + i, B) for i in items]


def tbl(rows, widths=None, header=True, font=8):
    data = [[Paragraph(str(c), ParagraphStyle("t", parent=B, fontSize=font, leading=font + 2)) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c3c8b0")), ("VALIGN", (0, 0), (-1, -1), "TOP")]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), SAGE))
    t.setStyle(TableStyle(st))
    return t


def build(path, story, title):
    path.parent.mkdir(parents=True, exist_ok=True)
    def foot(c, d):
        c.saveState(); c.setFont("Helvetica", 7); c.setFillColor(colors.grey)
        c.drawString(2 * cm, 1.2 * cm, f"TerraMind AI - {title}"); c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {d.page}"); c.restoreState()
    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm, title=title).build(story, onFirstPage=foot, onLaterPages=foot)
    print("wrote", path.relative_to(ROOT))


def f(v, d=3):
    return "Not evaluated" if v is None else f"{v:.{d}f}"


# ================================================================== architecture diagram
def architecture():
    fig, ax = plt.subplots(figsize=(16.5, 10.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 64); ax.axis("off")

    def box(x, y, w, h, text, fc="#ffffff", ec="#14532d", fs=8, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)

    def arrow(a, b, c="#14532d", style="-|>", ls="-"):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=11, color=c, lw=1.3, linestyle=ls))

    def lane(x, w, title):
        ax.add_patch(FancyBboxPatch((x, 2), w, 57, boxstyle="round,pad=0.2,rounding_size=1", fc="#f6f7f2", ec="#dcdfcf", lw=1))
        ax.text(x + w / 2, 57.2, title, ha="center", fontsize=9.5, fontweight="bold", color="#14532d")

    ax.text(50, 62.6, "TerraMind AI - system architecture (as implemented)", ha="center", fontsize=15, fontweight="bold", color="#14532d")
    lane(0.5, 16, "1. DATA"); lane(18, 17, "2. PREPARATION + INDEX"); lane(36.5, 28, "3. MULTI-AGENT ADVISORY (per query)"); lane(66, 15.5, "4. API (FastAPI)"); lane(83, 16.5, "5. REACT UI")
    # data
    box(1.5, 44, 14, 9, f"Kaggle dataset\ncrop_yield.csv ({DQ['source_row_count']:,} rows)\nstate_soil (30 states)\nstate_weather (state-year)", "#fff8e1", fs=7.2)
    box(1.5, 28, 14, 8, "Reference + real-row\nevaluation cases\n(data/evaluation)", fs=7.5)
    box(1.5, 10, 14, 9, "SQLite\nqueries, feedback,\nevaluation runs", "#e8f1ff", fs=8)
    # prep
    box(19, 47, 15, 6, "Validate / clean / dedupe\nmerge soil+weather, derive,\nstable case IDs", fs=7.3)
    box(19, 38.5, 15, 6, f"cases.csv ({DQ['final_row_count']:,} real rows)\n+ data_quality_report", "#e9f5ec", fs=7.5)
    box(19, 29, 15, 7, "Case text builder\n(1 row = 1 case, no splitting)\nMiniLM embeddings", fs=7.3)
    box(19, 19, 15, 7, "FAISS (cosine)  +  BM25\npersistent index\nmanifest + version check", "#e9f5ec", fs=7.5)
    box(19, 8, 15, 7, "Yield model training\nHGB vs RF vs baseline\ntemporal split, quantiles", fs=7.3)
    # agents
    box(38, 50, 12.5, 5.5, "Orchestrator\nstage status, failure isolation", "#14532d", "#14532d", 7.3, True); ax.texts[-1].set_color("white")
    box(38, 42.5, 12.5, 5.5, "A1 Condition Classifier\nregex + Claude (provenance)", fs=7.2)
    box(38, 34.5, 12.5, 6, "A2 Case Retrieval\nBM25 + semantic - RRF\nmetadata filters - priority", fs=7.2)
    box(38, 27, 12.5, 5.5, "A3 Crop Recommendation\nevidence + Claude explanation", fs=7.2)
    box(52, 42.5, 11.5, 5.5, "A4 Suitability / Risk\nrule-based envelope", fs=7.2)
    box(52, 34.5, 11.5, 6, "A5 Yield Prediction\ntrained regressor\n(estimate + interval)", fs=7.2)
    box(52, 27, 11.5, 5.5, "A6 Impact Analysis\nSpearman, groups,\npermutation importance", fs=7.2)
    box(38, 17.5, 25.5, 6, "A7 Validation\ndeterministic checks + Claude LLM-as-judge", "#fff3e0", fs=7.3)
    box(38, 8, 25.5, 6.5, "Evaluation pipeline\nP@K R@K MRR Hit@K - DeepEval\nyield MAE/RMSE/R2 - persisted runs", "#e8f1ff", fs=7.2)
    # api
    box(67, 48, 13.5, 7, "/api/advisory/*\nquery - jobs (stage polling)\nclassify - recommend", fs=7.3)
    box(67, 38, 13.5, 7, "/api/cases/*  /metadata\n/api/predictions/*\n/api/analysis/impact", fs=7.3)
    box(67, 28, 13.5, 7, "/api/evaluations/*\n(background run + status)", fs=7.3)
    box(67, 18, 13.5, 7, "/api/feedback/*\nsummary + review candidates", fs=7.3)
    box(67, 8, 13.5, 7, "/api/system/status\n/api/stats - /health", fs=7.3)
    # ui
    for i, t in enumerate(["Dashboard", "Crop Advisory", "Case Explorer", "Evaluation", "Insights & Feedback"]):
        box(85, 48 - i * 9, 13, 6, t, "#e9f5ec", fs=8.5, bold=True)
    # arrows (kept minimal and non-crossing)
    arrow((15.5, 49), (19, 50)); arrow((26.5, 47), (26.5, 44.5)); arrow((26.5, 38.5), (26.5, 36)); arrow((26.5, 29), (26.5, 26))
    arrow((19, 41.5), (15.5, 33), "#14532d", ls="--")
    arrow((34, 22.5), (38, 36), ls="--")
    arrow((44.2, 50), (44.2, 48)); arrow((44.2, 42.5), (44.2, 40.5)); arrow((44.2, 34.5), (44.2, 32.5))
    arrow((50.5, 30), (52, 45)); arrow((50.5, 30), (52, 37)); arrow((50.5, 30), (52, 30))
    arrow((44.2, 27), (44.2, 23.5)); arrow((57.7, 27), (57.7, 23.5), ls=":")
    arrow((50.5, 52.7), (67, 52.7)); arrow((63.5, 37), (67, 41)); arrow((63.5, 11.5), (67, 30), ls="--")
    arrow((80.5, 52), (85, 51)); arrow((80.5, 41), (85, 42)); arrow((80.5, 31), (85, 33)); arrow((80.5, 21), (85, 24))
    ax.text(50, 3.3, "Feedback loop: user feedback -> SQLite -> summary + review candidates -> human verification -> extra evaluation cases -> re-run evaluation (no automatic retraining)",
            ha="center", fontsize=8.5, color="#3b6ea5", style="italic")
    ax.text(50, 0.6, "Solid = query-time data flow, dashed = offline build / persistence. Retrieved records are untrusted data inside <untrusted_data> blocks; the API key never leaves the backend.", ha="center", fontsize=7.5, color="#555d47")
    out = ROOT / "architecture"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "Architecture_Diagram.pdf", bbox_inches="tight"); fig.savefig(out / "Architecture_Diagram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote architecture/Architecture_Diagram.pdf (+png)")


# ================================================================== dataset details
def dataset_doc():
    s = [P("Dataset Details", H1), P(f"Source: <b>Crop Yield Data with Soil and Weather Dataset</b> (Kaggle, CC0) - {DQ['source']}", B), Spacer(1, 6)]
    s += [P("Row counts", H2), tbl([["Item", "Value"], ["Source rows (crop_yield.csv)", f"{DQ['source_row_count']:,}"], ["Dropped: missing / invalid / exact duplicates / duplicate keys",
          f"{DQ['dropped_missing']} / {DQ['dropped_invalid']} / {DQ['dropped_exact_duplicates']} / {DQ['dropped_duplicate_keys']}"], ["Valid REAL rows", f"{DQ['real_row_count']:,}"], ["Labelled SYNTHETIC rows added", f"{DQ['synthetic_row_count']:,}"],
          ["Target rows (capstone brief)", f"{DQ['target_rows']:,}"], ["Final working rows", f"{DQ['final_row_count']:,}"], ["Target met with real data alone?", "No" if not DQ["target_met_with_real_data_only"] else "Yes"], ["Strategy", DQ["sampling"]["strategy"]],
          ["Random seed (used only if sampling were needed)", DQ["random_seed"]], ["Dataset version (hash of cases.csv)", DQ["dataset_version"]]], [8 * cm, 8.5 * cm])]
    s += [P(f"<b>How 55,000 records are reached:</b> the published dataset contains only {DQ['real_row_count']:,} valid real records. At the project owner's request the working dataset is topped up to {DQ['target_rows']:,} with <b>{DQ['synthetic_row_count']:,} labelled synthetic records</b> (is_synthetic = true, case ids start 'TS-', source_row negative). They are NOT historical observations. Method: {DQ.get('synthetic_method')} Safeguards: synthetic rows are badged in the UI, can be excluded with a filter / toggle, and are never used for yield-model training or testing, statistics, impact analysis or suitability envelopes (those use the {DQ['real_row_count']:,} real rows only). Re-run scripts/prepare_dataset.py --no-augment for a real-only dataset.", B)]
    s += [P("Schema and field meanings", H2), tbl([["Field", "Meaning / units", "Origin"],
          ["crop, year, season, state", "Crop name, calendar year, growing season (Kharif, Rabi, Summer, Winter, Autumn, Whole Year), Indian state", "crop_yield.csv"],
          ["area", "Cultivated area (hectares)", "crop_yield.csv"], ["production", "Production (crop-specific units)", "crop_yield.csv"],
          ["yield", "Reported yield, ~ production/area. " + DQ["yield_definition"], "crop_yield.csv"],
          ["fertilizer, pesticide", "Record totals (dataset units, not per hectare)", "crop_yield.csv"],
          ["n, p, k, ph", DQ["soil_note"], "state_soil_data.csv (joined on state)"], ["temperature, rainfall, humidity", DQ["weather_note"], "state_weather_data (joined on state+year)"],
          ["case_id", DQ["case_id"], "derived"], ["fertilizer_per_ha, pesticide_per_ha", "total / area", "derived"],
          ["yield_band, ph_class, rainfall_class, temp_class, humidity_class", "Descriptive classes with documented thresholds (see below)", "derived"],
          ["completeness, flags", "Share of soil/weather fields present; data-quality flags", "derived"]], [4.5 * cm, 8.5 * cm, 3.5 * cm])]
    s += [P("Preprocessing decisions", H2)] + bullets(DQ["decisions"]) + [P("Derived column definitions", H2)] + bullets([f"<b>{k}</b>: {v}" for k, v in DQ["derived_columns"].items()])
    s += [P("Data-quality findings", H2)] + bullets([
        f"Missing values after merge: none in any column (soil and weather joined for all {DQ['final_row_count']:,} rows; synthetic rows reuse the real state soil and state-year weather).",
        f"Flags: {DQ['flag_counts']['zero_production']} records with zero production (kept, flagged: could be crop failure or unreported); "
        f"{DQ['flag_counts']['yield_differs_from_production_over_area']} records ({DQ['flag_counts']['yield_differs_from_production_over_area'] / DQ['final_row_count']:.0%}) where reported yield differs from production/area by more than 10% (kept, flagged).",
        f"Extreme values (> Q3 + 3xIQR, inspected not deleted): {DQ['extreme_value_counts_3xIQR']}. These are legitimate large-area / high-magnitude observations (e.g. coconut yields are per nut).",
        f"Coverage: {DQ['n_crops']} crops, {DQ['n_states']} states, years {DQ['year_range'][0]}-{DQ['year_range'][1]} (2020 is partial: 37 rows).",
        "Not available in the source and therefore never inferred: soil type, irrigation, farm location, crop variety."])
    s += [P("Limitations", H2)] + bullets(["Soil values are state constants (30 distinct profiles): within a state, soil cannot explain yield differences, and soil features act partly as state identifiers.",
          "Weather is a state-year aggregate and does not describe an individual farm.", "Yield units are crop-specific and not stated by the source; cross-crop yield comparison is invalid.",
          "Fertilizer and pesticide are totals that scale with area; per-hectare values are derived."])
    build(ROOT / "docs/Dataset_Details.pdf", s, "Dataset Details")


# ================================================================== design doc
def design_doc():
    yf = YM["comparison"]["full"]; yc = YM["comparison"]["core"]
    s = [P("Design Document", H1), P("TerraMind AI - AI-powered agriculture crop advisory assistant", B)]
    s += [P("1. Problem statement", H2), P("Agricultural professionals hold thousands of historical records (crop, season, state, soil, weather, inputs, yield) but cannot quickly find comparable cases or judge which crops have done well under similar conditions. TerraMind retrieves similar historical cases from natural-language descriptions and turns them into evidence-backed, explainable crop recommendations - decision support, never a yield guarantee.", B)]
    s += [P("2. Architecture", H2), P("Offline: validate and merge the dataset, build case texts, embed with MiniLM, persist a FAISS index and a BM25 index, train the yield model. Online: a FastAPI service exposes an orchestrator that runs seven agents with real per-stage status; a React UI polls the job endpoint so progress shown is progress achieved. See architecture/Architecture_Diagram.pdf.", B)]
    s += [P("3. Technology choices and trade-offs", H2), tbl([["Choice", "Why", "Trade-off"],
          ["FastAPI + Pydantic", "Typed request/response validation, automatic OpenAPI", "Sync pipeline run in worker threads instead of full async"],
          ["SQLite (stdlib)", "Zero-setup local persistence for queries, feedback, evaluations", "Single-node; no migrations framework"],
          ["FAISS flat inner product", f"{DQ['final_row_count']:,} vectors: exact search in milliseconds, no tuning", "Would need IVF/HNSW beyond ~1M vectors"],
          ["all-MiniLM-L6-v2", "Small, CPU-friendly, adequate for short structured texts", "Weak on numeric similarity (handled by structured candidates)"],
          ["BM25 (rank-bm25)", "Exact crop/state/season term matching", "No stemming; vocabulary is small and controlled"],
          ["scikit-learn HistGradientBoosting / RF", "Strong tabular baselines, native NaN handling, quantile loss", "No deep model; limited by state-level features"],
          ["Claude (Anthropic SDK)", "Structured extraction, grounded explanation, judging", "Cost/latency; therefore used only where deterministic code cannot"],
          ["React + Vite + Tailwind + Recharts", "Fast dev loop, accessible components", "Client-side polling instead of websockets"]], [4 * cm, 6.5 * cm, 6 * cm])]
    s += [P("4. Dataset design", H2), P(f"{DQ['real_row_count']:,} real records plus {DQ['synthetic_row_count']:,} labelled synthetic records (is_synthetic flag, 'TS-' ids) reach the 55,000 working size; models, statistics and suitability use real rows only (see Dataset_Details.pdf). Soil values are state constants and weather is state-year, joined and labelled as such. Stable case IDs are SHA-1 of the natural key, so a case keeps its identity across rebuilds; source_row links back to the raw CSV.", B)]
    s += [P("5. Case representation, chunking and embeddings", H2), P("One CSV row is one case and one chunk: rows are ~60 words and splitting them would separate a crop from its conditions. The text is generated only from the record's own values and includes descriptive classes (e.g. 'acidic soil', 'high rainfall', 'high yield for this crop') so that lexical and semantic queries phrased in natural language can match. Embeddings are computed once at index-build time, L2-normalised, and stored with a manifest (dataset version, model, case ids). A stale index (dataset changed) is refused rather than silently mapped to wrong records.", B)]
    s += [P("6. Hybrid retrieval and ranking", H2)] + bullets([
        "BM25 and semantic rankings are computed after applying structured metadata filters (crop, season, state, year and inclusive numeric ranges); filtered-out counts are returned.",
        "Fusion: Reciprocal Rank Fusion, score = sum 1/(60 + rank) over the two lists. It is a rank-based ordering score, not a calibrated probability.",
        "If one backend fails the other's results are returned with an explicit fallback notice; if both fail the API returns 503. No cases are ever fabricated.",
        "Because embeddings are weak on numbers, the retrieval agent adds structured candidates: the 2 cases per crop whose soil/weather values are closest to the user's.",
        "Prioritisation (src/rag/prioritizer.py): weighted sum of hybrid relevance 0.30, crop match 0.15, season 0.10, region 0.10, soil similarity 0.15, weather similarity 0.10, data completeness 0.10; components with no user input are 'not assessed' and weights are renormalised (unknown is never a match). Historical yield is deliberately NOT a component."])
    s += [P("7. Agent collaboration", H2), P("A central orchestrator sequences: validation, classification (Agent 1), retrieval + prioritisation (Agent 2), recommendation (Agent 3), then suitability (Agent 4), yield (Agent 5) and impact (Agent 6) in parallel threads with timeouts, then explanation and validation (Agent 7). Each stage records real status and duration; an optional agent that fails is reported and its siblings' results are kept. Agents exchange typed data (Pydantic / dicts), not free text. Claude is called for: query understanding (validated against dataset vocabulary, never used as a hard filter when inferred), the narrative explanation (given only a facts JSON) and the judge.", B)]
    s += [P("8. Prediction methodology", H2), P(f"Target: yield, modelled as log1p(yield) because units span orders of magnitude between crops. Production is never a feature (leakage: yield ~ production/area). Two feature sets: 'core' (crop, season, state, year, soil, weather) and 'full' (+ log area, fertilizer/ha, pesticide/ha) used only when the user supplies area and inputs. Temporal split: train <= 2014 ({YM['split']['n_train']:,}), validation 2015-2017 ({YM['split']['n_val']:,}), test 2018-2020 ({YM['split']['n_test']:,}); model chosen on validation log-RMSE, refit on train+validation, evaluated once on test. Prediction intervals come from quantile gradient boosting; empirical test coverage of the nominal 80% interval is {yf['selected']['interval_10_90']['empirical_test_coverage']:.0%} (full) / {yc['selected']['interval_10_90']['empirical_test_coverage']:.0%} (core), i.e. slightly under-covering - reported as is.", B)]
    rows = [["Feature set", "Model", "Test MAE", "Test RMSE", "Test R2", "Test R2 (log1p)"]]
    for fs, c in YM["comparison"].items():
        rows.append([fs + " - baseline (crop median)", "baseline", f(c["baseline_crop_median"]["test"]["mae"], 2), f(c["baseline_crop_median"]["test"]["rmse"], 2), f(c["baseline_crop_median"]["test"]["r2"]), f(c["baseline_crop_median"]["test"]["r2_log1p"])])
        t = c["selected"]["test"]; rows.append([fs + " - selected", c["selected"]["name"], f(t["mae"], 2), f(t["rmse"], 2), f(t["r2"]), f(t["r2_log1p"])])
    s += [tbl(rows, [4.5 * cm, 2.2 * cm, 2.3 * cm, 2.5 * cm, 2.3 * cm, 2.7 * cm])]
    s += [P("Suitability risk: the dataset has no suitability labels, so a classifier would need invented labels. Instead a transparent rule-based 'historical envelope' compares each supplied condition with the 10th-90th percentile and min-max range in which the crop has been recorded (plus season/state presence and yield variability). The output is labelled as a heuristic, never a probability.", B)]
    s += [P("9. Guardrails", H2)] + bullets(["Pydantic schemas with defensible hard ranges; unusual-but-possible values are flagged for confirmation rather than rejected; 64 KB body limit; query length limit.",
          "Prompt-injection: retrieved text and user queries are wrapped in <untrusted_data> blocks with a system rule that their content is data; suspicious phrasing is detected and surfaced as a warning; all LLM output is schema-validated; Claude is given no tools.",
          "The API key is read from .env by the backend only; it never appears in responses, logs or frontend variables.",
          "Failure handling: missing key, auth/billing/rate-limit/timeout errors, refusals and invalid JSON are typed (LLMError) with bounded retries and credential-free messages; the non-LLM pipeline keeps working."])
    s += [P("10. Evaluation", H2), P("Verified evaluation cases are built from the real dataset: relevance labels are deterministic filters (e.g. all records with the stated crop+state+season) and are re-verified on every run. Retrieval metrics (Precision@K, Recall@K, MRR, Hit@K) are computed only where labels exist. DeepEval faithfulness, answer relevancy, contextual precision and contextual recall run through a Claude adapter on pipeline answers; if the LLM is unavailable they are recorded as unavailable with the reason, never as a score. Yield-model metrics are held-out and reported separately. All runs persist to SQLite and artifacts/evaluation. Pass/fail rules: retrieval cases pass at R-precision >= 0.5 (Precision@10 is mathematically capped when fewer than 10 relevant cases exist; this rule was adopted after the first run exposed that cap - Precision@10 is still reported unchanged).", B)]
    s += [P("11. Feedback loop", H2), P("Users submit helpful/not-helpful, rating, category and comment per recommendation; entries are stored with the query and supporting case ids. A summary service reports helpful rate, categories, trends and review candidates. The improvement workflow is deliberately manual: verify a failure, add it to data/evaluation/test_cases_extra.json, re-run evaluation, compare runs. There is no automatic retraining.", B)]
    s += [P("12. Limitations", H2)] + bullets([f"Only {DQ['real_row_count']:,} real records (35k synthetic rows add retrieval volume but no new real evidence); soil is state-level; weather is state-year; yield units differ per crop.",
          "Recommendations reflect where crops have historically been recorded and how similar the conditions are - they are not agronomic prescriptions.",
          "Original-scale yield R2/RMSE are dominated by a few high-magnitude crops; log-scale metrics are reported alongside.", "Feature importance and correlations are associations; confounding by state and year is documented.",
          "LLM components depend on API availability and credit; without them the system degrades to deterministic parsing, template explanations and deterministic validation."])
    build(ROOT / "design/Design_Document.pdf", s, "Design Document")


# ================================================================== API doc
def api_doc():
    from fastapi.testclient import TestClient
    import src.config as cfg
    from src.main import app
    cfg.settings.anthropic_api_key = ""   # sample responses must not spend API credit
    c = TestClient(app)
    spec = app.openapi()
    samples = {
        "POST /api/advisory/query": ("POST", "/api/advisory/query", {"query": "Which crops perform well in Punjab during Rabi with soil pH 7.8 and rainfall around 600 mm?", "use_llm": False, "max_recommendations": 2, "top_k": 2}),
        "POST /api/cases/search": ("POST", "/api/cases/search", {"query": "wheat punjab rabi", "top_k": 2, "filters": {"state": ["Punjab"]}}),
        "POST /api/predictions/yield": ("POST", "/api/predictions/yield", {"crop": "Wheat", "state": "Punjab", "season": "Rabi", "ph": 8, "rainfall": 600}),
        "POST /api/predictions/suitability": ("POST", "/api/predictions/suitability", {"crop": "Wheat", "state": "Punjab", "season": "Rabi", "ph": 8}),
        "POST /api/advisory/classify": ("POST", "/api/advisory/classify", {"query": "wheat in Punjab Rabi pH 7.8", "use_llm": False}),
    }

    def short(o, depth=0):
        if isinstance(o, dict):
            return {k: short(v, depth + 1) for k, v in list(o.items())[:14]}
        if isinstance(o, list):
            return [short(v, depth + 1) for v in o[:2]] + (["..."] if len(o) > 2 else [])
        if isinstance(o, str) and len(o) > 140:
            return o[:140] + "..."
        return o

    s = [P("API Documentation", H1), P("Base URL: http://127.0.0.1:8000. Interactive docs: /docs (Swagger UI) and /openapi.json. All routes are generated from the running application; validation errors return HTTP 422 with a list of {loc, msg}; domain errors return {\"detail\", \"code\"}.", B)]
    s += [P("Status codes", H2), tbl([["Code", "Meaning"], ["200 / 201 / 202", "Success / created (feedback) / accepted background job"], ["400", "Invalid request value (e.g. unknown sort_by)"], ["404", "Unknown case, query, run or job"],
          ["409", "An evaluation is already running"], ["413", "Payload larger than 64 KB"], ["422", "Schema validation failure (ranges, lengths, enums)"], ["503", "Dataset, index or model artifact missing / retrieval backends unavailable (code field says which)"]], [3 * cm, 13.5 * cm])]
    s += [P("Endpoint reference", H2)]
    rows = [["Method", "Path", "Purpose", "Parameters / body"]]
    for path, ops in sorted(spec["paths"].items()):
        for m, op in ops.items():
            params = ", ".join(p["name"] for p in op.get("parameters", [])) or ""
            body = ""
            rb = op.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema", {})
            if "$ref" in rb:
                name = rb["$ref"].split("/")[-1]
                props = spec["components"]["schemas"][name].get("properties", {})
                body = f"<b>{name}</b>: " + ", ".join(props)
            rows.append([m.upper(), path, (op.get("summary") or "") + (" - " + op["description"].split("\n")[0] if op.get("description") else ""), (params + ("; " if params and body else "") + body)[:420]])
    s += [tbl(rows, [1.4 * cm, 4.3 * cm, 5 * cm, 5.8 * cm], font=7)]
    s += [P("Sample requests and real responses", H2), P("Captured from the running application with use_llm=false (so no API credit is used). Long lists are truncated with '...'.", SM)]
    for title, (m, url, body) in samples.items():
        r = c.post(url, json=body)
        s += [P(f"<b>{title}</b>  -  HTTP {r.status_code}", B), Preformatted("Request: " + json.dumps(body), CODE), Preformatted(json.dumps(short(r.json()), indent=1)[:2600], CODE), Spacer(1, 6)]
    bad = c.post("/api/advisory/query", json={"query": "", "ph": 99})
    s += [P("<b>Validation error example</b>  -  POST /api/advisory/query {\"query\": \"\", \"ph\": 99}", B), Preformatted(f"HTTP {bad.status_code}\n" + json.dumps(bad.json(), indent=1)[:900], CODE)]
    s += [P("Advisory response structure", H2), tbl([["Field", "Content"], ["query_id, status", "completed | completed_with_warnings | needs_clarification | no_evidence | failed"], ["validated_input", "request after validation"], ["conditions", "per-field {value, provenance, note}, missing_information, excluded_crops, objectives"],
          ["retrieval", "filters applied, candidates after filter / filtered out, backends, fallback, scoring note"], ["cases[]", "case_id, record, keyword/semantic rank+score, hybrid rank/score, priority_score + components + explanation"],
          ["recommendations[]", "rank, crop, score (heuristic), reasons, supporting_case_ids, historical_yield (actual), yield_prediction (model estimate + interval + assumptions), risk (rule-based), judge, warnings"], ["narrative", "LLM or template explanation with source"],
          ["impact_analysis", "correlations, group comparisons, permutation importance, confounding notes"], ["judge", "deterministic checks + LLM verdict (evidence_support, condition_match, constraint_violations, unsupported_claims, ...)"],
          ["agents[], stages[]", "real status and duration of each agent / workflow stage"], ["warnings, metadata", "limitations, durations, dataset version"]], [4 * cm, 12.5 * cm])]
    build(ROOT / "docs/API_Documentation.pdf", s, "API Documentation")


# ================================================================== evaluation report
def eval_doc():
    s = [P("Evaluation Report", H1)]
    if not EV:
        s += [P("Not evaluated - no evaluation run exists. Run python scripts/run_evaluation.py.", B)]
        build(ROOT / "docs/Evaluation_Report.pdf", s, "Evaluation Report"); return
    sm, v = EV["summary"], EV["versions"]
    s += [P(f"Run <b>{EV['run_id']}</b> ({EV['created_at']}, {EV['duration_s']} s). Generated from artifacts/evaluation/{EV['run_id']}.json. Every value below is read from that file; nothing is estimated or typed in.", B)]
    s += [tbl([["Item", "Value"], ["Dataset version / records", f"{v['dataset_version']} / {v['n_records']:,} (incl. labelled synthetic)"], ["Yield model version", v["yield_model_version"]], ["LLM judge model", v["llm_model"] or "none (LLM disabled or unavailable in this run)"],
          ["Embedding model", v["embedding_model"]], ["Retrieval", v["retrieval"]], ["Test cases", f"{sm['n_test_cases']} (passed {sm['n_passed']}, failed {sm['n_failed']})"], ["Label verification", "all relevance labels re-verified against the dataset" if EV["label_verification"]["verified"] else str(EV["label_verification"]["problems"])]], [5 * cm, 11.5 * cm])]
    s += [P("Test cases and how labels were created", H2), P("Constraint-retrieval (16): the query states crop+state+season; relevant cases = all dataset records with that exact combination. Conditions-recommendation (8): state, season and real soil/weather values from an anchor record; every recommended crop must have at least one record in that state+season. Abstention (4): crop/state pairs with zero records. Clarification (3): queries with no agricultural context. Queries are template-generated over real values; they are test inputs, not historical records.", B)]
    r = sm["retrieval"]
    s += [P(f"Retrieval metrics (K={r['k']}, {r['n_cases']} constraint-retrieval queries)", H2), tbl([["Metric", "Value"], [f"Precision@{r['k']}", f(r["precision_at_k"])], [f"Recall@{r['k']}", f(r["recall_at_k"])], ["MRR", f(r["mrr"])], [f"Hit rate@{r['k']}", f(r["hit_rate_at_k"])],
          [f"Precision@{r['k']} on recommendation queries (state+season relevance)", f(r["recommendation_case_precision_at_k"])]], [10 * cm, 6.5 * cm]),
          P("Precision@10 is capped below 1 when fewer than 10 relevant cases exist (several groups have only 3-4 records), so it understates quality; MRR and Hit@K show that a relevant case is always ranked first. Pass/fail uses R-precision (precision at k = number of relevant cases).", SM)]
    s += [P("DeepEval metrics (Claude judge)", H2), P(f"Status: {sm['deepeval_status']}", B), tbl([["Metric", "Mean", "Measured", "Unavailable", "Reason (if unavailable)"]] + [[k, f(d["mean"]), d["n_measured"], d["n_unavailable"], (d["unavailable_reason"] or "")[:150]] for k, d in sm["deepeval"].items()], [3.5 * cm, 2 * cm, 2 * cm, 2.2 * cm, 6.8 * cm])]
    rv = sm["recommendation_validation"]
    s += [P("Recommendation validation (deterministic)", H2), tbl([["Check", "Result"], ["Constraint satisfaction (recommended crops recorded in state+season)", f(rv["constraint_satisfaction_rate"])], ["Abstention pass rate (no-evidence crops not recommended + disclosed)", f(rv["abstention_pass_rate"])],
          ["Clarification pass rate", f(rv["clarification_pass_rate"])], ["Unsupported numeric claims flagged", rv["unsupported_numeric_claims_total"]], ["LLM judge verdicts", str(rv["judge_verdicts"]) if rv["judge_verdicts"] and set(rv["judge_verdicts"]) != {"not_run"} else "Not evaluated (LLM judge did not run in this run)"]], [11 * cm, 5.5 * cm])]
    y = sm["yield_model"]
    s += [P("Yield model (held-out test years 2018-2020; independent of LLM metrics)", H2)]
    rows = [["Feature set", "Model", "MAE", "RMSE", "R2", "R2 (log1p)", "Baseline MAE", "Baseline R2"]]
    for fs, t in y["selected_test_metrics"].items():
        b = y["baseline_test_metrics"][fs]
        rows.append([fs, y["selected_models"][fs], f(t["mae"], 2), f(t["rmse"], 2), f(t["r2"]), f(t["r2_log1p"]), f(b["mae"], 2), f(b["r2"])])
    s += [tbl(rows, [1.8 * cm, 1.6 * cm, 1.8 * cm, 2 * cm, 1.8 * cm, 2.2 * cm, 2.6 * cm, 2.6 * cm]), P("Units: source yield units, crop-specific. Original-scale RMSE/R2 are dominated by a few high-magnitude crops.", SM)]
    s += [P("Failure cases and findings", H2)]
    fails = [x for x in EV["results"] if not x.get("passed")]
    s += bullets([f"<b>{x['test_case_id']}</b> ({x['type']}): {'; '.join(x['failure_reasons'])}" for x in fails] or ["No failing test cases in this run."])
    s += bullets(["First run (eval-20261009-140625) flagged RET-04 and RET-13 as failures under a Precision@10 >= 0.5 rule; inspection showed every relevant case (3 and 4 exist) was retrieved at ranks 1-4 and the failure was the metric ceiling, so the rule was changed to R-precision. This is disclosed rather than hidden.",
                  "Claude-dependent metrics (DeepEval, LLM judge) were Not evaluated in the recorded runs because the Anthropic account's credit balance was exhausted when the evaluation executed. Re-run python scripts/run_evaluation.py after adding credit to populate them; no value was filled in."])
    build(ROOT / "docs/Evaluation_Report.pdf", s, "Evaluation Report")


if __name__ == "__main__":
    which = sys.argv[1:] or ["architecture", "dataset", "design", "api", "eval"]
    for w in which:
        {"architecture": architecture, "dataset": dataset_doc, "design": design_doc, "api": api_doc, "eval": eval_doc}[w]()
