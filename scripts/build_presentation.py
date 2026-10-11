"""python scripts/build_presentation.py -> presentation/Project_Presentation.pptx (numbers read from real artifacts)"""
import glob
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
DQ = json.loads((ROOT / "data/processed/data_quality_report.json").read_text(encoding="utf-8"))
YM = json.loads((ROOT / "artifacts/models/yield_metrics.json").read_text(encoding="utf-8"))
ev = sorted(glob.glob(str(ROOT / "artifacts/evaluation/eval-*.json")))
EV = json.loads(Path(ev[-1]).read_text(encoding="utf-8")) if ev else None
SHOTS = ROOT / "docs/screenshots"
GREEN, EM, SAGE, INK = RGBColor(0x14, 0x30, 0x1F), RGBColor(0x05, 0x96, 0x69), RGBColor(0x55, 0x5D, 0x47), RGBColor(0x1C, 0x2A, 0x21)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
blank = prs.slide_layouts[6]


def slide(title, subtitle=None):
    s = prs.slides.add_slide(blank)
    bar = s.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(0.18)); bar.fill.solid(); bar.fill.fore_color.rgb = GREEN; bar.line.fill.background()
    tb = s.shapes.add_textbox(Inches(0.6), Inches(0.35), Inches(12), Inches(0.8)).text_frame
    tb.text = title; r = tb.paragraphs[0].runs[0]; r.font.size = Pt(30); r.font.bold = True; r.font.color.rgb = GREEN
    if subtitle:
        p = tb.add_paragraph(); p.text = subtitle; p.runs[0].font.size = Pt(14); p.runs[0].font.color.rgb = SAGE
    return s


def bullets(s, items, x=0.6, y=1.5, w=6.2, size=17):
    tf = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(5.5)).text_frame
    tf.word_wrap = True
    for i, t in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = "• " + t; p.runs[0].font.size = Pt(size); p.runs[0].font.color.rgb = INK; p.space_after = Pt(9)


def pic(s, path, x, y, w=None, h=None):
    if Path(path).exists():
        s.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)


def notes(s, text):
    s.notes_slide.notes_text_frame.text = text


def table(s, rows, x, y, w, colw=None, size=13):
    t = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(0.4 * len(rows))).table
    for i, r in enumerate(rows):
        for j, c in enumerate(r):
            cell = t.cell(i, j); cell.text = str(c)
            for p in cell.text_frame.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(size); run.font.bold = i == 0
    return t


def f(v, d=3):
    return "Not evaluated" if v is None else f"{v:.{d}f}"


# 0 title
s = prs.slides.add_slide(blank)
bg = s.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height); bg.fill.solid(); bg.fill.fore_color.rgb = GREEN; bg.line.fill.background()
tf = s.shapes.add_textbox(Inches(0.9), Inches(2.3), Inches(11.5), Inches(3)).text_frame
tf.text = "TerraMind AI"; tf.paragraphs[0].runs[0].font.size = Pt(54); tf.paragraphs[0].runs[0].font.bold = True; tf.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
for t, sz in (("AI-Powered Agriculture Crop Advisory Assistant", 24), ("Smarter crop decisions. Evidence-driven farming.", 16)):
    p = tf.add_paragraph(); p.text = t; p.runs[0].font.size = Pt(sz); p.runs[0].font.color.rgb = RGBColor(0xBC, 0xD8, 0xC3)

# 1 problem
s = slide("1. Problem statement")
bullets(s, ["Agricultural teams hold thousands of historical records: crop, season, state, soil, weather, inputs, yield.",
            "Finding comparable farming cases by hand is slow and inconsistent.",
            "Typical questions: which crops did well under similar conditions? Why do yields differ? What alternative crop fits my season and region?",
            "Goal: natural-language questions in, evidence-backed crop advice out — with every claim traceable to real cases."], w=12)
notes(s, "Frame the pain: manual search through records. Emphasise traceability.")

# 2 need
s = slide("2. Business need")
bullets(s, ["Faster, more consistent crop-selection discussions for advisers and farmers.", "Transparent evidence: show the historical cases behind every recommendation.",
            "Honest uncertainty: keep observed yield separate from model estimates; never promise outcomes.", "Safe by design: guardrails, validation and a judge check before advice is shown."], w=12)

# 3 solution
s = slide("3. Solution overview")
bullets(s, ["Hybrid RAG (BM25 + semantic, fused by RRF) over historical cases with metadata filters.", "Seven cooperating agents behind one orchestrator.",
            "Trained yield model, rule-based suitability risk, soil/weather/input impact analysis.", "Claude for query understanding, grounded explanation and LLM-as-judge.",
            "FastAPI backend, React dashboard, DeepEval + retrieval metrics, persistent feedback loop."], w=6.2)
pic(s, SHOTS / "s-dashboard.png", 7.0, 1.5, w=6.0)

# 4 dataset
s = slide("4. Dataset and preprocessing", "Kaggle: Crop Yield Data with Soil and Weather Dataset (CC0)")
bullets(s, [f"{DQ['source_row_count']:,} real source rows (all valid) + {DQ['synthetic_row_count']:,} labelled synthetic rows = {DQ['final_row_count']:,}.", "Synthetic rows are flagged, badged in the UI and excluded from model testing and statistics.",
            f"{DQ['n_crops']} crops, {DQ['n_states']} states, {DQ['year_range'][0]}–{DQ['year_range'][1]}.", "Soil N/P/K/pH are state-level constants; weather is state-year — joined and labelled as such.",
            "Stable case IDs, quality flags, outliers inspected not deleted.", f"Yield ≈ production/area; units are crop-specific."], w=12, size=16)

# 5 architecture
s = slide("5. Architecture")
pic(s, ROOT / "architecture/Architecture_Diagram.png", 0.5, 1.2, h=6.0)

# 6 retrieval
s = slide("6. Hybrid retrieval")
bullets(s, ["One CSV row = one case = one chunk (no splitting of short structured rows).", "BM25 keywords + MiniLM embeddings in a persistent FAISS index with a version manifest.",
            "Metadata filters first (crop, season, state, year, numeric ranges), then rank.", "Reciprocal Rank Fusion: score = Σ 1/(60+rank) — a ranking score, not a probability.",
            "Fallbacks: one backend down → the other answers, with a visible notice.", "Prioritisation: relevance + crop/season/region + soil/weather similarity + completeness. Yield is never a ranking input."], w=12, size=16)

# 7 agents
s = slide("7. Agent collaboration")
table(s, [["Agent", "Responsibility"], ["1 Classifier", "Regex + Claude → conditions with provenance"], ["2 Retrieval", "Hybrid search, filters, prioritisation"], ["3 Recommendation", "Group evidence by crop, explain"],
          ["4 Suitability", "Rule-based historical-envelope risk"], ["5 Yield", "Trained regressor + interval"], ["6 Impact", "Correlations, groups, permutation importance"], ["7 Validation", "Deterministic checks + LLM judge"]], 0.6, 1.5, 7.3)
pic(s, SHOTS / "s-advisory-conditions.png", 8.2, 1.5, w=4.9)
notes(s, "Agents 4-6 run in parallel; a failing optional agent does not discard the others.")

# 8 recommendation + yield
yf = YM["comparison"]["full"]["selected"]; yb = YM["comparison"]["full"]["baseline_crop_median"]["test"]
s = slide("8. Recommendation and yield prediction")
bullets(s, ["Each recommendation: reasons, supporting case IDs, actual historical yield, predicted yield (clearly an estimate), risk, judge verdict.",
            f"Model: {yf['name'].upper()} on log1p(yield), temporal split (train ≤2014, test 2018–2020).", f"Held-out test: MAE {yf['test']['mae']:.2f}, RMSE {yf['test']['rmse']:.1f}, R² {yf['test']['r2']:.3f} (baseline R² {yb['r2']:.3f}); log-scale R² {yf['test']['r2_log1p']:.3f}.",
            f"10–90% interval covers {yf['interval_10_90']['empirical_test_coverage']:.0%} on test (nominal 80%) — slightly under-covers; reported as is."], w=6.2, size=15)
pic(s, SHOTS / "s-advisory-recs.png", 6.9, 1.4, w=6.2)

# 9 evaluation
s = slide("9. Actual evaluation results")
if EV:
    sm = EV["summary"]; r = sm["retrieval"]
    table(s, [["Metric", "Value"], [f"Precision@{r['k']}", f(r["precision_at_k"])], [f"Recall@{r['k']}", f(r["recall_at_k"])], ["MRR", f(r["mrr"])], [f"Hit rate@{r['k']}", f(r["hit_rate_at_k"])],
              ["Test cases passed", f"{sm['n_passed']} / {sm['n_test_cases']}"], ["Constraint satisfaction", f(sm["recommendation_validation"]["constraint_satisfaction_rate"])],
              ["DeepEval (4 metrics)", "Not evaluated" if all(d["mean"] is None for d in sm["deepeval"].values()) else ", ".join(f"{k}: {f(d['mean'], 2)}" for k, d in sm["deepeval"].items())]], 0.6, 1.5, 6.4, size=13)
    bullets(s, [f"Run {EV['run_id']}; labels verified against the dataset.", "DeepEval and the LLM judge need Claude credit; they were unavailable in the recorded run and are shown as Not evaluated, not estimated.",
                "Pass rule for retrieval is R-precision (Precision@10 is capped when only 3–4 relevant cases exist) — disclosed."], x=7.3, w=5.6, size=14)
else:
    bullets(s, ["Not evaluated"], w=12)

# 10 demo
s = slide("10. Demonstration workflow")
bullets(s, ["Ask: “Which crops perform well in Punjab during Rabi with pH 7.8, rainfall ~600 mm, N 150?”", "Watch real stage progress (validate → classify → retrieve → prioritise → recommend → analyse → judge).",
            "Open a supporting case; compare actual vs predicted yield; read the risk factors.", "Submit feedback; review it on Insights; re-run evaluation."], w=5.8, size=15)
pic(s, SHOTS / "s-explorer.png", 6.6, 1.4, w=6.5)

# 11 learnings
s = slide("11. Challenges and learnings")
bullets(s, ["Real data was smaller (19,689 vs 55,000): the gap is filled with clearly labelled synthetic rows, kept out of evaluation of the model.", "Embeddings are weak on numbers → added structured numeric candidates to retrieval.",
            "A naive Precision@10 pass rule mis-scored perfect retrievals; fixed and disclosed.", "LLM billing/availability failures degrade gracefully; metrics stay honest.",
            "Next: more granular soil data, farm-level labels for a real suitability classifier, calibrated intervals."], w=12, size=16)

out = ROOT / "presentation"
out.mkdir(exist_ok=True)
prs.save(out / "Project_Presentation.pptx")
print("wrote presentation/Project_Presentation.pptx")
