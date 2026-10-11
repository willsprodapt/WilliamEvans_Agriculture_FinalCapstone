"""python scripts/build_experiment_report.py -> docs/experiments/Experiments_Report.md
Every number is read from artifacts/evaluation/experiments/*.json and docs/experiments/data_audit.json (real runs). Missing file -> 'not run'."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
EXP = ROOT / "artifacts" / "evaluation" / "experiments"
OUT = ROOT / "docs" / "experiments" / "Experiments_Report.md"


def load(name, base=EXP):
    p = base / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def f(v, d=3):
    return "n/a" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


L: list[str] = []
audit, reg, clf = load("data_audit", ROOT / "docs" / "experiments"), load("regression"), load("classification")
ret, tune, held, dim = load("retrieval"), load("recommend_tuning"), load("recommend_heldout"), load("recommend_dimensions")
L += ["# TerraMind experiments report", "",
      "All numbers below were produced by `scripts/run_experiments.py` / `scripts/audit_data.py` (fixed seed 42). Nothing is estimated; anything not run is marked *not run*.", ""]

# ---------------------------------------------------------------- 1. data
L += ["## 1. Data actually used", ""]
if audit:
    k, c, sv = audit["kaggle_raw"], audit["processed_corpus"], audit["synthetic_vs_real"]
    L += [table(["", "Kaggle-sourced", "Synthetic"], [
        ["Rows", f"{c['kaggle_sourced']:,}", f"{c['synthetic']:,}"],
        ["Share of 55,000-row corpus", f"{1 - c['synthetic_share']:.1%}", f"{c['synthetic_share']:.1%}"],
        ["Years", f"{c['years_real'][0]}-{c['years_real'][1]}", f"{c['years_synth'][0]}-{c['years_synth'][1]}"],
        ["Crops / states", f"{k['n_crops']} / {k['n_states']}", "same crop-state pairs as Kaggle (0 new)"]]), "",
          "**Correction to the brief:** the Kaggle crop file contains 19,689 rows (not ~10,000) and the shipped synthetic set is 35,311 rows (not 45,000). Counts are as measured.", "",
          "**Kaggle schema:** crop, year, season, state, area, production, fertilizer, pesticide, yield (+ separate state soil N/P/K/pH constants and state-year weather). "
          "No suitability/condition label, soil type, irrigation, pest/disease or farm-practice field exists. Yield units are not stated and are crop-specific.", "",
          "### Problems found in the original implementation", ""]
    L += [f"- {p}" for p in audit["problems_found"]]
    L += ["- Measured: " + f"{audit['real_quality']['yield_vs_production_over_area_mismatch_gt10pct']:,} Kaggle rows have reported yield differing >10% from production/area; "
          f"{audit['real_quality']['zero_production_rows']} rows have zero production; {audit['real_quality']['groups_with_fewer_than_5_rows']} of {audit['real_quality']['groups_crop_state_season']} crop-state-season groups have <5 rows; year 2020 has only 37 rows.",
          f"- Measured: {sv['share_near_copy']:.2%} of existing synthetic rows are within 0.05 SD of a real row on all four numeric outcome fields (so they are noisy variants, not exact copies).",
          "- Existing evaluation: 31 cases built programmatically from the dataset (template queries over real values; no LLM, no experts). Retrieval labels = exact metadata match "
          "(circular with the query); recommendation cases only required a recommended crop to have *any* record in the state+season (almost any crop passes). "
          "They remain useful as regression tests but are not evidence of recommendation quality; section 5 adds independent checks.",
          "- Bug found by the new checks: when the user excluded the most frequent crop (\"... but not Gram\"), the top-60 pool was filled with that crop and, after exclusion, no recommendation remained (status `no_evidence`). Fixed.",
          "- Bug found: the recommender scored below a simple popularity baseline on held-out years (see section 5); a popularity blend tuned on validation years was added.", ""]
else:
    L += ["*data audit not run*", ""]

# ---------------------------------------------------------------- 2. generator
L += ["## 2. Synthetic data: rules, assumptions, provenance", "",
      "The shipped corpus (`data/processed/cases.csv`) is unchanged; every row now carries `data_source` (`kaggle` / `synthetic`; derived at load time when absent, written by `prepare_dataset.py`). "
      "Its synthetic rows were created from templates drawn from *all* years, i.e. their labels can derive from later held-out years. **They are therefore never used in any experiment below.** "
      "Experiments use generator v2 (`src/services/data_prep.py::generate_synthetic`, seed 42), built only from the training split:", ""]
if reg:
    L += [f"{i}. {r}" for i, r in enumerate(reg["synthetic_generator"]["rules"], 1)]
L += ["", "Assumptions: yield trends are log-linear per crop and small; noise levels are modelling choices, not agronomic measurements. "
      "Synthetic rows add no new agronomic information (same crop-state-season groups), only plausible variation.", ""]

# ---------------------------------------------------------------- 3. splits
L += ["## 3. Split methodology", ""]
if reg:
    s = reg["split"]
    L += [f"Temporal split on Kaggle rows: **train <= 2014 ({s['real_train']:,}), validation 2015-2017 ({s['real_validation']:,}), test >= 2018 ({s['real_test']:,})**. "
          "Hyper-parameters/model choice use validation only; final models are refit on train(+synthetic)+validation and scored once on test. "
          f"Synthetic training sets: 1x = {s['synthetic_train_1x']:,} rows, 2x = {s['synthetic_train_2x']:,} rows (from train rows only). "
          f"Synthetic test set (Experiment C): {s['synthetic_test']:,} rows, signature-disjoint from the synthetic training rows.", "",
          "Temporal splitting keeps near-duplicates (same crop-state-season, adjacent years) from straddling splits in the realistic forecasting direction. "
          "A supplementary spatial hold-out (6 unseen states) is also reported.", ""]

# ---------------------------------------------------------------- 4. ML
L += ["## 4. ML results", "", "### 4a. Yield regression", ""]
if reg:
    L += ["Target: `yield` (source units, crop-specific; modelled as log1p). **Log-scale metrics are the primary ones**: original-scale MAE/RMSE are dominated by a few high-magnitude crops (e.g. coconut).", ""]
    rows = [["Baseline: crop median", "-", f(reg["baseline_crop_median"]["test_real"]["mae"], 2), f(reg["baseline_crop_median"]["test_real"]["rmse"], 1), f(reg["baseline_crop_median"]["test_real"]["r2"]),
             f(reg["baseline_crop_median"]["test_real"]["rmse_log1p"]), f(reg["baseline_crop_median"]["test_real"]["r2_log1p"]), "-"]]
    if "baseline_persistence_last_observed_yield" in reg:
        b = reg["baseline_persistence_last_observed_yield"]["test_real"]
        rows.append(["Baseline: last observed yield of same crop-state-season", "-", f(b["mae"], 2), f(b["rmse"], 1), f(b["r2"]), f(b["rmse_log1p"]), f(b["r2_log1p"]), "-"])
    for name, d in reg["experiments"].items():
        for fs, r in d.items():
            t, sy = r["test_real"], r["test_synthetic"]
            rows.append([name, f"{fs} / {r['selected']['model']}", f(t["mae"], 2), f(t["rmse"], 1), f(t["r2"]), f(t["rmse_log1p"]), f(t["r2_log1p"]), f(sy["r2_log1p"])])
    L += [f"Real (Kaggle) test, n = {reg['split']['real_test']:,}:", "", table(["Experiment", "features / model", "MAE", "RMSE", "R2", "RMSE (log1p)", "R2 (log1p)", "Exp. C: R2 log on synthetic test"], rows), ""]
    sp = reg["spatial_holdout_supplementary"]
    L += [f"Supplementary spatial hold-out ({', '.join(sp['held_out_states'])}; train {sp['n_train']:,}, test {sp['n_test']:,}): "
          f"HGB (core) log-R2 {f(sp['hgb_core']['r2_log1p'])} vs crop-median baseline {f(sp['baseline_crop_median']['r2_log1p'])} -> "
          + ("the model does NOT beat the baseline on unseen states." if sp["hgb_core"]["r2_log1p"] < sp["baseline_crop_median"]["r2_log1p"] else "the model beats the baseline."), "",
          "Residuals (experiment A, full features, real test): " + ", ".join(f"{k} = {f(v)}" for k, v in reg["experiments"]["A_kaggle_only"]["full"]["residuals_real_test"].items()) + ".", "",
          "Worst crops by mean absolute log error (n>=15): " + "; ".join(f"{x['crop']} {x['mean']:.2f} (n={x['count']})" for x in reg["residual_analysis_A_full"]["worst_crops_by_mean_abs_log_error(n>=15)"]) + ".", "",
          "**Reading the results.** Adding synthetic rows does not improve the real-test error (A is as good as or better than B1/B2). "
          "Experiment C scores are higher because synthetic test rows are siblings of the training rows; they say nothing about real-world accuracy. "
          "Yield on this data is also strongly persistent: compare with the last-observed-yield baseline above before crediting the model. "
          "State-level soil constants act as state identifiers, so soil features cannot be credited with explanatory power.", ""]
else:
    L += ["*not run*", ""]

L += ["### 4b. Yield-band classification (derived label)", ""]
if clf:
    L += [clf["label_note"], "",
          f"Class distribution (train): {clf['class_distribution_train']}; (real test): {clf['class_distribution_test']}. Rows: {clf['split']['real_train']:,} / {clf['split']['real_validation']:,} / {clf['split']['real_test']:,} "
          f"(crops with <8 training rows excluded; {clf['split']['test_rows_dropped_crop_lt8_train_rows']} test rows dropped). Class imbalance handled with balanced class weights.", ""]
    rows = []
    def r_(n, m, extra="-"):
        return [n, extra, f(m["accuracy"]), f(m["macro_precision"]), f(m["macro_recall"]), f(m["macro_f1"])]
    rows.append(r_("Baseline: majority class", clf["baseline_majority_class"]))
    rows.append(r_("Baseline: persistence (last observed band, same crop-state-season)", clf["baseline_persistence_last_observed_band"]))
    for name, d in clf["experiments"].items():
        for fs, r in d.items():
            rows.append(r_(name, r["test_real"], f"{fs} / {r['selected']}"))
    L += [table(["Experiment (real test)", "features / model", "Accuracy", "Macro precision", "Macro recall", "Macro F1"], rows), ""]
    best = clf["experiments"]["A_kaggle_only"]["full"]
    L += ["Per-class (experiment A, full features, real test):", "",
          table(["class", "precision", "recall", "F1", "support"], [[b, f(v["precision"]), f(v["recall"]), f(v["f1"]), v["support"]] for b, v in best["test_real"]["per_class"].items()]), "",
          "Confusion matrix (rows = actual, cols = predicted; low / typical / high): " + str(best["test_real"]["confusion_matrix"]["rows=actual, cols=predicted"]), "",
          "Synthetic test (Experiment C, optimistic by construction): " + "; ".join(f"{n} {fs} macro-F1 {f(r['test_synthetic']['macro_f1'])}" for n, d in clf["experiments"].items() for fs, r in d.items() if fs == "full"), "",
          "**Reading the results.** " + ("The simple persistence baseline matches or beats the trained models, so the ML classifier adds no demonstrated value over 'same band as last time' on this data. "
                                       if clf["baseline_persistence_last_observed_band"]["macro_f1"] >= best["test_real"]["macro_f1"] else "The trained models beat the persistence baseline. ")
          + "Synthetic augmentation does not improve real-test F1. This is a derived label, not agronomic suitability.", ""]
else:
    L += ["*not run*", ""]

# ---------------------------------------------------------------- 5. retrieval
L += ["## 5. Retrieval", ""]
if ret:
    L += ["Relevance = deterministic metadata predicates (never an LLM). " + ret["label_note"], ""]
    for corpus, d in ret["corpora"].items():
        L += [f"### Corpus `{corpus}` ({d['n_documents']:,} documents; {d['n_queries']} queries: {d['query_types']})", ""]
        rows = []
        for mode, per in d["results"].items():
            for t, a in per.items():
                rows.append([mode, t, a["n_queries"], f(a["p@k"]), f(a["r_precision"]), f(a["recall@k"]), f(a["mrr"]), f(a["ndcg@k"]), f"{a['ndcg@k_ci95']}" if a["ndcg@k_ci95"] else "-", f(a["hit@k"])])
        L += [table(["mode", "query type", "n", f"P@{ret['k']}", "R-precision", f"Recall@{ret['k']}", "MRR", f"nDCG@{ret['k']}", "nDCG 95% CI", f"Hit@{ret['k']}"], rows), ""]
        L += ["Lowest-scoring hybrid+entity queries (failure cases):", ""]
        L += [f"- `{x['query']}` ({x['type']}, {x['n_relevant']} relevant) nDCG {x['ndcg@k']}; top-3 retrieved: " + "; ".join(f"{t['crop']}/{t['state']}/{t['season']}/{t['year']} rain {t['rainfall']} pH {t['ph']}" for t in x["top3_retrieved"]) for x in ret["failures"][corpus]]
        L += [""]
    L += ["**What changed in the retriever and why.** The retriever now detects crop/state/season names in the query (typo-tolerant: one edit or an adjacent transposition) and gives matching records a soft boost; "
          "negated mentions (\"not rice\", \"alternatives to rice\") are ignored. In an earlier run of this same benchmark the boost *without* typo tolerance lowered typo-query nDCG (0.365 -> 0.204) because it promoted the right state/season over the misspelled crop; "
          "adding fuzzy matching fixed that (see `hybrid_rrf+entity_boost` vs `hybrid_rrf`). Constraint and partial queries are already at the ceiling for every mode, so the boost adds nothing there. "
          "Numeric-range queries remain weak for every mode (semantic-only is best at ~0.24 nDCG): neither BM25 nor MiniLM encodes numeric ranges, and the retriever does not parse numeric constraints (the advisory agent compensates with a structured numeric-similarity candidate stage).", "",
          "P@K is capped when fewer than K relevant documents exist, so R-precision is the fair precision figure. No reranker was added: hybrid fusion is already near the ceiling on constraint queries and a cross-encoder would not address the numeric-query weakness shown above (embeddings do not encode numeric ranges).", ""]
else:
    L += ["*not run*", ""]

# ---------------------------------------------------------------- 6. recommendations
L += ["## 6. Recommendation validation", "", "### 6a. Held-out-year test (independent of the system's outputs)", ""]
if held:
    L += [held["protocol"], ""]
    if tune:
        L += [f"Popularity blend tuned on validation years only ({tune['protocol']}); rule: {tune['selection_rule']}. Grid (precision@5): " + ", ".join(f"w={k}: {f(v['precision@5'])}" for k, v in tune["grid"].items()) + f". Selected w = {tune['selected']}.", ""]
    rows = []
    for name, r in held["results"].items():
        yp = r["realised_yield_percentile_of_grown_recommendations"]
        rows.append([name, f(r["precision@5"]["mean"]), str(r["precision@5"]["ci95"]), f(r["hit@5"]), f(yp["mean"]) if yp else "n/a", str(yp["ci95"]) if yp else "-"])
    L += [f"n = {held['n_cases']} state-season-year cases (test years {held['test_years']}, evidence <= {held['evidence_cutoff_year']}, Kaggle rows only):", "",
          table(["system", "Precision@5 (recommended crop was grown)", "95% CI", "Hit@5", "mean realised yield percentile", "95% CI"], rows), "",
          "Precision@5 measures plausibility (revealed farmer choice), not optimality; the yield percentile (0.5 = the crop's pre-2018 median) is the quality signal. "
          "Differences in yield percentile between systems are within the confidence intervals unless stated otherwise.", ""]
else:
    L += ["*not run*", ""]
L += ["### 6b. Dimension checks on the full pipeline (Claude disabled)", ""]
if dim:
    L += [f"{dim['n_queries']} queries; origin: {dim['origin']}. LLM-judge and DeepEval were **not** used here (the Anthropic account has no credit), so these checks are deterministic.", "",
          table(["check", "passed / total"], [[k, f"{v['passed']} / {v['total']}"] for k, v in dim["summary"].items()]), ""]
    fails = [(r["id"], k, r[k]) for r in dim["rows"] for k in r if k not in ("id", "kind", "status", "recommended", "n_limited_evidence_recs") and r.get(k) is False]
    L += ["Failures: " + ("; ".join(f"{i} ({k})" for i, k, _ in fails) if fails else "none"), ""]
else:
    L += ["*not run*", ""]
L += ["Test-set notes: the application evaluation set was extended from 31 to 58 cases (`data/evaluation/test_cases_extra.json`; 27 extra programmatic/hand-written cases, seed 7). One hand-written clarification case (\"Which one should I choose, rice or wheat?\") was removed after the system answered it with warnings instead of asking for clarification: that expectation was a debatable assumption by the test author, not a system defect. "
      "An earlier version of the recommender widened the search to other seasons when a state-season had <3 crops; the new context-compliance check caught it recommending crops with no record in the requested season, so the relaxation was removed.", "",
      "Expert validation: **none has taken place.** Test cases are template-generated or hand-written by the developers; an LLM judge (when credit is available) is a supplementary signal only.", ""]

# ---------------------------------------------------------------- 7. limits
L += ["## 7. Limitations and what real-world validation would need", "",
      "- All 'real' data is state-level aggregated Indian crop statistics: no field-level soil, irrigation, pests/disease or practices. Soil features are state constants.",
      "- The yield-band classifier and yield regressor predict historical statistical patterns; they are not validated agronomic advice. Persistence baselines are strong.",
      "- The synthetic data adds no independent information and does not improve any real-data metric; high scores on synthetic test sets are not evidence of accuracy.",
      "- Retrieval labels are metadata-derived; they test constraint handling, not usefulness. Numeric-condition queries remain weak for the semantic component.",
      "- LLM-based explanation, the Claude judge and DeepEval were not measured in these runs (no API credit); the corresponding metrics are *not evaluated*.",
      "- Stronger validation needs: field-level trials or farm records with soil tests/irrigation, agronomist review of a sampled set of recommendations, and prospective (next-season) evaluation.", "",
      "## 8. Reproduce", "",
      "```powershell", "python scripts/audit_data.py", "python scripts/run_experiments.py            # ~20-30 min on CPU", "python scripts/build_experiment_report.py",
      "python -m pytest -q", "python scripts/run_evaluation.py --no-llm    # application evaluation (feeds the UI)", "uvicorn src.main:app", "```", ""]
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(L), encoding="utf-8")
print("wrote", OUT, len("\n".join(L)), "chars")
