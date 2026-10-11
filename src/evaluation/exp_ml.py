"""Leak-safe ML experiments: yield regression and yield-band classification.

Experiment A  Kaggle-only      train = Kaggle rows with year <= 2014
Experiment B  Kaggle+synthetic train = A + synthetic rows generated ONLY from the A training rows (generator v2); same real test set
Experiment C  synthetic test   held-out synthetic rows (generated from the A training rows with a different seed, signature-disjoint
                               from the B training rows). Reported separately and NEVER mixed with the real test result.

Real test = Kaggle rows with year >= 2018 (never used for fitting, tuning, selection, thresholds or synthetic generation).
Validation = Kaggle rows 2015-2017 (model selection only). Final models are refit on train(+synthetic)+validation.
Everything is seeded (SEED) and reported with exact sample counts.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from ..config import settings
from ..models.yield_model import CATS, FEATURE_SETS, CropMedianBaseline, _frame, _metrics, _pipeline
from ..services.data_prep import GENERATOR_V2_RULES, generate_synthetic

SEED = settings.random_seed
TRAIN_END, VAL_END = 2014, 2017
BANDS = ["low", "typical", "high"]
REG_CANDIDATES = {"hgb": dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0),
                  "rf": dict(n_estimators=120, min_samples_leaf=2)}


def load_real() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(settings.cases_path)
    if "data_source" not in df:
        df["data_source"] = np.where(df["is_synthetic"].astype(bool), "synthetic", "kaggle")
    real = df[df["data_source"] == "kaggle"]
    real = real.reset_index(drop=True)
    weather = real[["state", "year", "temperature", "rainfall", "humidity"]].dropna().drop_duplicates(["state", "year"])
    return real, weather


def split_real(real: pd.DataFrame):
    tr = real[real["year"] <= TRAIN_END].reset_index(drop=True)
    va = real[(real["year"] > TRAIN_END) & (real["year"] <= VAL_END)].reset_index(drop=True)
    te = real[real["year"] > VAL_END].reset_index(drop=True)
    return tr, va, te


def build_synthetic_sets(tr: pd.DataFrame, weather: pd.DataFrame):
    """Training synthetic (two sizes) and an independent held-out synthetic test set; all derived from `tr` only."""
    syn_big = generate_synthetic(tr, 2 * len(tr), SEED, weather)
    syn_1x, syn_2x = syn_big.head(len(tr)), syn_big
    cand = generate_synthetic(tr, 4000, SEED + 101, weather)
    sig = lambda d: set(zip(d["crop"], d["state"], d["season"], d["year"], d["area"].round(1), d["yield"].round(4)))  # noqa: E731
    used = sig(syn_2x)
    syn_test = cand[[s not in used for s in zip(cand["crop"], cand["state"], cand["season"], cand["year"], cand["area"].round(1), cand["yield"].round(4))]].reset_index(drop=True)
    return syn_1x.reset_index(drop=True), syn_2x.reset_index(drop=True), syn_test


def _counts(**frames) -> dict:
    return {k: int(len(v)) for k, v in frames.items()}


# ======================================================================================== regression
def _select_and_eval(cfg_name: str, train: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame, syn_test: pd.DataFrame, fs: str) -> dict:
    cols = FEATURE_SETS[fs]
    f = lambda d: _frame(d)  # noqa: E731
    Xtr, Xva, Xte, Xsy = f(train), f(va), f(te), f(syn_test)
    ytr, yva, yte, ysy = (np.log1p(d["yield"]) for d in (Xtr, Xva, Xte, Xsy))
    out: dict = {"n_train_rows": int(len(train)), "feature_set": fs, "candidates": {}}
    val_scores = {}
    for name, kw in REG_CANDIDATES.items():
        t0 = time.time()
        pipe = _pipeline(name, cols, **kw).fit(Xtr[cols], ytr)
        m = _metrics(yva, pipe.predict(Xva[cols]))
        val_scores[name] = m["rmse_log1p"]
        out["candidates"][name] = {"val_real": m, "fit_s": round(time.time() - t0, 1)}
    best = min(val_scores, key=val_scores.get)
    # final: refit on train (+synthetic, if any) + real validation; evaluate once on the real test and the synthetic test
    trva_X = pd.concat([Xtr, Xva], ignore_index=True)
    final = _pipeline(best, cols, **REG_CANDIDATES[best]).fit(trva_X[cols], np.log1p(trva_X["yield"]))
    p_te, p_sy = final.predict(Xte[cols]), final.predict(Xsy[cols])
    out["selected"] = {"model": best, "selection_metric": "validation (real, 2015-2017) RMSE on log1p(yield)", "refit_on_rows": int(len(trva_X))}
    out["test_real"] = {**_metrics(yte, p_te), "n": int(len(Xte))}
    out["test_synthetic"] = {**_metrics(ysy, p_sy), "n": int(len(Xsy)), "note": "optimistic by construction: siblings of the training rows"}
    res = p_te - yte.to_numpy()
    out["_pred"] = (p_te, yte.to_numpy(), Xte)
    out["residuals_real_test"] = {
        "mean_log_residual(bias)": float(res.mean()), "median_abs_log_residual": float(np.median(np.abs(res))),
        "p05": float(np.quantile(res, .05)), "p95": float(np.quantile(res, .95)),
        "corr_actual_pred_log": float(np.corrcoef(p_te, yte.to_numpy())[0, 1]),
        "share_within_20pct": float(np.mean(np.abs(np.expm1(p_te) - np.expm1(yte.to_numpy())) <= 0.2 * np.clip(np.expm1(yte.to_numpy()), 1e-9, None))),
    }
    return out


def run_regression() -> dict:
    real, weather = load_real()
    tr, va, te = split_real(real)
    syn_1x, syn_2x, syn_test = build_synthetic_sets(tr, weather)
    result = {
        "task": "yield regression (target: `yield`, source units, crop-specific; fitted on log1p)",
        "split": {"strategy": f"temporal, Kaggle rows only: train <= {TRAIN_END}, validation {TRAIN_END + 1}-{VAL_END}, test > {VAL_END}",
                  **_counts(real_train=tr, real_validation=va, real_test=te, synthetic_train_1x=syn_1x, synthetic_train_2x=syn_2x,
                            synthetic_test=syn_test), "seed": SEED},
        "synthetic_generator": {"version": "v2 (train-only templates)", "rules": GENERATOR_V2_RULES},
        "experiments": {},
    }
    base = CropMedianBaseline().fit(pd.concat([tr, va])[["crop"]], np.log1p(pd.concat([tr, va])["yield"]))
    yte = np.log1p(te["yield"])
    result["baseline_crop_median"] = {"test_real": _metrics(yte, base.predict(te[["crop"]])),
                                      "test_synthetic": _metrics(np.log1p(syn_test["yield"]), base.predict(syn_test[["crop"]]))}
    # persistence baseline: last observed yield of the same crop-state-season (train+validation), else the crop median
    last = pd.concat([tr, va]).sort_values("year").groupby(["crop", "state", "season"])["yield"].last()
    pers = np.array([last.get(k, np.nan) for k in zip(te["crop"], te["state"], te["season"])])
    med = np.expm1(base.predict(te[["crop"]]))
    pers = np.where(np.isnan(pers), med, pers)
    result["baseline_persistence_last_observed_yield"] = {"test_real": _metrics(yte, np.log1p(pers)),
                                                          "share_test_rows_with_history": float(np.mean([k in last.index for k in zip(te["crop"], te["state"], te["season"])]))}
    configs = {"A_kaggle_only": tr, "B1_kaggle_plus_1x_synthetic": pd.concat([tr, syn_1x], ignore_index=True),
               "B2_kaggle_plus_2x_synthetic": pd.concat([tr, syn_2x], ignore_index=True)}
    pred_store = {}
    for name, train in configs.items():
        result["experiments"][name] = {}
        for fs in ("core", "full"):
            r = _select_and_eval(name, train, va, te, syn_test, fs)
            pred_store[(name, fs)] = r.pop("_pred")
            result["experiments"][name][fs] = r
    # residual detail for the experiment-A full model
    p, y, X = pred_store[("A_kaggle_only", "full")]
    d = pd.DataFrame({"crop": X["crop"].values, "year": X["year"].values, "abs_log": np.abs(p - y), "res": p - y})
    result["residual_analysis_A_full"] = {
        "by_year": d.groupby("year")["abs_log"].agg(["mean", "count"]).round(4).reset_index().to_dict("records"),
        "worst_crops_by_mean_abs_log_error(n>=15)": d.groupby("crop")["abs_log"].agg(["mean", "count"]).query("count >= 15").sort_values("mean", ascending=False).head(8).round(4).reset_index().to_dict("records"),
        "actual_vs_predicted_sample(150)": [{"actual": round(float(np.expm1(a)), 4), "predicted": round(float(np.expm1(b)), 4), "crop": c}
                                           for a, b, c in zip(y[:150 * 14:14], p[:150 * 14:14], X["crop"].values[:150 * 14:14])],
    }
    # spatial hold-out (supplementary): unseen states, all years
    rng = np.random.default_rng(SEED)
    states = sorted(real["state"].unique())
    held = list(rng.choice(states, size=6, replace=False))
    rtr, rte = real[~real["state"].isin(held)], real[real["state"].isin(held)]
    cols = FEATURE_SETS["core"]
    pipe = _pipeline("hgb", cols, **REG_CANDIDATES["hgb"]).fit(_frame(rtr)[cols], np.log1p(rtr["yield"]))
    cm = CropMedianBaseline().fit(rtr[["crop"]], np.log1p(rtr["yield"]))
    result["spatial_holdout_supplementary"] = {
        "held_out_states": held, "n_train": int(len(rtr)), "n_test": int(len(rte)),
        "hgb_core": _metrics(np.log1p(rte["yield"]), pipe.predict(_frame(rte)[cols])),
        "baseline_crop_median": _metrics(np.log1p(rte["yield"]), cm.predict(rte[["crop"]])),
        "note": "state is an unseen category here; soil N/P/K/pH (state constants) are the only state descriptors the model can use"}
    return result


# ======================================================================================== classification
def _clf_pipe(kind: str, cols: list[str]) -> Pipeline:
    cats = [c for c in cols if c in CATS]
    pre = ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), cats)],
                            remainder="passthrough", verbose_feature_names_out=False)
    pre.set_output(transform="pandas")
    if kind == "hgb":
        est = HistGradientBoostingClassifier(categorical_features=[True] * len(cats) + [False] * (len(cols) - len(cats)), max_iter=300,
                                             learning_rate=0.06, class_weight="balanced", random_state=SEED)
    elif kind == "rf":
        est = RandomForestClassifier(n_estimators=150, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
    else:
        est = DummyClassifier(strategy="most_frequent")
    return Pipeline([("pre", pre), ("est", est)])


def _band(df: pd.DataFrame, thr: pd.DataFrame) -> pd.Series:
    t = df[["crop"]].merge(thr, left_on="crop", right_index=True, how="left")
    y = df["yield"].to_numpy()
    lab = np.where(y <= t["q25"].to_numpy(), "low", np.where(y > t["q75"].to_numpy(), "high", "typical"))
    return pd.Series(np.where(t["q25"].isna(), None, lab), index=df.index)


def _clf_metrics(y_true, y_pred) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=BANDS, zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "macro_f1": float(f1_score(y_true, y_pred, labels=BANDS, average="macro", zero_division=0)),
            "macro_precision": float(p.mean()), "macro_recall": float(r.mean()),
            "per_class": {b: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])} for i, b in enumerate(BANDS)},
            "confusion_matrix": {"labels": BANDS, "rows=actual, cols=predicted": confusion_matrix(y_true, y_pred, labels=BANDS).tolist()}}


def run_classification() -> dict:
    real, weather = load_real()
    tr, va, te = split_real(real)
    syn_1x, syn_2x, syn_test = build_synthetic_sets(tr, weather)
    cnt = tr.groupby("crop")["yield"].agg(["count", lambda s: s.quantile(.25), lambda s: s.quantile(.75)])
    cnt.columns = ["n", "q25", "q75"]
    thr = cnt[cnt["n"] >= 8][["q25", "q75"]]
    dropped = int((~te["crop"].isin(thr.index)).sum())
    sets = {k: d[d["crop"].isin(thr.index)].reset_index(drop=True) for k, d in
            dict(tr=tr, va=va, te=te, s1=syn_1x, s2=syn_2x, st=syn_test).items()}
    for d in sets.values():
        d["band"] = _band(d, thr)
    result = {
        "task": "yield-band classification (low <= crop's training 25th pct < typical <= 75th pct < high)",
        "label_note": ("The label is DERIVED from yield with thresholds fitted on training rows only. No ground-truth agronomic suitability label "
                       "exists in the data, so this is NOT a suitability classifier; it predicts whether a crop-state-season-year is likely to fall in "
                       "the crop's low/typical/high historical yield band."),
        "split": {**_counts(real_train=sets["tr"], real_validation=sets["va"], real_test=sets["te"], synthetic_train_1x=sets["s1"],
                            synthetic_train_2x=sets["s2"], synthetic_test=sets["st"]), "test_rows_dropped_crop_lt8_train_rows": dropped, "seed": SEED},
        "class_distribution_train": sets["tr"]["band"].value_counts().to_dict(),
        "class_distribution_test": sets["te"]["band"].value_counts().to_dict(),
        "experiments": {},
    }
    # baselines on the real test: majority, and persistence (last observed band of the same crop-state-season group)
    maj = _clf_pipe("dummy", ["crop"]).fit(sets["tr"][["crop"]], sets["tr"]["band"])
    result["baseline_majority_class"] = _clf_metrics(sets["te"]["band"], maj.predict(sets["te"][["crop"]]))
    hist = pd.concat([sets["tr"], sets["va"]]).sort_values("year").groupby(["crop", "state", "season"])["band"].last()
    keys = list(zip(sets["te"]["crop"], sets["te"]["state"], sets["te"]["season"]))
    pers = pd.Series([hist.get(k, "typical") for k in keys])
    result["baseline_persistence_last_observed_band"] = {**_clf_metrics(sets["te"]["band"], pers),
                                                         "share_test_rows_with_history": float(np.mean([k in hist.index for k in keys]))}
    configs = {"A_kaggle_only": sets["tr"], "B1_kaggle_plus_1x_synthetic": pd.concat([sets["tr"], sets["s1"]], ignore_index=True),
               "B2_kaggle_plus_2x_synthetic": pd.concat([sets["tr"], sets["s2"]], ignore_index=True)}
    for name, train in configs.items():
        result["experiments"][name] = {}
        for fs in ("core", "full"):
            cols = FEATURE_SETS[fs]
            Xtr, Xva, Xte, Xst = (_frame(d) for d in (train, sets["va"], sets["te"], sets["st"]))
            sel = {}
            for kind in ("hgb", "rf"):
                pipe = _clf_pipe(kind, cols).fit(Xtr[cols], train["band"])
                sel[kind] = f1_score(sets["va"]["band"], pipe.predict(Xva[cols]), labels=BANDS, average="macro", zero_division=0)
            best = max(sel, key=sel.get)
            trva = pd.concat([Xtr, Xva], ignore_index=True)
            final = _clf_pipe(best, cols).fit(trva[cols], pd.concat([train["band"], sets["va"]["band"]], ignore_index=True))
            result["experiments"][name][fs] = {
                "validation_macro_f1_by_model": {k: float(v) for k, v in sel.items()}, "selected": best,
                "test_real": _clf_metrics(sets["te"]["band"], final.predict(Xte[cols])),
                "test_synthetic": {**_clf_metrics(sets["st"]["band"], final.predict(Xst[cols])), "note": "optimistic by construction"},
            }
    return result
