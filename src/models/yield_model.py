"""Crop yield regression: training + inference.

Target: `yield` (source units, crop-specific; ~production/area). Modelled as log1p(yield) because yields differ
by orders of magnitude between crops; metrics are reported on BOTH the original scale (MAE/RMSE/R2) and log scale.

Leakage control: `production` is never a feature (yield ~ production/area). `area` and per-hectare inputs are only
used in the 'full' feature set; the 'core' set (crop, season, state, year, soil, weather) is used when the user has
not supplied area/inputs. Temporal split: train <= 2014, validation 2015-2017, test >= 2018.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from ..config import settings

CATS = ["crop", "season", "state"]
CORE_NUM = ["year", "n", "p", "k", "ph", "temperature", "rainfall", "humidity"]
FULL_EXTRA = ["log_area", "fertilizer_per_ha", "pesticide_per_ha"]
FEATURE_SETS = {"core": CATS + CORE_NUM, "full": CATS + CORE_NUM + FULL_EXTRA}
TRAIN_END, VAL_END = 2014, 2017
SEED = settings.random_seed


class ModelUnavailable(RuntimeError):
    pass


def _frame(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["log_area"] = np.log1p(x["area"]) if "area" in x else np.nan
    return x


def _pipeline(kind: str, cols: list[str], **kw) -> Pipeline:
    cats = [c for c in cols if c in CATS]
    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), cats)],
        remainder="passthrough", verbose_feature_names_out=False)
    pre.set_output(transform="pandas")
    if kind == "hgb":
        est = HistGradientBoostingRegressor(categorical_features=[True] * len(cats) + [False] * (len(cols) - len(cats)),
                                            random_state=SEED, **kw)
    elif kind == "rf":
        est = RandomForestRegressor(**{"n_estimators": 250, "min_samples_leaf": 2, "n_jobs": -1, "random_state": SEED, **kw})
    else:
        raise ValueError(kind)
    return Pipeline([("pre", pre), ("est", est)])


class CropMedianBaseline:
    """Baseline: median log-yield of the crop in the training data (global median for unseen crops)."""
    def fit(self, X, y):
        self.med = pd.Series(y.values, index=X["crop"].values).groupby(level=0).median()
        self.glob = float(np.median(y))
        return self

    def predict(self, X):
        return X["crop"].map(self.med).fillna(self.glob).to_numpy()


def _metrics(y_true_log, y_pred_log) -> dict:
    yt, yp = np.expm1(y_true_log), np.clip(np.expm1(y_pred_log), 0, None)
    return {
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "r2": float(r2_score(yt, yp)),
        "mae_log1p": float(mean_absolute_error(y_true_log, y_pred_log)),
        "rmse_log1p": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "r2_log1p": float(r2_score(y_true_log, y_pred_log)),
    }


def train(cases_path: Path | None = None, out_dir: Path | None = None) -> dict:
    cases_path = Path(cases_path or settings.cases_path)
    out_dir = Path(out_dir or settings.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(cases_path)
    if "is_synthetic" in df:
        df = df[~df["is_synthetic"].astype(bool)].reset_index(drop=True)   # train/evaluate on REAL records only
    rep = json.loads(settings.quality_report_path.read_text()) if settings.quality_report_path.exists() else {}
    df = _frame(df)
    y = np.log1p(df["yield"])
    tr, va, te = df["year"] <= TRAIN_END, (df["year"] > TRAIN_END) & (df["year"] <= VAL_END), df["year"] > VAL_END
    result = {"split": {"strategy": f"temporal: train year<={TRAIN_END}, validation {TRAIN_END + 1}-{VAL_END}, test >{VAL_END}",
                        "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum())},
              "feature_sets": FEATURE_SETS, "target": "yield (log1p-transformed for fitting)", "comparison": {}}
    bundle = {"models": {}, "schema": FEATURE_SETS}
    candidates = {"hgb": dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0),
                  "rf": {}}
    for fs, cols in FEATURE_SETS.items():
        comp = {}
        # baseline
        b = CropMedianBaseline().fit(df.loc[tr, cols], y[tr])
        comp["baseline_crop_median"] = {"val": _metrics(y[va], b.predict(df.loc[va, cols]))}
        fitted = {}
        for name, kw in candidates.items():
            t0 = time.time()
            pipe = _pipeline(name, cols, **kw).fit(df.loc[tr, cols], y[tr])
            comp[name] = {"val": _metrics(y[va], pipe.predict(df.loc[va, cols])), "fit_seconds": round(time.time() - t0, 1)}
            fitted[name] = pipe
        best = min(candidates, key=lambda n: comp[n]["val"]["rmse_log1p"])
        # refit selected model on train+validation, evaluate once on the held-out test years
        trva = tr | va
        final = _pipeline(best, cols, **candidates[best]).fit(df.loc[trva, cols], y[trva])
        pred_te = final.predict(df.loc[te, cols])
        test_m = _metrics(y[te], pred_te)
        base_te = CropMedianBaseline().fit(df.loc[trva, cols], y[trva])
        comp["baseline_crop_median"]["test"] = _metrics(y[te], base_te.predict(df.loc[te, cols]))
        # empirical 10-90% interval via quantile gradient boosting (log scale)
        q = {}
        for a in (0.1, 0.9):
            qp = _pipeline("hgb", cols, loss="quantile", quantile=a, max_iter=250, learning_rate=0.06).fit(df.loc[trva, cols], y[trva])
            q[a] = qp
        lo, hi = q[0.1].predict(df.loc[te, cols]), q[0.9].predict(df.loc[te, cols])
        coverage = float(np.mean((y[te] >= np.minimum(lo, hi)) & (y[te] <= np.maximum(lo, hi))))
        # permutation importance on test (log-scale RMSE), original columns
        pi = permutation_importance(final, df.loc[te, cols], y[te], n_repeats=5, random_state=SEED,
                                    scoring="neg_root_mean_squared_error", n_jobs=1)
        imp = sorted(({"feature": c, "importance": float(m), "std": float(s)} for c, m, s in
                      zip(cols, pi.importances_mean, pi.importances_std)), key=lambda d: -d["importance"])
        comp["selected"] = {"name": best, "selection_metric": "validation RMSE on log1p(yield)",
                            "refit_on": "train+validation", "test": test_m,
                            "interval_10_90": {"method": "HistGradientBoosting quantile regression (log scale)",
                                               "empirical_test_coverage": coverage, "nominal_coverage": 0.8}}
        result["comparison"][fs] = comp
        result.setdefault("permutation_importance", {})[fs] = imp
        bundle["models"][fs] = {"model": final, "q10": q[0.1], "q90": q[0.9], "name": best}

    # per-crop test errors for the full model (scale-free, informative)
    fm = bundle["models"]["full"]["model"]
    pt = fm.predict(df.loc[te, FEATURE_SETS["full"]])
    err = pd.DataFrame({"crop": df.loc[te, "crop"].values, "ae_log": np.abs(pt - y[te].values)})
    result["per_crop_test_mae_log1p_top"] = (err.groupby("crop")["ae_log"].agg(["mean", "count"])
                                             .sort_values("count", ascending=False).head(15)
                                             .reset_index().rename(columns={"mean": "mae_log1p"}).to_dict("records"))
    version = "yield-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    meta = {
        "model_version": version, "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": rep.get("dataset_version", "unknown"), "n_rows": int(len(df)), "seed": SEED,
        "units": "same unit as the source 'Yield' column for the given crop (typically tonnes/hectare; crop-specific)",
        "limitations": [
            "Soil N/P/K/pH are state constants, so they act as state identifiers; importance is confounded with state.",
            "Weather is a state-year aggregate; no farm-level data (soil type, irrigation, variety) is available.",
            "Original-scale R2/RMSE are dominated by a few high-magnitude crops (e.g. coconut); see log1p metrics.",
            "Predictions are estimates for decision support, not guarantees.",
        ],
    }
    bundle["meta"] = meta
    joblib.dump(bundle, out_dir / "yield_model.joblib")
    result["meta"] = meta
    (out_dir / "yield_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


# --------------------------------------------------------------------------------- inference
class YieldPredictor:
    def __init__(self, path: Path | None = None):
        path = Path(path or settings.model_dir / "yield_model.joblib")
        if not path.exists():
            raise ModelUnavailable(f"Yield model not found at {path}. Run: python scripts/train_yield_model.py")
        self.bundle = joblib.load(path)
        self.meta = self.bundle["meta"]
        mp = path.parent / "yield_metrics.json"
        self.metrics = json.loads(mp.read_text()) if mp.exists() else {}

    def predict(self, inp: dict) -> dict:
        crop, state, season = inp.get("crop"), inp.get("state"), inp.get("season")
        if not crop or not state or not season:
            raise ValueError("crop, state and season are required for yield prediction")
        row = {"crop": crop, "state": state, "season": season, "year": inp.get("year") or 2020}
        for k in ["n", "p", "k", "ph", "temperature", "rainfall", "humidity"]:
            row[k] = inp.get(k) if inp.get(k) is not None else np.nan
        has_full = inp.get("area") and inp.get("fertilizer") is not None and inp.get("pesticide") is not None
        fs = "full" if has_full else "core"
        if has_full:
            row["area"] = float(inp["area"])
            row["fertilizer_per_ha"] = float(inp["fertilizer"]) / float(inp["area"])
            row["pesticide_per_ha"] = float(inp["pesticide"]) / float(inp["area"])
        X = _frame(pd.DataFrame([row]))
        cols = FEATURE_SETS[fs]
        for c in cols:
            if c not in X:
                X[c] = np.nan
        m = self.bundle["models"][fs]
        p = float(np.clip(np.expm1(m["model"].predict(X[cols])[0]), 0, None))
        lo, hi = np.expm1(m["q10"].predict(X[cols])[0]), np.expm1(m["q90"].predict(X[cols])[0])
        lo, hi = sorted([float(max(lo, 0)), float(max(hi, 0))])
        cov = self.metrics.get("comparison", {}).get(fs, {}).get("selected", {}).get("interval_10_90", {}).get("empirical_test_coverage")
        unseen = [n for n, v in (("crop", crop), ("state", state), ("season", season)) if v not in self._known(n)]
        return {
            "predicted_yield": round(p, 4), "unit": self.meta["units"], "is_estimate": True,
            "interval_10_90": {"low": round(lo, 4), "high": round(max(hi, p if hi < p else hi), 4),
                               "method": "quantile gradient boosting (log scale), empirical coverage on held-out test years: "
                                         + (f"{cov:.0%} (nominal 80%)" if cov is not None else "unknown")},
            "feature_set": fs, "model_name": m["name"], "model_version": self.meta["model_version"],
            "assumptions": ([f"year assumed {row['year']}"] if not inp.get("year") else [])
                           + ([f"{k} not supplied (left missing)" for k in ["n", "p", "k", "ph", "temperature", "rainfall", "humidity"] if np.isnan(row[k])]),
            "unseen_categories": unseen,
            "limitations": self.meta["limitations"],
            "test_metrics": self.metrics.get("comparison", {}).get(fs, {}).get("selected", {}).get("test"),
        }

    def _known(self, name: str):
        enc = self.bundle["models"]["core"]["model"].named_steps["pre"].named_transformers_["cat"]
        i = CATS.index(name)
        return set(enc.categories_[i])
