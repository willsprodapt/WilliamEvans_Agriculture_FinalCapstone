"""In-memory structured case store backed by data/processed/cases.csv (single source of truth for records)."""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import settings
from ..schemas.common import RANGE_FIELDS, CaseFilters

PUBLIC_COLUMNS = [
    "case_id", "source_row", "crop", "year", "season", "state", "area", "production", "yield", "fertilizer", "pesticide",
    "n", "p", "k", "ph", "temperature", "rainfall", "humidity", "fertilizer_per_ha", "pesticide_per_ha",
    "yield_band", "ph_class", "rainfall_class", "temp_class", "humidity_class", "completeness", "flags", "is_synthetic", "data_source",
]
SORTABLE = ["year", "yield", "area", "production", "fertilizer", "pesticide", "rainfall", "temperature", "humidity", "ph", "n", "p", "k", "crop", "state"]


def _clean(v):
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if (isinstance(v, float) and math.isnan(v)) or (isinstance(v, np.floating) and np.isnan(v)) else float(v)
    return v


class DatasetUnavailable(RuntimeError):
    pass


class CaseStore:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or settings.cases_path)
        if not self.path.exists():
            raise DatasetUnavailable(f"Processed dataset not found at {self.path}. Run: python scripts/prepare_dataset.py")
        self.df = pd.read_csv(self.path)
        self.df["flags"] = self.df["flags"].fillna("")
        if "is_synthetic" not in self.df:
            self.df["is_synthetic"] = False
        self.df["is_synthetic"] = self.df["is_synthetic"].astype(bool)
        # provenance label ("kaggle" = observed record from the source file, "synthetic" = generated); derived when the CSV predates the column
        if "data_source" not in self.df:
            self.df["data_source"] = np.where(self.df["is_synthetic"], "synthetic", "kaggle")
        self.real_df = self.df[~self.df["is_synthetic"]]      # statistics, suitability, models use REAL rows only
        self.id_to_idx = {cid: i for i, cid in enumerate(self.df["case_id"])}
        self._lower = {c: self.df[c].astype(str).str.lower() for c in ["crop", "season", "state"]}
        rp = settings.quality_report_path
        self.report = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else {}
        self.dataset_version = self.report.get("dataset_version", "unknown")
        # per-field spread used for similarity scaling (2 x std)
        self.scale = {c: float(2 * self.df[c].std()) or 1.0 for c in
                      ["n", "p", "k", "ph", "rainfall", "temperature", "humidity"]}

    def __len__(self):
        return len(self.df)

    # ----- records -------------------------------------------------------------------
    def record(self, idx: int, include_text: bool = False) -> dict:
        row = self.df.iloc[idx]
        cols = PUBLIC_COLUMNS + (["text"] if include_text else [])
        return {c: _clean(row[c]) for c in cols}

    def get(self, case_id: str, include_text: bool = True) -> dict | None:
        i = self.id_to_idx.get(case_id)
        return None if i is None else self.record(i, include_text)

    # ----- filtering -----------------------------------------------------------------
    def mask(self, filters: CaseFilters | None) -> np.ndarray:
        m = np.ones(len(self.df), dtype=bool)
        if filters is None:
            return m
        for col in ("crop", "season", "state"):
            vals = getattr(filters, col)
            if vals:
                m &= self._lower[col].isin([v.strip().lower() for v in vals]).to_numpy()
        if getattr(filters, "origin", None):
            want = {o.lower() for o in filters.origin}
            m &= np.where(self.df["is_synthetic"].to_numpy(), "synthetic" in want, "real" in want)
        if filters.year_min is not None:
            m &= (self.df["year"] >= filters.year_min).to_numpy()
        if filters.year_max is not None:
            m &= (self.df["year"] <= filters.year_max).to_numpy()
        for name in RANGE_FIELDS:
            lo, hi = getattr(filters, f"{name}_min"), getattr(filters, f"{name}_max")
            if lo is not None:   # NaN comparisons are False -> missing values never satisfy a numeric filter
                m &= (self.df[name] >= lo).to_numpy()
            if hi is not None:
                m &= (self.df[name] <= hi).to_numpy()
        return m

    def list_cases(self, filters: CaseFilters | None, sort_by: str = "year", order: str = "desc",
                   page: int = 1, page_size: int = 25, text: str | None = None) -> dict:
        if sort_by not in SORTABLE:
            raise ValueError(f"sort_by must be one of {SORTABLE}")
        m = self.mask(filters)
        sub = self.df[m]
        if text:
            t = text.strip().lower()
            sub = sub[(sub["crop"].str.lower().str.contains(t, regex=False)) | (sub["state"].str.lower().str.contains(t, regex=False))
                      | (sub["season"].str.lower().str.contains(t, regex=False)) | (sub["case_id"].str.lower() == t)]
        sub = sub.sort_values(sort_by, ascending=(order == "asc"), na_position="last")
        total = len(sub)
        page_df = sub.iloc[(page - 1) * page_size: page * page_size]
        items = [self.record(self.id_to_idx[cid]) for cid in page_df["case_id"]]
        return {"items": items, "total": total, "page": page, "page_size": page_size,
                "pages": max(1, math.ceil(total / page_size))}

    # ----- metadata -----------------------------------------------------------------
    def options(self) -> dict:
        d = self.df
        ranges = {c: {"min": _clean(d[c].min()), "max": _clean(d[c].max())} for c in RANGE_FIELDS if c in d.columns}
        return {
            "crops": sorted(d["crop"].unique().tolist()),
            "states": sorted(d["state"].unique().tolist()),
            "seasons": sorted(d["season"].unique().tolist()),
            "years": {"min": int(d["year"].min()), "max": int(d["year"].max())},
            "ranges": ranges,
            "origins": ["real", "synthetic"], "unavailable_fields": ["soil_type", "irrigation", "farm_location", "latitude", "longitude"],
            "notes": "N, P, K, pH are state-level constants; temperature/rainfall/humidity are state-year aggregates.",
        }

    def stats(self) -> dict:
        d = self.df
        top_crops = d["crop"].value_counts().head(15)
        return {
            "records": int(len(d)), "real_records": int((~d["is_synthetic"]).sum()), "synthetic_records": int(d["is_synthetic"].sum()), "crops": int(d["crop"].nunique()), "states": int(d["state"].nunique()),
            "seasons": int(d["season"].nunique()), "years": [int(d["year"].min()), int(d["year"].max())],
            "crop_counts": [{"name": k, "count": int(v)} for k, v in d["crop"].value_counts().items()],
            "state_counts": [{"name": k, "count": int(v)} for k, v in d["state"].value_counts().items()],
            "season_counts": [{"name": k, "count": int(v)} for k, v in d["season"].value_counts().items()],
            "yearly_counts": [{"year": int(k), "count": int(v)} for k, v in d["year"].value_counts().sort_index().items()],
            "median_yield_by_crop": [{"crop": k, "median_yield": float(v)} for k, v in
                                     d.groupby("crop")["yield"].median().loc[top_crops.index].items()],
            "dataset_version": self.dataset_version,
        }


_store: CaseStore | None = None


def get_store() -> CaseStore:
    global _store
    if _store is None:
        _store = CaseStore()
    return _store


def reset_store():
    global _store
    _store = None
